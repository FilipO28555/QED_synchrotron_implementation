import time as timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import kv
from scipy import constants as const

import cv2
import torch
torch.cuda.empty_cache()
from torch_interpolations import RegularGridInterpolator as interpolate

# precomputation of F1 and F2
def F1(z_q):
    if z_q > 2.9e-6:
        integral = quad(lambda x: kv(5/3, x), z_q, np.inf)[0]
        return z_q*integral
    else:
        return 2.15 * z_q**(1/3)
def F2(z_q):
    return  z_q*kv(2/3, z_q)

def precompute_F1_F2():
    print("Creating lookup tables for F1 and F2...")
    # cashed F2
    x = np.logspace(-20, 2, 512) 
    valuesF2 = [F2(z) for z in x]
    valuesF2 = torch.tensor(valuesF2, device='cuda', dtype=torch.float64)
    x = [torch.tensor(x, device='cuda', dtype=torch.float64)]
    cashedF2 = interpolate(x, valuesF2)
    
    # cashed F1
    x = np.logspace(-20, 2, 512) # valid to delta = 1e-7
    valuesF1 = [F1(z) for z in x]
    # failsafe for F1 if integration fails
    for i in range(len(valuesF1)-1,0,-1):
        if valuesF1[i] < 0:
            valuesF1[i] = 2.15* x[i]**(1/3) 
        
    valuesF1 = torch.tensor(valuesF1, device='cuda', dtype=torch.float64)
    x = [torch.tensor(x, device='cuda', dtype=torch.float64)]
    cashedF1 = interpolate(x, valuesF1)
    
    return cashedF1, cashedF2
cashedF1, cashedF2 = precompute_F1_F2()

# expect cuda tensors
def interpolateF_(x, cashedF):
    return cashedF([x])

# general formula for Heff -> v, B, E in SI are tensors of 3-tensors
def Heff_CUDA(v,vmag, B, E ):
    l1 = ((torch.cross(v,B)+E)**2).sum(-1)
    l2 = dotTensor(v/vmag,E)**2
    # if l1 < l2: -> very important
    mask = l1 > l2   
    return torch.sqrt( (l1 - l2)*mask )

# delta, gamma, Heff, are vectors
def P_CUDA(delta, gamma, Heff, dt):
    global cashedF1, cashedF2
    # Calculating chi and z_q. Es - in SI, Heff - in SI, gamma - unitless
    chi = gamma * Heff / Es
    z_q = 2 * delta / (3 * chi * (1 - delta)) 
    
    F1_result = interpolateF_(z_q, cashedF1)
    F2_result = interpolateF_(z_q, cashedF2)
    
    numericFactor = dt * (q**2 * ((m_e * c) /( hbar**2 * eps0 * 4 * np.pi))) # <- lot of numerical noise propably - went to float64

    requirement1 = numericFactor * 1.5*chi**(2/3) / gamma
    requirement2 = numericFactor * 0.5*chi**( 1 ) / gamma
        
    if requirement1.max() > 1e-1 or requirement2.max() > 1e-1:
        print("requirement1 = ", requirement1.max().item())
        print("requirement2 = ", requirement2.max().item())
    
    numericFactor *= np.sqrt(3)/(2 * np.pi)
    numerator1 = (1 - delta)*chi
    numerator2 = (F1_result + 3 * delta * z_q * chi / 2 * F2_result )
        
    denominator = gamma * delta
    propability = numericFactor * (numerator1*numerator2)/denominator
    return propability

# Heff, gamma are tensors
def Generate_photon_CUDA(Heff, gamma, dt):   
    r1 = torch.rand(gamma.shape, device='cuda', dtype=torch.float64)    
    P = 3*r1**2*P_CUDA(r1**3, gamma, Heff, dt)
    r2 = torch.rand(P.shape, device='cuda', dtype=torch.float64)
    mask = r2 <= P
    return (r1*mask)**3

# gamma and Heff are tensors
def calculate_dt(gamma, Heff):
    chi = gamma * Heff / Es    
    numericFactor = (q**2 * m_e * c /( hbar**2 * eps0 * 4 * np.pi))
    requirement1 = numericFactor * 1.5*chi**(2/3) / gamma
    requirement2 = numericFactor * 0.5*chi**( 1 ) / gamma
    # dt < 0.1/requirement(1/2)
    return torch.min(0.1/requirement1.max(), 0.1/requirement2.max()).item()*0.99

# make x a tensor
def Tens(x):
    return torch.tensor(x, dtype=torch.float64, device='cuda')
# dot product of two tensors of 3-tensors. Torch dot product only works with 3-tensors
def dotTensor(x,y):
    return (x*y).sum(-1)

#Physical constants:
q = const.elementary_charge  # Elementary charge in Coulombs
m_e = const.electron_mass  # Electron mass in kg
qm = q / m_e  # Charge-to-mass ratio for electron in C/kg
c = const.c  # Speed of light in m/s
hbar = const.hbar  # Reduced Planck constant in J*s

miu0 = const.mu_0 # Vacuum permeability in H/m
eps0 = 1.0 / miu0 / c / c # Vacuum permittivity in F/m
Es = m_e**2*c**3/q/hbar # Shwinger limit - Electric field strength in V/m

                    #Laser parameters:
w = 2*np.pi*c/800e-9 # Angular frequency for 800nm laser
k = Tens([0,w/c,0]) # Wave vector for 800nm laser
a0 = 100        #Normalized vector potential of and electromagnetic wave
init_gamma = 20 #Initial gamma value for the particles
El = a0*m_e*w*c/q #Electric field strength in V/m

                    #Time and space:
Tperiod = 2*np.pi/w #Period of the laser
substeps = 11
dt = Tperiod/10 # 11*35 = 385 steps per period
dt /= substeps
time = 0

# for visualization -> full image is 10x10 laser periods
kappa = w/c/2/np.pi # wave number
scale = 1/kappa 
scaleX = scale*20 
scaleY = scale*20


                    #Laser envelope:
# parameters
sigX = scale #sigma = laser period
sigY = scale*3
def envelope(x):
    global time, sigX, sigY
    # sig = 1e-6
    x0 = (time)*c-scaleY/2
    y0 = scaleX/2
    return torch.exp(-(x[:,1]-x0)**2/sigX**2) * torch.exp(-(x[:,0]-y0)**2/sigY**2)

# define the laser field
# x is a tensor of 3-tensors
def getEandB(x):
    global time
    
    #laser field
    phase = w*time - dotTensor(x,k)
    magnitude = torch.cos(phase)*El
    magnitude = magnitude * envelope(x)
        # magnitude = El
    vectorE = torch.zeros_like(x)
    vectorE[:,0] = magnitude
    vectorB = torch.zeros_like(x)
    vectorB[:,2] = -magnitude/c
    
    return vectorE, vectorB

SynchrotronQ = True # Synchrotron radiation on/off
energyAll = [] # list of energies of emitted photons
def update_Boris(positions,velocities,dt):
    E,B = getEandB(positions)

    #relativistic correction
    vmag = torch.norm(velocities,dim=-1)
    gamma = 1/torch.sqrt(1-(vmag/c)**2)
    
    momentum = m_e*velocities*gamma.unsqueeze(-1)
    
    if SynchrotronQ:
        heff = Heff_CUDA(velocities,vmag.unsqueeze(-1), B, E ) 
        if heff.max() > 108635699/2:
            # substepping
            new_dt = calculate_dt(gamma, heff)
            substeps = int(dt/new_dt)
            substeps = substeps if substeps > 0 else 1
            for _ in range(substeps):
                # Generate photons
                deltas = Generate_photon_CUDA(heff, gamma, new_dt)
                # update momentum
                momentum = momentum - momentum/torch.norm(momentum,dim=1).unsqueeze(-1) * (deltas*(gamma*m_e*c)).unsqueeze(-1)
                # save energies to the list
                mask = deltas > 0
                # if mask.any():
                #     print(f"heff = {heff.max().item()}")
                deltas = deltas[mask]
                energy = deltas*gamma[mask]*(m_e*c**2*6.2415e+18) # in eV
                # energy = energy.cpu().numpy()
                energyAll.append(energy)
        
    
    momMinus = momentum + q*E*dt/2
    mommag = torch.norm(momMinus,dim=-1)
    gamma = 1/torch.sqrt(1+(mommag/(m_e*c))**2)
    gamma = gamma.unsqueeze(-1)
    t = qm*B*(dt/2)*gamma
    momPrime = momMinus + torch.cross(momMinus,t)
    
    tmag2 = torch.norm(t,dim=-1)**2 
    tmag2 = tmag2.unsqueeze(-1)
    s = 2*t/(1+tmag2)
    
    momentum = momMinus + torch.cross(momPrime,s)
    
    momentum = momentum + q*E*dt/2
    gamma = torch.sqrt(1+(torch.norm(momentum,dim=-1)/(m_e*c))**2)
    gamma = gamma.unsqueeze(-1)

    velocities = momentum/(m_e*gamma)
    
    return positions + velocities*dt, velocities

# Initial conditions:
# Fast drawing of points on GPU
def draw_points(pos,img,color):
    # Apply mask
    mask_x = pos[:, 0].ge(0) & pos[:, 0].lt(SIZE)
    mask_y = pos[:, 1].ge(0) & pos[:, 1].lt(SIZE)
    pos = pos[mask_x & mask_y]
    # Prepare indices for advanced indexing
    rows = pos[:, 1]
    cols = pos[:, 0]
    # Advanced indexing for batched addition
    img[rows, cols] += color
    return img, pos, (mask_x & mask_y)

def plotEnergy(save=False):                    
    energy = torch.cat(energyAll, dim=0).cpu().numpy()
    if len(energyAll) == 0:
        energy = [0]
    # rescale energy to eV
    energy = energy
    
    # histogram of the generated photons
    minExp = np.floor(np.log10(np.min(energy)))
    maxExp = np.ceil(np.log10(np.max(energy)))
    bins = np.logspace(minExp, maxExp, 150)
    a,b = np.histogram(energy,bins=bins)
    # normalize on density
    a = a*(b[1:]+b[:-1])/(b[1:]-b[:-1])/2
    plt.bar(b[:-1],a,width=b[1:]-b[:-1])
    # title
    plt.title("Histogram of generated photons. Number of photons = " + str(len(energy)))
    plt.xlabel("energy [eV]")
    plt.ylabel("count/width")
    plt.xscale('log')
    plt.yscale('log')

    plt.show()
    plt.close()


# Create some balls
n_balls = 50_000
maxBalls = 50_000
displayBalls = n_balls if n_balls<maxBalls else maxBalls

def reset(gamma):
    positions = ((torch.rand(n_balls,3,dtype=torch.float64, device='cuda')/2)+Tens([0.25,0.3,0]))*Tens([scaleX,3*scaleY,0])
    
    velocities = torch.zeros_like(positions)
    init_vel = c * np.sqrt(1 - 1 / gamma**2)
    velocities[:,1] = -init_vel
    positions[:,1] += init_vel*dt*100

    return positions, velocities

positions, velocities = reset(init_gamma)

# Simulation loop
SIZE = 800 # size of the window

start = timeit.time()+1
iter = 0
fps = 1
plot_time = 1e-13
while True:
    # FPS counter
    if iter%10==0:
        cv2.setWindowTitle('image', "fps = {:.0f} \t\t".format(fps/10)+"mean time from last frame = {:.0f} ms".format(1/fps*10000)+"  simulation time = {:.3e} s".format(time))
        # print("fps = {:.0f} \t\t".format(fps/10)+"mean time from last frame = {:.0f} ms".format(1/fps*10000))
        fps = 0
    else:
        fps += 1/(timeit.time()-start)
    start = timeit.time()
    iter+=1
    
    # SIMULATION
    for _ in range(substeps):
        positions, velocities = update_Boris(positions,velocities,dt)
        time += dt
        if time > plot_time:
            plotEnergy(save=False)
            plot_time += 1e-14
    
    # VISUALIZATION
    img = torch.zeros((SIZE, SIZE, 3), dtype=torch.uint8, device='cuda')
    pos = (positions[:maxBalls, :2] * SIZE / torch.tensor([scaleX, scaleY], device='cuda')).type(torch.int32)
    col = torch.tensor([[0, 0, 200]], dtype=torch.uint8, device='cuda')
    img, pos, mask = draw_points(pos,img,col)
    
            # show Electric field
    E,B = getEandB(positions[:maxBalls][mask])
    E = (E/El).type(torch.float32)
    B = (B*c/El).type(torch.float32)
    EB = torch.cat((E,B),dim=0)
    
    #Draw E and B
    line_len = 10
    lines = torch.zeros((pos.shape[0]*2,line_len,2), dtype=torch.float32, device='cuda') # tensor of 2d points
    lines[:,0,:] = torch.cat((pos[:],pos[:]),dim=0)
    
    for i in range(1,line_len):
        lines[:,i,0] = lines[:,i-1,0] + EB[:,0]
        lines[:,i,1] = lines[:,i-1,1] + EB[:,2]
    lines = lines.type(torch.int32)
    # reshape lines to be a vector of 2d points
    lines = lines.reshape((-1, 2))
    # Draw E
    col = torch.tensor([[0, 200, 0]], dtype=torch.uint8, device='cuda')
    img, _, _ = draw_points(lines[:line_len*pos.shape[0]],img,col)
    # Draw B
    col = torch.tensor([[200, 0, 0]], dtype=torch.uint8, device='cuda')
    img, _, _ = draw_points(lines[line_len*pos.shape[0]:],img,col)
    
    img = img.cpu().numpy()
    
    # show image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    
    # if R key pressed - reset
    if cv2.waitKey(1) & 0xFF == ord('r'):
        time = 0
        positions, velocities = reset(init_gamma)





