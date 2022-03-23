#%%
import numpy as np
# from numba import njit
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#%%
############################################################
# Parameters from Wachter and Seo (2019)
gamma = 3.0   # risk aversion
psi = 1.0     # EIS (for completeness, not used anywhere)
beta = 0.012  # discount rate
muc = 0.0252   # drift of diffusion part of consumption
sigmac = 0.02 # vol of diffusion part of consumption
phi = 2.6 # leverage
lambda_bar = 0.0355 # unconditional average disaster probability
kappa = 0.080 # mean reversion of disaster probability
sigma_lambda = 0.067 # vol of disaster probability

# Parameters for lognormal jump size
muZ = -0.3
sigmaZ = 0.15

############################################################
# Derived constants

# Assume lognormal jump size
Ee1mgammaZ = np.exp((1-gamma)*muZ + 0.5*(1-gamma)**2 * sigmaZ**2)  # E[e^{(1-\gamma)Z_t}]
EephimgammaZ = np.exp((phi-gamma)*muZ + 0.5*(phi-gamma)**2 * sigmaZ**2)  # E[e^{(\phi-\gamma)Z_t}]

b = (kappa + beta)/sigma_lambda**2 - np.sqrt(((kappa + beta)/sigma_lambda**2)**2 - 2*(Ee1mgammaZ - 1)/sigma_lambda**2)
a = ((1-gamma)/beta) * (muc - 0.5*gamma*sigmac**2) + (1-gamma)*np.log(beta) + b*kappa*lambda_bar/beta

zeta_phi = np.sqrt((b*sigma_lambda**2 - kappa)**2 + 2*(Ee1mgammaZ - EephimgammaZ)*sigma_lambda**2)
eta = beta*(1-gamma)*np.log(beta) - beta*a - beta

############################################################
# Calculate linearization parameters
def aphi(tau):
    part1 = ((phi-1)*(muc+0.5*(phi-gamma)*sigmac**2) - beta - kappa*lambda_bar/sigma_lambda**2 * (zeta_phi + b*sigma_lambda**2 - kappa))*tau
    part2 = -2*kappa*lambda_bar/sigma_lambda**2 * np.log(((zeta_phi+b*sigma_lambda**2-kappa)*(np.exp(-zeta_phi*tau) - 1) + 2*zeta_phi)/(2*zeta_phi))
    return part1 + part2

def bphi(tau):
    num = 2*(Ee1mgammaZ - EephimgammaZ)*(1-np.exp(-zeta_phi*tau))
    den = (zeta_phi+b*sigma_lambda**2-kappa)*(1-np.exp(-zeta_phi*tau)) - 2*zeta_phi
    return num/den

############################################################
# Solve for Gstar
tau_grid = np.arange(0.0, 15000.0, 0.001)
dtau = tau_grid[1] - tau_grid[0]
Gstar = np.sum(np.exp(aphi(tau_grid) + bphi(tau_grid)*lambda_bar)*dtau)
print(f"P/D at average disaster probability: {Gstar:.2f}")

bphi_star = (1/Gstar)*np.sum(bphi(tau_grid)*np.exp(aphi(tau_grid) + bphi(tau_grid)*lambda_bar)*dtau)
print(f"Linearization parameter: {bphi_star:.2f}")

#%%
# ############################################################
# Function for solving ODE system given complex parameters

T = 365.0/365.0

# Solving the ODE from \tau=T to \tau=0
def calculatePSI(a_, b_, v):
    u1, u2 = (a_ + 1j * v * b_)[0], (a_ + 1j * v * b_)[1]
    beta1 = u1  # beta_1(t) = const = u_1
    xi = b*beta - (np.exp(beta1*muZ + 0.5*np.power(beta1, 2)*sigmaZ**2) - 1)

    dt = 0.0001
    tgrid = np.arange(0.0, T+dt, dt)
    beta2 = np.zeros(len(tgrid), dtype=complex)
    beta2[-1] = u2
    alpha = np.zeros(len(tgrid), dtype=complex)

    for i in range(len(tgrid) - 2, -1, -1):
        beta2[i] = beta2[i+1] - dt*(xi - kappa*beta2[i+1] - 0.5*sigma_lambda**2 * beta2[i+1]**2)
        alpha[i] = alpha[i+1] - dt*(-eta + (muc - 0.5*sigmac**2)*beta1 + kappa*lambda_bar*beta2[i+1] - 0.5*sigmac**2*beta1**2)

    return alpha[0], beta1, beta2[0]


def calculatePsiCDF(b_, v):
    u1, u2 = (1j * v * b_)[0], (1j * v * b_)[1]
    beta1 = u1  # beta_1(t) = const = u_1
    xi = - (np.exp(beta1*muZ + 0.5*np.power(beta1, 2)*sigmaZ**2) - 1)
    xi = 0.0

    dt = 0.0001
    tgrid = np.arange(0.0, T+dt, dt)
    beta2 = np.zeros(len(tgrid), dtype=complex)
    beta2[-1] = u2
    alpha = np.zeros(len(tgrid), dtype=complex)

    for i in range(len(tgrid) - 2, -1, -1):
        beta2[i] = beta2[i+1] - dt*(xi - kappa*beta2[i+1] - 0.5*sigma_lambda**2 * beta2[i+1]**2)
        alpha[i] = alpha[i+1] - dt*((muc - 0.5*sigmac**2)*beta1 + kappa*lambda_bar*beta2[i+1] - 0.5*sigmac**2*beta1**2)

    return alpha[0], beta1, beta2[0]

#%% Solving the ODEs to get the linear exponential coefficients for the transform

# Parameters for normalized puts
a_pK, b_pK = np.array([-gamma, b]), np.array([phi, bphi_star])
a_pS, b_pS = np.array([phi-gamma, b + bphi_star]), np.array([phi, bphi_star])

# Parameters for normalized calls
a_cK, b_cK = np.array([-gamma, b]), np.array([-phi, -bphi_star])
a_cS, b_cS = np.array([phi - gamma, b + bphi_star]), np.array([-phi, -bphi_star])

dv = 0.01
v_grid = np.arange(dv, 100.0 + dv, dv)

# Calculating \alpha(0), \beta(0) for all 4 objects:
ode_sol_pK = np.zeros((3, v_grid.shape[0]), dtype=complex)
ode_sol_pS = np.zeros((3, v_grid.shape[0]), dtype=complex)
ode_sol_cK = np.zeros((3, v_grid.shape[0]), dtype=complex)
ode_sol_cS = np.zeros((3, v_grid.shape[0]), dtype=complex)

for i in range(v_grid.shape[0]):
    if i % 1000 == 0:
        print(f"{i} out of {v_grid.shape[0]}")
    ode_sol_pK[:, i] = calculatePSI(a_pK, b_pK, v_grid[i])
    ode_sol_pS[:, i] = calculatePSI(a_pS, b_pS, v_grid[i])
    ode_sol_cK[:, i] = calculatePSI(a_cK, b_cK, v_grid[i])
    ode_sol_cS[:, i] = calculatePSI(a_cS, b_cS, v_grid[i])

ode_sol_pK0 = calculatePSI(a_pK, b_pK, 0)
ode_sol_pS0 = calculatePSI(a_pS, b_pS, 0)
ode_sol_cK0 = calculatePSI(a_cK, b_cK, 0)
ode_sol_cS0 = calculatePSI(a_cS, b_cS, 0)

#%% Inverting the transform with the initial conditions X_0 = (0, lambda_0)

lambda_0 = lambda_bar

psi_pK0 = np.exp(np.dot(ode_sol_pK0, [1, 0, lambda_0]))
psi_pS0 = np.exp(np.dot(ode_sol_pS0, [1, 0, lambda_0]))
psi_cK0 = np.exp(np.dot(ode_sol_cK0, [1, 0, lambda_0]))
psi_cS0 = np.exp(np.dot(ode_sol_cS0, [1, 0, lambda_0]))

psi_pK = np.exp(ode_sol_pK.T.dot(np.array([1, 0, lambda_0]).reshape((-1, 1)))).flatten()
psi_pS = np.exp(ode_sol_pS.T.dot(np.array([1, 0, lambda_0]).reshape((-1, 1)))).flatten()
psi_cK = np.exp(ode_sol_cK.T.dot(np.array([1, 0, lambda_0]).reshape((-1, 1)))).flatten()
psi_cS = np.exp(ode_sol_cS.T.dot(np.array([1, 0, lambda_0]).reshape((-1, 1)))).flatten()

# Specifying the strike for the options
Kn = 1.1
y_p = np.log(Kn) + bphi_star*lambda_0
y_c = -np.log(Kn) - bphi_star*lambda_0

G_pK = np.real(psi_pK0)/2 - 1/np.pi * np.sum(np.imag(psi_pK*np.exp(-1j*v_grid*y_p))/v_grid * dv)
G_pS = np.real(psi_pS0)/2 - 1/np.pi * np.sum(np.imag(psi_pS*np.exp(-1j*v_grid*y_p))/v_grid * dv)
G_cK = np.real(psi_cK0)/2 - 1/np.pi * np.sum(np.imag(psi_cK*np.exp(-1j*v_grid*y_c))/v_grid * dv)
G_cS = np.real(psi_cS0)/2 - 1/np.pi * np.sum(np.imag(psi_cS*np.exp(-1j*v_grid*y_c))/v_grid * dv)

P = Kn*np.exp(-b*lambda_0)*G_pK - np.exp(-(b + bphi_star)*lambda_0)*G_pS
C = np.exp(-(b + bphi_star)*lambda_0)*G_cS - Kn*np.exp(-b*lambda_0)*G_cK

print(f"Option prices:")
print(f"  P = {P:.5f}")
print(f"  C = {C:.5f}")

#%% Calculating the risk free rate

ode_sol_rf = calculatePSI(np.array([-gamma, b]), np.array([phi, bphi_star]), 0.0)
EpiT = np.real(np.exp(np.dot(ode_sol_rf, np.array([1, 0, lambda_0]))))*np.exp(-b*lambda_0)
rf = -np.log(EpiT)/T
rf_inst = beta + gamma*muc - gamma*sigmac**2 + lambda_0 *(Ee1mgammaZ - np.exp(-gamma*muZ + 0.5*gamma**2*sigmac**2))

print(f"Risk free rates")
print(f"  instantaneous   = {rf_inst:.5f}")
print(f"  1 year maturity = {rf:.5f}")


#%% Numerically inverting Black-Scholes to get implied volatilities

def d1d2(K, S, sigma, tau, r, q):
    F = np.exp((r - q)*tau)**S
    d1 = (np.log(F/K) + 0.5*sigma**2*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    return d1, d2, F

def callBS(K, S, sigma, tau, r, q):
    d1, d2, F = d1d2(K, S, sigma, tau, r, q)
    return np.exp(-r*tau)*(norm.cdf(d1)*F - norm.cdf(d2)*K)

def putBS(K, S, sigma, tau, r, q):
    d1, d2, F = d1d2(K, S, sigma, tau, r, q)
    return np.exp(-r*tau)*(norm.cdf(-d2)*K - norm.cdf(d1)*F)

S = 1.0
fsolve(lambda sigma: putBS(Kn*S, S, sigma, T, rf, 1/Gstar) - P, x0 = [0.5])



#%% Using approach in Wachter (2013) to solve for the risk free rate

tau = 1.0

EemgammaZ = np.exp(-gamma*muZ + 0.5*gamma**2 * sigmaZ**2) 
eta0 = np.sqrt(2*(EemgammaZ - Ee1mgammaZ)*sigma_lambda**2 - (b*sigma_lambda**2 - kappa)**2)

alpha0 = -muc - beta + gamma*sigmac**2 - (kappa*lambda_bar/sigma_lambda**2) *(b*sigma_lambda**2 - kappa)
alpha0 *= tau
alpha0 -= (2*kappa*lambda_bar/sigma_lambda**2) * np.log(np.cos(0.5*eta0*tau + np.arctan((b*sigma_lambda**2-kappa)/eta0))/np.cos(np.arctan((b*sigma_lambda**2-kappa)/eta0)))

beta0 = (1/sigma_lambda**2) * eta0 * np.tan(0.5*eta0*tau + np.arctan((b*sigma_lambda**2-kappa)/eta0)) - (b*sigma_lambda**2 - kappa)/sigma_lambda**2

rf_wachter = -(alpha0 + beta0*lambda_0)/tau
print("Closed form risk free rate from Wachter (2013):")
print(f"   rf = {rf_wachter:.8f}")

#%% Something else

dtau = 0.0001
tau_grid = np.arange(0.0, 1.0+dtau, dtau)
a0 = np.zeros(tau_grid.shape)
b0 = np.zeros(tau_grid.shape)
for i in range(0, tau_grid.shape[0]-1):
    b0[i+1] = b0[i] + dtau*(0.5*sigma_lambda**2 * b0[i]**2 + (b*sigma_lambda**2 - kappa)*b0[i] + (EemgammaZ - Ee1mgammaZ))
    a0[i+1] = a0[i] + dtau*(-muc - beta + gamma*sigmac**2 + kappa*lambda_bar*b0[i])

print(-np.log(np.exp(a0[-1] + b0[-1]*lambda_0)))

#%%
a_ = np.array([-gamma, b])
u1, u2 = a_[0], a_[1]
beta1 = u1  # beta_1(t) = const = u_1
xi = b*beta - (np.exp(beta1*muZ + 0.5*np.power(beta1, 2)*sigmaZ**2) - 1)
# xi = -xi

dt = 0.001
tgrid = np.arange(0.0, T+dt, dt)
beta2 = np.zeros(len(tgrid), dtype=complex)
beta2[-1] = u2
alpha = np.zeros(len(tgrid), dtype=complex)

for i in range(len(tgrid) - 2, -1, -1):
    beta2[i] = beta2[i+1] - dt*(xi - kappa*beta2[i+1] - 0.5*sigma_lambda**2 * beta2[i+1]**2)
    alpha[i] = alpha[i+1] - dt*(-eta + (muc - 0.5*sigmac**2)*beta1 + kappa*lambda_bar*beta2[i+1] - 0.5*sigmac**2*beta1**2)

-np.real(alpha[0] + (beta2[0]-b)*lambda_0)

#%% Using a closed form solution for beta(t)

sqrt_term = np.lib.scimath.sqrt(4*(b*beta - (np.exp(-gamma*muZ + 0.5*gamma**2 * sigmaZ**2) - 1))*(-0.5*sigma_lambda**2) - kappa**2)
k1 = np.arctan((b*2*(-0.5*sigma_lambda**2) - kappa)/sqrt_term)*2/sqrt_term - T
beta2_closed = (sqrt_term*np.tan(0.5*k1*sqrt_term) + kappa)/(2*(-0.5*sigma_lambda**2))

np.real((beta2_closed - b)*lambda_0)



#####################################################################
#####################################################################
#####################################################################
#%% Calculating CDF of S_T/S_0

bCDF = np.array([phi, bphi_star])
dv = 0.01
v_grid = np.arange(dv, 100.0 + dv, dv)
ode_sol_CDF = np.zeros((3, v_grid.shape[0]), dtype=complex)

for i in range(v_grid.shape[0]):
    if i % 1000 == 0:
        print(f"{i} out of {v_grid.shape[0]}")
    ode_sol_CDF[:, i] = calculatePsiCDF(bCDF, v_grid[i])

ode_sol_CDF0 = calculatePsiCDF(bCDF, 0.0)
psi_CDF0 = np.exp(np.dot(ode_sol_CDF0, [1, 0, lambda_0]))
psi_CDF = np.exp(ode_sol_CDF.T.dot(np.array([1, 0, lambda_0]).reshape((-1, 1)))).flatten()

#%% ... continuing
y = 1
ygrid = np.arange(0.01, 1.25, 0.01)
CDF_grid = np.zeros(ygrid.shape)*np.nan
for i in range(ygrid.shape[0]):
    yCDF = np.log(ygrid[i]) + bphi_star*lambda_0
    G_CDF = np.real(psi_CDF0)/2 - 1/np.pi * np.sum(np.imag(psi_CDF*np.exp(-1j*v_grid*yCDF))/v_grid * dv)
    CDF_grid[i] = G_CDF

PDF_grid = (CDF_grid[2:] - CDF_grid[:-2])/(2*(ygrid[1] - ygrid[0]))

fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.axhline(0.0)
ax.axhline(1.0)
ax.plot(ygrid, CDF_grid)

#%% ... calculating pdf

fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.axhline(0.0)
ax.axhline(1.0)
ax.plot(ygrid[1:-1], PDF_grid)


#%%

yCDF = np.log(0.01) + bphi_star*lambda_0
to_plot = np.real(psi_CDF0)/2 - 1/np.pi * np.cumsum(np.imag(psi_CDF*np.exp(-1j*v_grid*yCDF))/v_grid * dv)
plt.plot(to_plot)

#%% Testing normal distribution

def calculatePsiNorm(v):
    mu, sigma = 0.06, 0.18
    u = 1j * v
    beta = u

    dt = 0.0001
    tgrid = np.arange(0.0, T+dt, dt)
    alpha = np.zeros(len(tgrid), dtype=complex)

    for i in range(len(tgrid) - 2, -1, -1):
        alpha[i] = alpha[i+1] - dt*(-mu*beta - 0.5*beta**2 * sigma**2)

    return alpha[0], beta

dv = 0.01
vgrid = np.arange(dv, 100.0+dv, dv)
ode_sol_NormCDF = np.zeros((2, vgrid.shape[0]), dtype=complex)

for i in range(vgrid.shape[0]):
    if i % 1000 == 0:
        print(f"{i} out of {vgrid.shape[0]}")
    ode_sol_NormCDF[:, i] = calculatePsiNorm(vgrid[i])

ode_sol_NormCDF0 = calculatePsiNorm(0.0)
psi_NormCDF0 = np.exp(np.dot(ode_sol_NormCDF0, [1, 0]))
psi_NormCDF = np.exp(ode_sol_NormCDF.T.dot(np.array([1, 0]).reshape((-1, 1)))).flatten()

#%%

ygrid = np.arange(-0.5, 0.5, 0.0001)
CDF_grid = np.zeros(ygrid.shape)*np.nan
for i in range(ygrid.shape[0]):
    yCDF = ygrid[i]
    G_CDF = np.real(psi_NormCDF0)/2 - 1/np.pi * np.sum(np.imag(psi_NormCDF*np.exp(-1j*vgrid*yCDF))/vgrid * dv)
    CDF_grid[i] = G_CDF

#%% Comparing graphically
from scipy.stats import norm

fig, ax = plt.subplots(1, 1, figsize=(5,4))
ax.plot(ygrid, CDF_grid, label="From fourier")
ax.plot(ygrid, norm.cdf(ygrid, loc=0.06*T, scale=0.18*np.sqrt(T)), label="Theoretical")
ax.legend()


#%% Testing pure jump model

def calculatePsiPureJump(v):
    lambda_, muZ = 1.0, -0.1
    T = 1.0
    u = 1j * v
    beta = u

    dt = 0.0001
    tgrid = np.arange(0.0, T+dt, dt)
    alpha = np.zeros(len(tgrid), dtype=complex)

    for i in range(len(tgrid) - 2, -1, -1):
        alpha[i] = alpha[i+1] - dt*(-lambda_*(np.exp(muZ*beta) - 1))

    return alpha[0], beta

dv = 0.01
vgrid = np.arange(dv, 200.0+dv, dv)
ode_sol_PureJumpCDF = np.zeros((2, vgrid.shape[0]), dtype=complex)

for i in range(vgrid.shape[0]):
    if i % 1000 == 0:
        print(f"{i} out of {vgrid.shape[0]}")
    ode_sol_PureJumpCDF[:, i] = calculatePsiPureJump(vgrid[i])

ode_sol_PureJumpCDF0 = calculatePsiPureJump(0.0)
psi_PureJumpCDF0 = np.exp(np.dot(ode_sol_PureJumpCDF0, [1, 0]))
psi_PureJumpCDF = np.exp(ode_sol_PureJumpCDF.T.dot(np.array([1, 0]).reshape((-1, 1)))).flatten()

# %%

ygrid = np.arange(-0.5, 0.5, 0.0001)
CDF_grid = np.zeros(ygrid.shape)*np.nan
for i in range(ygrid.shape[0]):
    yCDF = ygrid[i]
    G_CDF = np.real(psi_PureJumpCDF0)/2 - 1/np.pi * np.sum(np.imag(psi_PureJumpCDF*np.exp(-1j*vgrid*yCDF))/vgrid * dv)
    CDF_grid[i] = G_CDF

def closedForm(x):
    lambda_, muZ, T = 1.0, -0.1, 1.0
    n_min = int(np.max([0.0, np.ceil(x/muZ)]))
    return np.sum([(lambda_*T)**n/np.math.factorial(n) * np.exp(-lambda_*T) for n in range(n_min, 50)])

fig, ax = plt.subplots(1, 1, figsize=(5,4))
ax.plot(ygrid, CDF_grid, label="From fourier")
ax.plot(ygrid, [closedForm(x) for x in ygrid], label="Closed form")
ax.legend()

#%%

u1, u2 = -gamma, b
v1, v2 = 0.0, 1.0

beta1 = u1
B1 = v1

dt = 0.001
tgrid = np.arange(0.0, T+dt, dt)
beta2 = np.zeros(tgrid.shape)*np.nan
alpha = np.zeros(tgrid.shape)*np.nan
B2 = np.zeros(tgrid.shape)*np.nan
A = np.zeros(tgrid.shape)*np.nan

beta2[-1] = u2
alpha[-1] = 0.0
B2[-1] = v2
A[-1] = 0

rho0 = -(muc - gamma*sigmac**2 + beta*(1-gamma)*np.log(beta) - beta*a)

for i in range(len(tgrid) - 2, -1, -1):
    beta2[i] = beta2[i+1] - dt*(beta*b - (Ee1mgammaZ - 1) + kappa*beta2[i+1] - 0.5*sigma_lambda**2*beta2[i+1]**2)
    alpha[i] = alpha[i+1] - dt*(rho0 - (muc - sigmac**2/2)*beta1 - kappa*lambda_bar*beta2[i+1] - 0.5*sigmac**2*beta1**2)
    B2[i] = B2[i+1] - dt*(kappa*B2[i+1] - sigma_lambda**2 * beta2[i+1]*B2[i+1])
    A[i] = A[i+1] - dt*(-kappa*lambda_bar*B2[i+1])

EQlambda = np.exp(alpha[0] + beta2[0]*lambda_0)*(A[0] + B2[0]*lambda_0)*np.exp(-b*lambda_0)
EQlambda







# %%

# def calculatePsiCDF(b_, v):
#     u1, u2 = (1j * v * b_)[0], (1j * v * b_)[1]
#     beta1 = u1  # beta_1(t) = const = u_1
#     xi = - (np.exp(beta1*muZ + 0.5*np.power(beta1, 2)*sigmaZ**2) - 1)
#     xi = 0.0

#     dt = 0.0001
#     tgrid = np.arange(0.0, T+dt, dt)
#     beta2 = np.zeros(len(tgrid), dtype=complex)
#     beta2[-1] = u2
#     alpha = np.zeros(len(tgrid), dtype=complex)

#     for i in range(len(tgrid) - 2, -1, -1):
#         beta2[i] = beta2[i+1] - dt*(xi - kappa*beta2[i+1] - 0.5*sigma_lambda**2 * beta2[i+1]**2)
#         alpha[i] = alpha[i+1] - dt*((muc - 0.5*sigmac**2)*beta1 + kappa*lambda_bar*beta2[i+1] - 0.5*sigmac**2*beta1**2)

#     return alpha[0], beta1, beta2[0]

# dv = 0.01
# vgrid = np.arange(dv, 100.0+dv, dv)
# ode_sol_lambdaCDF = np.zeros((1, vgrid.shape[0]), dtype=complex)

# for i in range(vgrid.shape[0]):
#     if i % 1000 == 0:
#         print(f"{i} out of {vgrid.shape[0]}")
#     ode_sol_lambdaCDF[:, i] = calculatePsiCDF(vgrid[i])

# ode_sol_NormCDF0 = calculatePsiCDF(0.0)
# psi_NormCDF0 = np.exp(np.dot(ode_sol_NormCDF0, [1, 0]))
# psi_NormCDF = np.exp(ode_sol_NormCDF.T.dot(np.array([1, 0]).reshape((-1, 1)))).flatten()



#%%