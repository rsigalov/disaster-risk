#%%
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.stats import gamma as gamma_dist
import seaborn as sns
from functools import reduce

#%%

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
Ee1mgammaZ = np.exp((1-gamma)*muZ + 0.5*(1-gamma)**2 * sigmaZ**2)  # E[e^{(1-\gamma)Z_t}]
EephimgammaZ = np.exp((phi-gamma)*muZ + 0.5*(phi-gamma)**2 * sigmaZ**2)  # E[e^{(\phi-\gamma)Z_t}]

b = (kappa + beta)/sigma_lambda**2 - np.sqrt(((kappa + beta)/sigma_lambda**2)**2 - 2*(Ee1mgammaZ - 1)/sigma_lambda**2)
a = ((1-gamma)/beta) * (muc - 0.5*gamma*sigmac**2) + (1-gamma)*np.log(beta) + b*kappa*lambda_bar/beta

zeta_phi = np.sqrt((b*sigma_lambda**2 - kappa)**2 + 2*(Ee1mgammaZ - EephimgammaZ)*sigma_lambda**2)
eta = beta*(1-gamma)*np.log(beta) - beta*a - beta
############################################################
# JTIX coefficients
muXQ = phi*muZ - phi*gamma*sigmaZ**2
sigmaXQ = phi*sigmaZ
Psi = np.exp(-0.5*(2*gamma*muZ - gamma**2 * sigmaZ**2))
Psi *= (1 + muXQ + 0.5*(muXQ**2 + sigmaXQ**2) - np.exp(muXQ + 0.5*sigmaXQ**2))
kappaQ = kappa - b*sigma_lambda
tau = 180.0/365.0

a1 = Psi/kappaQ * (1 - np.exp(-kappaQ*tau))
a2 = Psi * (lambda_bar*kappa/kappaQ * tau - lambda_bar * kappa/kappaQ**2 * (1-np.exp(-kappaQ*tau)))


#%%

# Wachter (2013) model implies that JTIX_t = a_1 \lambda_t + a_2 (for a particular maturity \tau)
# i.e. the coefficients a_1 and a_2 are functions of \tau. Denote X_t \equiv JTIX_t. Then the
# law of motion can be calculated as 
#
#       dX_t = a_1 \kappa (\bar{\lambda} - \lambda_t) + a_1\sigma_\lambda\sqrt{\lambda_t}dB_{\lambda, t}
#
# To make this a process of X_t only substitute \lambda_t = (X_t - a_2)/a_1
#
#       dX_t = a_1 \kappa (\bar{\lambda} - (X_t - a_2)/a_1) + \sigma_\lambda a_1\sqrt{(X_t - a_2)/a_1}dB_{\lambda, t}
# 
# Rearrange to get
#
#       dX_t = \kappa ((a_1 \bar{\lambda} + a_2) - X_t) + \sigma_\lambda a_1\sqrt{(X_t - a_2)/a_1}dB_{\lambda, t}


# def xvol(x):
#     return a1*np.sqrt((x-a2)/a1)

# def dxvol(x):
#     return 0.5*np.power((x-a2)/a1, -0.5)

# def ddxvol(x):
#     return 0.5*(-0.5)*(1/a1)*np.power((x-a2)/a1, -1.5)

# # Raw parameters of the disaster intensity process
# kappa = 0.08
# sigma_lambda = 0.067
# lambda_bar = 0.0355

# # Paramaters of JTIX process
# # a1 = 2.0
# # a2 = 0.1
# kappax = a1*kappa
# xbar = a1*lambda_bar + a2
# sigmax = sigma_lambda

# xmin = a2
# xmax = 0.15

# Nx = 10000 # for some reason a finer grid leads to instability
# dx = (xmax - xmin)/(Nx - 1)
# xgrid = np.linspace(xmin, xmax, Nx, endpoint=True)

# # Diagonal of discretization matrix: multiplying p_{i}^{(n+1)}
# diag_arr = kappax + sigmax**2*np.power(dxvol(xgrid), 2) + sigmax**2*xvol(xgrid)*ddxvol(xgrid)
# diag_arr += 1/dx * (kappax*(xbar-xgrid) - 2*xvol(xgrid)*sigmax**2*dxvol(xgrid))*np.where(xgrid < xbar, 1, 0)
# diag_arr += 1/dx * (-kappax*(xbar-xgrid) + 2*xvol(xgrid)*sigmax**2*dxvol(xgrid))*np.where(xgrid >= xbar, 1, 0)
# diag_arr += -np.power(xvol(xgrid), 2)*sigmax**2/dx**2

# # Upper subdiagonal of discretization matrix: multiplying p_{i+1}^{(n+1)}
# diagp1_arr = 1/dx * (-kappax*(xbar-xgrid) + 2*xvol(xgrid)*sigmax**2*dxvol(xgrid))*np.where(xgrid < xbar,1,0)
# diagp1_arr += 0.5*np.power(xvol(xgrid),2)*sigmax**2/dx**2

# # Lower subdiagonal of discretization matrix: multiplying p_{i-1}^{(n+1)}
# diagm1_arr = 1/dx * (kappax*(xbar-xgrid) - 2*xvol(xgrid)*sigmax**2*dxvol(xgrid))*np.where(xgrid >= xbar,1,0)
# diagm1_arr += 0.5*np.power(xvol(xgrid),2)*sigmax**2/dx**2

# # Forming matrices
# A = diags(
#     [diagm1_arr[1:], diag_arr, diagp1_arr[:-1]],
#     offsets=[-1, 0, 1]
# )
# I = eye(np.shape(xgrid)[0])
# dt = 0.1 # time step
# N_time_steps = 500

# # Initial condition (dirac mass)
# init_state = xbar
# init_state_ind = np.argmin(np.abs(init_state - xgrid))
# p = np.zeros((xgrid.shape[0], N_time_steps))
# p[:, 0] = np.zeros(xgrid.shape)
# p[init_state_ind, 0] = 0.1/dx

# # Adjusting for boundary condition p(x_min) = 0
# mult_matrix = 1/dt * I - A
# mult_matrix[0, 0] = 1.0
# mult_matrix[0, 1] = 0.0

# for t in range(N_time_steps-1):
#     p[:, t+1] = spsolve(mult_matrix, 1/dt * p[:, t])

# # Renormalizing at each step
# p = p/np.tile(np.sum(p, axis=0).reshape((1,-1)), (xgrid.shape[0], 1))
# p = p/dx

# # Plotting transition and stationary distribution
# cmap = get_cmap('viridis_r')
# t_list = [1, 2, 3, 4, 10, 25, 50, -1]
# color_list = [cmap(0.2 + x/len(t_list)/1.2) for x in range(len(t_list))]
# fig, axs = plt.subplots(1, 2, figsize=(10,4))
# dashed_line_style = {'linestyle':'--','linewidth':1.5,'alpha':0.5,'color':'black'}
# solid_line_style = {'linestyle':'-','linewidth':1.5,'alpha':0.5,'color':'black'}

# axs[0].plot(xgrid, p[:,t_list[-1]],
#         label=f'Stationary (t={p.shape[1]})', color=color_list[-1], linewidth=3.0)
# axs[0].axvline(xmax, label='State boundary', **solid_line_style)
# axs[0].axvline(xmin, **solid_line_style)
# axs[0].axhline(0.0, **solid_line_style)
# axs[0].set_title("Stationary distribution")

# axs[1].axvline(xgrid[init_state_ind], label='Initial state', **dashed_line_style)
# axs[1].axvline(xmax, label='State boundary', **solid_line_style)
# axs[1].axvline(xmin, **solid_line_style)
# axs[1].axhline(0.0, **solid_line_style)
# for t, color in zip(t_list[:-1], color_list[:-1]):
#     axs[1].plot(xgrid, p[:,t], label=f't={round(t*dt,1)}', color=color)
# axs[1].plot(xgrid, p[:,t_list[-1]],
#         label=f'Stationary (t={p.shape[1]})', color=color_list[-1], linewidth=3.0)
# axs[1].legend()
# axs[1].set_title("Evolution")
# plt.show()



#%%
# fig, ax = plt.subplots(1, 1, figsize=(5,4))
# ax.plot(xgrid, p[:,t_list[-1]]*6,
#         label=f'Stationary (t={p.shape[1]})', color=color_list[-1], linewidth=3.0)
# ax.hist(SPX[SPX["days"] == 30]["value"], bins=100)
# ax.set_xlim(-0.001, 0.02)



#%%
# fig, ax = plt.subplots(1, 1, figsize=(5,4))
# ax.plot(xgrid, p[:,t_list[-1]],
#         label=f'Stationary (t={p.shape[1]})', color=color_list[-1], linewidth=3.0)
# sns.kdeplot(SPX[SPX["days"] == 30]["value"], ax=ax, gridsize=1000, cut=0)
# ax.set_xlim(-0.001, 0.02)

#%% Comparing in logs
# fig, ax = plt.subplots(1, 1, figsize=(5,4))
# ax.plot(np.log(xgrid), p[:,t_list[-1]]/1000,
#         label=f'Stationary (t={p.shape[1]})', color=color_list[-1], linewidth=3.0)
# sns.kdeplot(np.log(SPX[SPX["days"] == 30]["value"]), ax=ax)





#%% Simulating a path for lambda and JTIX
# np.random.seed(19960228)
# dt = 1/(252*1000)
# dB = np.random.normal(0.0, 1, 252*1000*100)
# lambda_sim = np.zeros(dB.shape)*np.nan
# lambda_sim[0] = lambda_bar
# lambda_sim_sample = np.zeros(100*252)*np.nan

# for t in range(dB.shape[0]-1):
#     lambda_sim[t+1] = lambda_sim[t] + kappa*(lambda_bar - lambda_sim[t])*dt + sigma_lambda*np.sqrt(lambda_sim[t])*dB[t+1]*np.sqrt(dt)
#     # Sample at the end of the day
#     if (t+1) % 1000 == 0:
#         lambda_sim_sample[t//1000] = lambda_sim[t+1]

# # Calculating JTIX
# JTIX_sim = a1*lambda_sim_sample + a2

# plt.hist(np.log(SPX[SPX["days"] == 30]["value"]), label="Empirical")
# plt.hist(np.log(JTIX_sim), label="simulation")
# plt.legend()

#%% Comparing histograms:

# fig,axs = plt.subplots(2,1,figsize=(5, 7), sharex=True)
# axs[0].hist(np.log(SPX[SPX["days"] == 30]["value"]), bins=20, label="Empirical")
# axs[0].legend()
# axs[1].hist(np.log(JTIX_sim), bins=20, label="Simulation")
# axs[1].legend()



#%%

#%%





# %% Stationary distribution for the log
# Denote Y_t = log(X_t)

# Defining function to simplify notation
# def muY(y):
#     return a1*np.exp(-y)*kappa*(lambda_bar - (np.exp(y) - a2)/a1) - 0.5*a1**2*np.exp(-2*y)*((np.exp(y) - a2)/a1)*sigma_lambda**2

# def dmuY(y):
#     return 0.5*np.exp(-2*y)*(a1*np.exp(y)*(sigma_lambda**2 - 2*kappa*lambda_bar) - 2*a2*(np.exp(y)*kappa + a1*sigma_lambda**2))

# def sigmaY(y):
#     return a1 * np.exp(-y)*np.sqrt((np.exp(y) - a2)/a1)*sigma_lambda

# def dsigmaY(y):
#     return -np.exp(-y)*(np.exp(y) - 2*a2)*sigma_lambda/(2*np.sqrt((np.exp(y) - a2)/a1))

# def ddsigmaY(y):
#     return np.exp(-y)*(4*a2**2 - 6*a2*np.exp(y) + np.exp(2*y))*sigma_lambda/(4*a1*np.power((np.exp(y) - a2)/a1, 3/2))

# # Writing diagonals


# ymin = np.log(a2)
# ymax = np.log(0.20)

# Ny = 50 # for some reason a finer grid leads to instability
# dy = (ymax - ymin)/(Ny - 1)
# ygrid = np.linspace(ymin, ymax, Ny, endpoint=True)

# # Diagonal of discretization matrix: multiplying p_{i}^{(n+1)}
# diag_arr = -dmuY(ygrid) + np.power(dsigmaY(ygrid), 2) + sigmaY(ygrid)*ddsigmaY(ygrid)
# diag_arr = 1/dy * np.where(muY(ygrid) > 0, 1, 0) * (muY(ygrid) - 2*sigmaY(ygrid)*dsigmaY(ygrid))
# diag_arr = 1/dy * np.where(muY(ygrid) <= 0, 1, 0) * (-muY(ygrid) + 2*sigmaY(ygrid)*dsigmaY(ygrid))
# diag_arr = -np.power(sigmaY(ygrid), 2)/dy**2

# # Upper subdiagonal of discretization matrix: multiplying p_{i+1}^{(n+1)}
# diagp1_arr = 1/dy * np.where(muY(ygrid) > 0, 1, 0)*(-muY(ygrid) + 2*sigmaY(ygrid)*dsigmaY(ygrid))
# diagp1_arr += 0.5*np.power(sigmaY(ygrid), 2)/dy**2

# # Lower subdiagonal of discretization matrix: multiplying p_{i-1}^{(n+1)}
# diagm1_arr = 1/dy * np.where(muY(ygrid) <= 0, 1, 0)*(muY(ygrid) - 2*sigmaY(ygrid)*dsigmaY(ygrid))
# diagm1_arr += 0.5*np.power(sigmaY(ygrid), 2)/dy**2

# # Forming matrices
# A = diags(
#     [diagm1_arr[1:], diag_arr, diagp1_arr[:-1]],
#     offsets=[-1, 0, 1]
# )
# I = eye(np.shape(ygrid)[0])
# dt = 0.01 # time step
# N_time_steps = 10000

# # Initial condition (dirac mass)
# init_state = ygrid[np.argmin(np.abs(muY(ygrid)))]
# init_state_ind = np.argmin(np.abs(init_state - ygrid))
# p = np.zeros((ygrid.shape[0], N_time_steps))
# p[:, 0] = np.zeros(ygrid.shape)
# p[init_state_ind, 0] = 1.0/dy

# # Adjusting for boundary condition p(x_min) = 0
# mult_matrix = 1/dt * I - A
# mult_matrix[0, 0] = 1.0
# mult_matrix[0, 1] = 0.0
# for t in range(N_time_steps-1):
#     p[:, t+1] = spsolve(mult_matrix, 1/dt * p[:, t])

# # diag_arr[0] = 0.0
# # diagp1_arr[0] = 0.0
# # diag_arr[-1] += diagp1_arr[-1]
# # A = diags(
# #     [diagm1_arr[1:], diag_arr, diagp1_arr[:-1]],
# #     offsets=[-1, 0, 1]
# # )

# # # A[0, 0] = 0
# # # A[0, 1] = 0
# # # A[-1, -1] = A[-1, -1] + diagp1_arr[-1]

# # for t in range(N_time_steps-1):
# #     # p[:, t+1] = spsolve(mult_matrix, 1/dt * p[:, t])
# #     p[:, t+1] = p[:,t] + dt*A.dot(p[:, t])

# # Renormalizing at each step
# p = p/np.tile(np.sum(p, axis=0).reshape((1,-1)), (ygrid.shape[0], 1))
# p = p/dy

# fig, axs = plt.subplots(1, 2, figsize=(10,4))
# dashed_line_style = {'linestyle':'--','linewidth':1.5,'alpha':0.5,'color':'black'}
# solid_line_style = {'linestyle':'-','linewidth':1.5,'alpha':0.5,'color':'black'}

# axs[0].plot(ygrid, p[:,t_list[-1]],
#         label=f'Stationary (t={p.shape[1]})', color=color_list[-1], linewidth=3.0)
# axs[0].axvline(ymax, label='State boundary', **solid_line_style)
# axs[0].axvline(ymin, **solid_line_style)
# axs[0].axhline(0.0, **solid_line_style)
# axs[0].set_title("Stationary distribution")

# axs[1].axvline(ygrid[init_state_ind], label='Initial state', **dashed_line_style)
# axs[1].axvline(ymax, label='State boundary', **solid_line_style)
# axs[1].axvline(ymin, **solid_line_style)
# axs[1].axhline(0.0, **solid_line_style)
# for t, color in zip(t_list[:-1], color_list[:-1]):
#     axs[1].plot(ygrid, p[:,t], label=f't={round(t*dt,1)}', color=color)
# axs[1].plot(ygrid, p[:,t_list[-1]],
#         label=f'Stationary (t={p.shape[1]})', color=color_list[-1], linewidth=3.0)
# axs[1].legend()
# axs[1].set_title("Evolution")
# plt.show()

# %%
# fig, ax = plt.subplots(1,1)
# ax.plot(ygrid, p[:,-1])
# ax.set_xlim(-9, -5)

#%% Overlaying this with the empirical measure.Using semi-closed form distribution
df_list = []
for days in [30, 60, 90, 120 ,150, 180]:
    df_list.append(pd.read_csv(f"data/interpolated_D/interpolated_disaster_spx_{days}.csv"))
SPX = pd.concat(df_list, ignore_index=True)
SPX = SPX[SPX["variable"] == "D_clamp"]

muXQ = phi*muZ - phi*gamma*sigmaZ**2
sigmaXQ = phi*sigmaZ
Psi = np.exp(-0.5*(2*gamma*muZ - gamma**2 * sigmaZ**2))
Psi *= (1 + muXQ + 0.5*(muXQ**2 + sigmaXQ**2) - np.exp(muXQ + 0.5*sigmaXQ**2))
Psi *= 2
kappaQ = kappa - b*sigma_lambda

ndraws = SPX[SPX["days"] == 180]["value"].shape[0]
rv_list = []
day_list = [30, 90, 180]
for days in [30, 90, 180]:
    tau = days/365.0
    a1 = Psi/kappaQ * (1 - np.exp(-kappaQ*tau))
    a2 = Psi * (lambda_bar*kappa/kappaQ * tau - lambda_bar * kappa/kappaQ**2 * (1-np.exp(-kappaQ*tau)))
    alpha_gamma = 2*kappa/sigma_lambda**2 * lambda_bar
    beta_gamma = 2*kappa/sigma_lambda**2
    rv = gamma_dist(alpha_gamma, loc = 0., scale = 1/beta_gamma)
    rv_list.append(np.log(a1*rv.rvs(ndraws) + a2))

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for ax, rvs, days in zip(axs[0, :], rv_list, day_list):
    ax.hist(np.exp(rvs), bins = 20, alpha=0.5, label="Wachter (2013)")
    ax.hist(SPX[SPX["days"] == days]["value"], bins=20, alpha=0.5, label="Empirical")
    ax.set_title(f"Maturity: {days} days")
    ax.set_xlabel("JTIX")

for ax, rvs, days in zip(axs[1, :], rv_list, day_list):
    ax.hist(rvs, bins = 20, alpha=0.5, label="Wachter (2013)")
    ax.hist(np.log(SPX[SPX["days"] == days]["value"]), bins=20, alpha=0.5, label="Empirical")
    ax.set_title(f"Maturity: {days} days")
    ax.set_xlabel("log(JTIX)")

axs[0, 0].legend()
plt.tight_layout()
fig.savefig("SS_figures/JTIX_wachter2013.pdf")



# %% Comparing disaster probability from P/D ratio and from 

# 1. Get data on price earnings ratios from Robert Shiller's website
# http://www.econ.yale.edu/~shiller/data.htm
cape = pd.read_excel("data/cape_shiller.xls", sheet_name="Data")
cape.columns = list(cape.iloc[6, :].astype(str))
cape = cape.iloc[7:, :]
cape = cape.rename(columns={"Date": "date"})
cape["date"] = cape["date"].apply(lambda date: "-".join(str(date).split(".")) + "-01")
cape["date"] = pd.to_datetime(cape["date"], errors="coerce")
cape["date"] = cape["date"] + pd.offsets.MonthEnd(0)
cape = cape[cape["date"].notnull() & cape["CAPE"].notnull()]
cape = cape[["date", "CAPE"]]
cape["CAPE"] = cape["CAPE"].astype(float)

# 2. Following Wachter (2013) taking log of CAPE and subtracting 
#    average log(CAPE) over the whole sample.
cape["LogCAPE"] = np.log(cape["CAPE"])
cape["LogCAPE_demean"] = cape["LogCAPE"] - cape["LogCAPE"].mean()

# 3. Adding back the average price-dividend ratio from the model
#    (simulate lambda from Gamma distribution and apply the
#     sutable transformation)

def aphi(tau):
    part1 = ((phi-1)*(muc+0.5*(phi-gamma)*sigmac**2) - beta - kappa*lambda_bar/sigma_lambda**2 * (zeta_phi + b*sigma_lambda**2 - kappa))*tau
    part2 = -2*kappa*lambda_bar/sigma_lambda**2 * np.log(((zeta_phi+b*sigma_lambda**2-kappa)*(np.exp(-zeta_phi*tau) - 1) + 2*zeta_phi)/(2*zeta_phi))
    return part1 + part2

def bphi(tau):
    num = 2*(Ee1mgammaZ - EephimgammaZ)*(1-np.exp(-zeta_phi*tau))
    den = (zeta_phi+b*sigma_lambda**2-kappa)*(1-np.exp(-zeta_phi*tau)) - 2*zeta_phi
    return num/den

tau_grid = np.arange(0.0, 500.0, 0.001)
dtau = tau_grid[1] - tau_grid[0]
aphi_tau = aphi(tau_grid)
bphi_tau = bphi(tau_grid)

ndraws = 10000
alpha_gamma = 2*kappa/sigma_lambda**2 * lambda_bar
beta_gamma = 2*kappa/sigma_lambda**2
rv = gamma_dist(alpha_gamma, loc = 0., scale = 1/beta_gamma)
lambda_rvs = rv.rvs(ndraws)
PD_rvs = np.zeros(lambda_rvs.shape[0])*np.nan
for i in range(ndraws):
    if i % 1000 == 0:
        print(f"{i} out of {ndraws}")
    PD_rvs[i] = np.sum(np.exp(aphi_tau + bphi_tau*lambda_rvs[i])*dtau)

log_PD_rvs = np.log(PD_rvs)
cape["LogPD"] = cape["LogCAPE_demean"] + np.mean(log_PD_rvs)


# 4. Looping over the adjusted CAPE and inverting Wachter (2013) PD ratio
#    to solve for disaster intensity lambda
lambda_invert = np.zeros(cape.shape[0])*np.nan
err_invert = np.zeros(cape.shape[0])*np.nan
for i, log_PD in enumerate(cape["LogPD"]):
    if i % 100 == 0:
        print(f"{i} out of {cape.shape[0]}")
    PD_target = np.exp(log_PD)
    if i == 0:
        lambda_iter = lambda_bar
    else:
        lambda_iter = lambda_invert[i-1]

    max_iter = 100
    err = 1e6
    it = 0
    while (err > 1e-8) and (it < max_iter):
        PD = np.sum(np.exp(aphi_tau + bphi_tau*lambda_iter)*dtau)
        gradient = np.sum(bphi_tau*np.exp(aphi_tau + bphi_tau*lambda_iter)*dtau)
        lambda_iter = lambda_iter - (PD - PD_target)/gradient
        err = np.abs(PD - PD_target)
        it += 1

    lambda_invert[i] = lambda_iter
    err_invert[i] = err

cape["lambda_invert_PD"] = lambda_invert

cape.set_index("date")["lambda_invert_PD"].apply(lambda x: np.max([x, 0])).plot()

# 5. Using JTIX and mapping to disaster intensity to get lambda
df_list = []
for days in [30, 60, 90, 120 ,150, 180]:
    df_list.append(pd.read_csv(f"data/interpolated_D/interpolated_disaster_spx_{days}.csv"))
SPX = pd.concat(df_list, ignore_index=True)
SPX = SPX[SPX["variable"] == "D_clamp"]
SPX["date"] = pd.to_datetime(SPX["date"])

muXQ = phi*muZ - phi*gamma*sigmaZ**2
sigmaXQ = phi*sigmaZ
Psi = np.exp(-0.5*(2*gamma*muZ - gamma**2 * sigmaZ**2))
Psi *= (1 + muXQ + 0.5*(muXQ**2 + sigmaXQ**2) - np.exp(muXQ + 0.5*sigmaXQ**2))
kappaQ = kappa - b*sigma_lambda

df_list = []
for days in [30, 90, 180]:
    tau = days/365.0
    a1 = 2*Psi/kappaQ * (1 - np.exp(-kappaQ*tau))
    a2 = 2*Psi * (lambda_bar*kappa/kappaQ * tau - lambda_bar * kappa/kappaQ**2 * (1-np.exp(-kappaQ*tau)))
    SPXsub = SPX[SPX["days"] == days]
    SPXsub[f"lambda_invert_JTIX_{days}"] = (SPXsub["value"] - a2)/a1
    df_list.append(SPXsub[["date", f"lambda_invert_JTIX_{days}"]])

lambda_JTIX = reduce(lambda df1, df2: pd.merge(df1, df2, on="date"), df_list)
lambda_JTIX.set_index("date")["lambda_invert_JTIX_180"].plot()

# Taking only end-of-month observations
lambda_JTIX_mon = lambda_JTIX.copy()
lambda_JTIX_mon["date_mon"] = lambda_JTIX_mon["date"] + pd.offsets.MonthEnd(0)
lambda_JTIX_mon = lambda_JTIX_mon.sort_values("date")
lambda_JTIX_mon = lambda_JTIX_mon.groupby("date_mon")[["lambda_invert_JTIX_30", "lambda_invert_JTIX_90", "lambda_invert_JTIX_180"]].mean().reset_index().rename(columns={"date_mon":"date"})

# 6. Plotting disaster intensity based on JTIX
fig, ax = plt.subplots(1, 1, figsize=(9,4.5))
lambda_JTIX_mon.rename(
    columns = {"lambda_invert_JTIX_30": "30 day JTIX","lambda_invert_JTIX_90": "90 day JTIX","lambda_invert_JTIX_180": "180 day JTIX",}
).set_index("date").plot(ax=ax, alpha=0.8)
plt.savefig("SS_figures/intensity_from_JTIX.pdf")

# 7. Comparing PD and JTIX based approaches:
fig, axs = plt.subplots(2, 1, figsize=(9,9))
cape[cape["date"] >= "1996-01-01"].set_index("date")["lambda_invert_PD"].plot(ax=axs[0], label="From PD", alpha=0.8, linewidth=1.5, linestyle="--", color="tab:blue")
cape[cape["date"] >= "1996-01-01"].set_index("date")["lambda_invert_PD"].apply(lambda x: np.max([x,0])).plot(ax=axs[0], label="From PD (set <0 to 0)", alpha=0.8, linewidth=3.0, color="tab:blue")
lambda_JTIX_mon.set_index("date")["lambda_invert_JTIX_180"].plot(ax=axs[0], label="From 180 day JTIX", alpha=0.8, linewidth=3.0, color="tab:orange")
axs[0].legend()
axs[0].set_xlabel("")

cape[cape["date"] >= "1996-01-01"].set_index("date")["lambda_invert_PD"].plot(ax=axs[1], label="From PD", alpha=0.8, linewidth=1.5, linestyle="--", color="tab:blue")
cape[cape["date"] >= "1996-01-01"].set_index("date")["lambda_invert_PD"].apply(lambda x: np.max([x,0])).plot(ax=axs[1], label="From PD (set <0 to 0)", alpha=0.8, linewidth=3.0, color="tab:blue")
lambda_JTIX_mon.set_index("date")["lambda_invert_JTIX_30"].plot(ax=axs[1], label="From 30 day JTIX", alpha=0.8, linewidth=3.0, color="tab:orange")
axs[1].legend()
axs[1].set_xlabel("")

fig.tight_layout()
plt.savefig("SS_figures/intensity_from_JTIX_vs_PD.pdf")
