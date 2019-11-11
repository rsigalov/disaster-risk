"""
This script looks at test asset from He, Kelly and Manela (2017)
to see how well the disaster factor prices the cross section
of portfolios across different asset classes
"""

import numpy as np
from numpy import vstack, hstack, sqrt, power, ones, zeros, eye, kron, mean, std, sum, nanmean, diag, abs
import pandas as pd
from numpy.linalg import inv
import statsmodels.api as sm
import statsmodels.formula.api as smf
from functools import reduce
from matplotlib import pyplot as plt

from scipy import optimize

import seaborn as sns
from functools import reduce
import crsp_comp

HKM_ports = pd.read_csv("data/He_Kelly_Manela_Factors_And_Test_Assets_monthly.csv")
HKM_ports.rename(columns = {"yyyymm":"date"}, inplace = True)
HKM_ports["date"] = pd.to_datetime(HKM_ports["date"].astype(int).astype(str) + "01", format = "%Y%m%d")
HKM_ports["date"] = HKM_ports["date"] + pd.offsets.MonthEnd(0)

level_D = pd.read_csv("estimated_data/disaster_risk_measures/disaster_risk_measures.csv")
level_D["date"] = pd.to_datetime(level_D["date"])
level_D = level_D[(level_D.variable == "D_clamp") & (level_D.level == "Ind") & (level_D.maturity == "level")]
level_D = level_D.set_index("date")["value"].rename("level_D")

# Comparing disaster measure and intermediary capital measure (both as difference factor)
level_D_HKM = pd.merge(level_D, HKM_ports.set_index("date")["intermediary_capital_ratio"], 
                       left_index = True,right_index = True)

def normalize(x):
    return (x - np.mean(x))/np.std(x)

sns.regplot(data = level_D_HKM.diff().apply(normalize), x = "intermediary_capital_ratio", y = "level_D")
smf.ols(formula = "intermediary_capital_ratio ~ level_D", data = level_D_HKM.diff().apply(normalize)).fit().summary()


# Going on to calculating the regressions for all portfolios
# 1. Calculating betas in time series regressions of each portfolio
#    on level disaster measure:
HKM_ports = pd.merge(HKM_ports, level_D.diff(), left_on = "date", right_index = True, left = True)
test_port_names = HKM_ports.columns[5:-1]
test_port_names = [x for x in test_port_names if "All_" not in x]

reg_list = []
for test_port in test_port_names:
    reg_list.append(smf.ols(formula = test_port + " ~ level_D", data = HKM_ports).fit())

# 2. Calculating average return on all portfolios:
port_mean_ret = np.array(HKM_ports[test_port_names].mean())
port_beta = pd.DataFrame({"beta": [x.params[1] for x in reg_list],
                          "mean_ret": port_mean_ret})
port_beta.index = test_port_names

# Assigning asset classes:
FF_base = ["mkt_rf", "smb", "hml"]
FF_sort = [x for x in test_port_names if "FF25" in x]
US_bonds = [x for x in test_port_names if "US_bonds_" in x]
Sov_bonds = [x for x in test_port_names if "Sov_bonds_" in x]
Options = [x for x in test_port_names if "Options_" in x]
CDS = [x for x in test_port_names if "CDS_" in x]
Commod = [x for x in test_port_names if "Commod_" in x]
FX = [x for x in test_port_names if "FX_" in x]
groups = [FF_base, FF_sort, US_bonds, Sov_bonds, Options, 
          CDS, Commod, FX]
group_name = ["FF_base", "FF_sort", "US_bonds", "Sov_bonds", "Options",
              "CDS", "Commod", "FX"]

port_beta["asset_class"] = None
for i in range(port_beta.shape[0]):
    port_name = port_beta.index[i]
    for i_group in range(len(groups)):
        if port_name in groups[i_group]:
            port_beta.loc[:,"asset_class"].iloc[i] = group_name[i_group]
            break

# Running a cross sectional regression:
cross_reg = smf.ols(formula = "mean_ret ~ beta - 1", data = port_beta).fit()
port_beta["predict_mean_ret"] = port_beta["beta"]*cross_reg.params[0]

sns.scatterplot(data = port_beta, x = "predict_mean_ret", y = "mean_ret", hue = "asset_class")


df_to_plot = port_beta[port_beta.asset_class == "Options"] #[~port_beta.isin(["Commod", "Options"])]
sns.scatterplot(data = df_to_plot, x = "beta", y = "mean_ret", hue = "asset_class")

sns.regplot(data = port_beta[port_beta.asset_class == "Options"], 
            x = "beta", y = "mean_ret")

smf.ols(formula = "mean_ret ~ beta", data = port_beta[port_beta.asset_class == "Options"]).fit().summary()

############################################################################
# Function for Fama MacBeth regression with corrected SEs from HKM (2017) 
def naExx(x):
    xnarows = np.isnan(x).any(axis=1)
    completex = x[~xnarows]
    naT = completex.shape[0]
    Exx = (1/naT) * completex.T @ completex
    return Exx, naT

def xsaptest(excessreturns, factors):
    # get dimensions
    T, n = excessreturns.shape
    k = factors.shape[1]

    # TS regression for betas
    beta = zeros((n, k + 1))
    epst = zeros((T, n))
    Xf = hstack((ones((T,1)), factors))

    for i in range(n):
        lm = sm.OLS(excessreturns[:,i], Xf, missing = "drop").fit()
        beta[i,:] = lm.params
        epst[:,i] = excessreturns[:,i] - Xf @ beta[i,:]

    Sigma, naT = naExx(epst)  

    # FM regressions
    X = hstack((ones((n,1)), beta[:,1:]))
    lambdat = zeros((T,k+1))
    for t in range(T):
        lm = sm.OLS(excessreturns[t,:], X).fit()
        lambdat[t,:] = lm.params


    # Fama-MacBeth point estimates and standard errors
    lambdaFM = zeros((k+1,1))
    selambdaFM = zeros((k+1,1))
    tlambdaFM = zeros((k+1,1))
    for f in range(k+1):
        notnanlambda = ~np.isnan(lambdat[:,f])
        nonmissinglambdatf = lambdat[notnanlambda,f]
        lambdaFM[f] = mean(nonmissinglambdatf)
        selambdaFM_i = power(nonmissinglambdatf - lambdaFM[f],2)
        selambdaFM_i = sum(selambdaFM_i)/T
        selambdaFM[f] = sqrt(selambdaFM_i)/sqrt(T)
        tlambdaFM[f] = lambdaFM[f]/selambdaFM[f]

    # We apply namean here, so that potentially the returns and factor means
    # are estimated over different time-periods, which is consistent with
    # the Fama-MacBeth treatment of the unbalanced panel
    Erx = nanmean(excessreturns, axis = 0).reshape((n,1))
    Ef = nanmean(factors, axis = 0).reshape((k,1))

    # pricing errors
    predEr = beta @ lambdaFM
    alpha = Erx - predEr

    # chi-squared statistic for pricing errors
    invXX = inv(X.T @ X)
    covalpha = (1/naT) * (eye(n)-X @ invXX @ X.T)*Sigma*(eye(n)-X @ invXX @ X.T).T
    chistat = alpha.T @ np.linalg.pinv(covalpha) @ alpha

    ### GMM standard errors a-la Cochrane (2005)

    # first add time-series moments
    ut = epst.copy()
    for i in range(k):
        ut = hstack((ut, epst * (factors[:,i].reshape((T,1)) @ ones((1,n)))))

    # now add cross sectional moments
    ut = hstack((ut, excessreturns - ones((T,1)) @ (predEr).T))
    
    # Dimension of ut: should be N + N*K + K
    utdim = ut.shape[1]

    # following usual advice to demean S. (Ols moments already are, but not cross sectional moments)
    ut_demeaned = ut - nanmean(ut, axis = 0).reshape((1,utdim))

    # Spectral density matrix
    S = naExx(ut_demeaned)[0]

    # GMM selection matrix
    a1 = hstack((eye(n*(k+1)), zeros((n*(k+1),n))))
    a2 = hstack((zeros((k+1,n*(k+1))), X.T))
    a = vstack((a1,a2))

    # GMM jacobian
    Eff = naExx(factors)[0]
    
    d1 = hstack((np.ones((1,1)), Ef.T))
    d2 = hstack((Ef, Eff))
    d = vstack((d1, d2))
    d = kron(d, eye(n)) # look at how kron works
    d1 = hstack((d, zeros((n*(k+1),k+1))))
    d2 = hstack((zeros((n,n)), kron(lambdaFM[1:,:].T, eye(n)), X))
    d = -vstack((d1,d2))

    # standard errors for λ
    
    sigma2gmm = (1/naT) * inv(a@d)@a@S@a.T@inv(a@d).T
    selambdaGMM = sqrt(diag(sigma2gmm)[-(k+1):])
    tlambdaGMM = lambdaFM.flatten() / selambdaGMM

#    rsquared = var(X*λ)/var(Erx)

    # mean absolute pricing errors (MAPE)
    mape = mean(abs(alpha))

    return beta, lambdaFM, tlambdaFM, tlambdaGMM, mape, chistat



to_est_df = HKM_ports[Options + ["mkt_rf", "intermediary_capital_risk_factor"]].dropna()
R = np.array(to_est_df[Options])
F = np.array(to_est_df[["mkt_rf", "intermediary_capital_risk_factor"]])
beta, lambdaFM, tlambdaFM, tlambdaGMM, mape, chistat = xsaptest(R, F)

groups = [FF_sort, US_bonds, Sov_bonds, Options, CDS, Commod, FX]
all_group = reduce(lambda arr1, arr2: arr1 + arr2, [FF_sort, US_bonds, Sov_bonds, Options, CDS, Commod])
groups = groups + [all_group]

R = np.array(HKM_ports[all_group])
F = np.array(HKM_ports[["mkt_rf", "intermediary_capital_risk_factor"]])
beta, lambdaFM, tlambdaFM, tlambdaGMM, mape, chistat = xsaptest(R, F)

################################################
# Replicating results in HKM (2017)

asset_class_lambda = {}
groups = [FF_sort, US_bonds, Sov_bonds, Options, CDS, Commod, FX]
all_group = reduce(lambda arr1, arr2: arr1 + arr2, groups)
groups = groups + [all_group]
group_names = ["FF_sort", "US_bonds", "Sov_bonds", "Options", "CDS", "Commod", "FX", "All"]


def estimate_prices_of_risk(df, risk_factor):
    factor_names = ["mkt_rf", risk_factor]
    
    for i_group, group in enumerate(groups):
        print("Group %s" % group_names[i_group])
        ports = df[group + factor_names]
        ports = ports.dropna()
        F = np.array(ports[factor_names])
        R = np.array(ports[group])
        
        beta, lambdaFM, tlambdaFM, tlambdaGMM, mape, chistat = xsaptest(R, F)
        res = {"lambdaFM": lambdaFM,
               "tlambdaFM": tlambdaFM,
               "tlambdaGMM": tlambdaGMM,
               "mape": mape,
               "chistat": chistat}
        asset_class_lambda[group_names[i_group]] = res
        
    mktrf_est = [asset_class_lambda[x]["lambdaFM"].flatten()[1] for x in asset_class_lambda.keys()]
    mktrf_t = [asset_class_lambda[x]["tlambdaGMM"].flatten()[1] for x in asset_class_lambda.keys()]
    capital_est = [asset_class_lambda[x]["lambdaFM"].flatten()[2] for x in asset_class_lambda.keys()]
    capital_t = [asset_class_lambda[x]["tlambdaGMM"].flatten()[2] for x in asset_class_lambda.keys()]
    
    res_df = pd.DataFrame({"mktrf_est":mktrf_est,
                  "mktrf_t":mktrf_t,
                  "capital_est":capital_est,
                  "capital_t":capital_t})
    res_df.index = group_names
    return res_df


# Using quarterly test assets:
HKM_ports = pd.read_csv("data/He_Kelly_Manela_Factors_And_Test_Assets.csv")
HKM_ports.rename(columns = {"yyyyq":"date"}, inplace = True)
HKM_ports["date"] = pd.to_datetime(HKM_ports["date"].astype(int).astype(str) + "01", format = "%Y%m%d")
HKM_ports["date"] = HKM_ports["date"] + pd.offsets.MonthEnd(0)
HKM_ports = HKM_ports.set_index("date")

# Merging with level factor
level_D = pd.read_csv("estimated_data/disaster_risk_measures/disaster_risk_measures.csv")
level_D["date"] = pd.to_datetime(level_D["date"])
level_D = level_D[(level_D.variable == "D_clamp") & (level_D.level == "Ind") & (level_D.maturity == "level")]
level_D = level_D.set_index("date")["value"].rename("level_D")

HKM_ports = pd.merge(HKM_ports, level_D, left_index = True, right_index = True, how = "left")
HKM_ports["disaster_risk_factor"] = HKM_ports["level_D"].diff()

quarterly_capital_df = estimate_prices_of_risk(HKM_ports, "intermediary_capital_risk_factor")
quarterly_disaster_df = estimate_prices_of_risk(HKM_ports, "disaster_risk_factor")

# Using monthly
HKM_ports = pd.read_csv("data/He_Kelly_Manela_Factors_And_Test_Assets_monthly.csv")
HKM_ports.rename(columns = {"yyyymm":"date"}, inplace = True)
HKM_ports["date"] = pd.to_datetime(HKM_ports["date"].astype(int).astype(str) + "01", format = "%Y%m%d")
HKM_ports["date"] = HKM_ports["date"] + pd.offsets.MonthEnd(0)
HKM_ports = HKM_ports.set_index("date")

HKM_ports = pd.merge(HKM_ports, level_D, left_index = True, right_index = True, how = "left")
HKM_ports["disaster_risk_factor"] = HKM_ports["level_D"].diff()

monthly_capital_df = estimate_prices_of_risk(HKM_ports, "intermediary_capital_risk_factor")
monthly_disaster_df = estimate_prices_of_risk(HKM_ports, "disaster_risk_factor")

# Creating a Table with asset class regressions:
path = "SS_tables/asset_class_price_of_risk.tex"
f = open(path, "w")
f.write("\small")
f.write("\\begin{tabular}{lccccccc}\n")
f.write("\\toprule \n")
f.write("  & Sorted FF & US Bonds & Sov. Bonds & Options & CDS & Commod & FX \\\\ \n")
f.write("\cline{2-8} \n")

panels_names = ["Risk Factor = Intermediary Capital, Quarterly",
                "Risk Factor = Intermediary Capital, Monthly",
                "Risk Factor = Disaster Risk, Quarterly",
                "Risk Factor = Disaster Risk, Monthly"]

for idf, df in enumerate([quarterly_capital_df, monthly_capital_df, quarterly_disaster_df, monthly_disaster_df]):

    f.write("\hline \\\\[-1.8ex] \n")
    f.write("\multicolumn{7}{l}{\\textbf{%s}} \\\\ \n" % panels_names[idf])
    f.write("\hline \\\\[-1.8ex] \n")
        
    f.write("Risk Factor & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  \\\\ \n".format(*list(df["capital_est"])))
    f.write(" & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(df["capital_t"])))
    
    f.write("MKT & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  \\\\ \n".format(*list(df["mktrf_est"])))
    f.write(" & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(df["mktrf_t"])))
    
f.write("\\bottomrule \n")
f.write("\end{tabular}")  
f.close()


FF = crsp_comp.load_FF()
# Using different portfolios:
ME_OP_VW = pd.read_csv("data/ME_OP_VW_portfolios.csv")
ME_OP_VW.rename(columns = {"yyyymm":"date"}, inplace = True)
ME_OP_VW["date"] = pd.to_datetime(ME_OP_VW["date"].astype(int).astype(str) + "01", format = "%Y%m%d")
ME_OP_VW["date"] = ME_OP_VW["date"] + pd.offsets.MonthEnd(0)
ME_OP_VW = ME_OP_VW.set_index("date")
test_ports = ME_OP_VW.columns
ME_OP_VW = pd.merge(ME_OP_VW, level_D, left_index = True, right_index = True, how = "left")
ME_OP_VW = pd.merge(ME_OP_VW, FF, left_index = True, right_index = True, how = "left")
ME_OP_VW["disaster_risk_factor"] = ME_OP_VW["level_D"].diff()
ME_OP_VW = ME_OP_VW.dropna()
beta, lambdaFM, tlambdaFM, tlambdaGMM, mape, chistat = xsaptest(np.array(ME_OP_VW[test_ports]), np.array(ME_OP_VW[["MKT", "disaster_risk_factor"]]))

plt.scatter(beta[:,2], np.mean(np.array(ME_OP_VW[test_ports]), axis = 0))






################################################################
# Calculating Fama-Mcbeth procedure with GMM correction for
# cross asset correlation and the fact that betas are estimated

#
#
#
#
#
#def estimate_spectral_density(u, lags = None):
#    dimu, T = u.shape
#    
#    Ktilde = K + 1
#    
#    lags = int(np.ceil(1.2 * float(T)**(1.0/3)))
#    w = 1 - np.arange(0,lags+1)/(lags+1)
#    
#    Gamma = np.zeros((lags+1, dimu, dimu))
#    for lag in range(lags+1):
#        Gamma[lag] = u[:,lag:] @ u[:,:T-lag].T
#
#    Gamma = Gamma/T
#    S = Gamma[0].copy()
#    for i in range(1,lags+1):
#        S = S + w[i] * (Gamma[i] + Gamma[i].T)
#    
#    return S
#
#to_est_df = HKM_ports[FF_sort + ["mkt_rf"]].dropna()
#R = np.array(to_est_df[FF_sort])
#F = np.array(to_est_df[["mkt_rf"]])
#
#
##def GMM_joint_beta_lambda(R, F):
## Getting dimensions:
#TR, N = R.shape
#TF, K = F.shape
#
## Transposing since it is easier to work with
#R = R.T
#F = F.T
#
## Checking if the same time dimension
#if TR != TF: 
#    raise ValueError("Time dimension of R and F should be the same!")
#else:
#    T = TR
#    
#
#def reshape_b(b):
#    # dividing b into groups of parameters and reshaping
#    a = np.reshape(b[:N], (N,1))
#    beta = b[N:(N+N*K)]
#    beta = np.reshape(beta, (N, K))
#    lambda_ = np.reshape(b[(N+N*K):], (K,1))
#    
#    return a, beta, lambda_
#    
## Function for moments
#def residuals(b):
#    a, beta, lambda_ = reshape_b(b)
#    
#    residuals1 = R - np.tile(a,(1, T)) - beta @ F
#    F_repeated = reduce(lambda arr1, arr2: np.vstack((arr1, arr2)),
#                        [np.tile(np.reshape(F[k,:],(1,T)),(N,1)) for k in range(K)])
#    residuals2 = np.tile(residuals1, (K,1)) * F_repeated
#    
#    residuals3 = R - beta @ lambda_
#    
#    residuals = np.vstack((residuals1, residuals2, residuals3))
#    
#    return residuals
#
#def moments(b):
#    return np.mean(residuals(b), axis = 1)
#
#def moment_derivative(b):
#    a, beta, lambda_ = reshape_b(b)
#    
#    F_mean = np.reshape(np.mean(F, axis = 1), (K,1))
#    F_var = (1/T) * F @ F.T
#    mder1 = np.hstack((-np.eye(N), -np.kron(np.eye(N), F_mean.T), np.zeros((N,K))))
#    mder2 = np.hstack((-np.kron(np.eye(N), F_mean), -np.kron(np.eye(N), F_var), np.zeros((N*K,K))))
#    mder3 = np.hstack((np.zeros((N,N)), -np.kron(np.eye(N), lambda_.T), np.zeros((N,K))))
#    mder = np.vstack((mder1,mder2,mder3))
#    
#    return mder
#
##def FOC(b, W):
##    return moment_derivative(b).T @ W @ moments(b)
#
#
## Estimating first stage:
#def to_min(b):
#    b = np.reshape(b, (N+N*K+K,1))
#    return moments(b).T @ np.eye(N+N*K+N) @ moments(b)
#
#x0 = np.zeros(N+N*K+K)
#gmm_min = optimize.minimize(
#        to_min, x0, tol = 1e-9, method = 'Nelder-Mead', 
#        options = {"disp":True, "maxiter":1e6})
#b_sol_1 = gmm_min["x"]
#u_1 = residuals(b_sol_1)
#S_hat_1 = estimate_spectral_density(u_1)
#
#a, beta, lambda_ = reshape_b(b_sol_1)
#
#
## Estimating second stage:
#def to_min(b):
#    b = np.reshape(b, (N+N*K+K,1))
#    return moments(b).T @ inv(S_hat_1) @ moments(b)
#
#gmm_min = optimize.minimize(to_min, x0, tol = 1e-8, options = {"disp":True})
#b_sol_2 = gmm_min["x"]
#u_2 = residuals(b_sol_2)
#S_hat_2 = estimate_spectral_density(u_2)
#
## Estimating estimate covariance matrix:
#d = moment_derivative(b_sol_2)
#vcov = (1/T) * np.inv(d.T @ np.inv(S_hat_2) @ d)
#    
#
#
#reg_res_list = []
#to_est_df = HKM_ports[Options + ["mkt_rf", "level_D"]]
#
#for port in Options:
#    reg_res_list.append(smf.ols(formula = port + "~mkt_rf+level_D", data = to_est_df).fit())
#    
#reg_res_df = pd.DataFrame({"mean": np.array(HKM_ports[Options].mean()),
#                           "beta_mkt": [x.params[1] for x in reg_res_list],
#                           "beta_d": [x.params[2] for x in reg_res_list]})
#
#smf.ols(formula = "mean~beta_mkt+beta_d-1", data = reg_res_df).fit().summary()




