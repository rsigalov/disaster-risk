import numpy as np
from numpy.random import normal
from numpy.linalg import inv, cholesky
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from statsmodels.iolib.summary2 import summary_col # For summarizing regression results
import os
from matplotlib import pyplot as plt
from functools import reduce
from stargazer.stargazer import Stargazer
os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")
import crsp_comp
import EAPv2 as EAP

import wrds
import crsp_comp

########################################################v
# Running FF portfolio regressions on FF factors:

df_port_agg = pd.read_csv("estimated_data/disaster_sorts/port_sort_ret.csv").rename(columns = {"Unnamed: 0":"date"})
df_port_agg["date"] = pd.to_datetime(df_port_agg["date"])

df_port_agg.rename(columns = {"ew_1":  "ew_5", "ew_2":"ew_4","ew_3":"ew_3","ew_4":"ew_2","ew_5":"ew_1"}, inplace = True)
df_port_agg.rename(columns = {"vw_1":"vw_5", "vw_2":"vw_4","vw_3":"vw_3","vw_4":"vw_2","vw_5":"vw_1"}, inplace = True)

# Subtracting the risk free rate from all portfolios:
FF = crsp_comp.load_FF()
RF = FF[["RF"]]
df_port_agg = pd.merge(df_port_agg, RF, left_on = "date", right_index = True, how = "left")
for port_name in ["ew_" + str(x +1) for x in range(5)] + ["vw_" + str(x +1) for x in range(5)]:
    df_port_agg.loc[:,port_name] = df_port_agg.loc[:,port_name] - df_port_agg.loc[:,"RF"]
df_port_agg.drop(columns = "RF", inplace = True)

df_port_agg["ew_diff"] = df_port_agg["ew_5"] - df_port_agg["ew_1"]
df_port_agg["vw_diff"] = df_port_agg["vw_5"] - df_port_agg["vw_1"]

# NBER recession indicator:
df_port_agg["rec"] = np.where(
        (df_port_agg["date"] >= "2001-04-01") & (df_port_agg["date"] < "2001-12-01") |
        (df_port_agg["date"] >= "2008-01-01") & (df_port_agg["date"] < "2009-07-01"), 1, 0)

# Loading FF:
ff = crsp_comp.load_FF()

level_list = np.unique(df_port_agg["level"])
variable_list = np.unique(df_port_agg["variable"])
maturity_list = np.unique(df_port_agg["maturity"])
reg_var_list = ["ew_" + str(x+1) for x in range(5)] + ["ew_diff"] + ["vw_" + str(x+1) for x in range(5)] + ["vw_diff"]

name_list = []
results_list = []
for level in level_list:
    for variable in variable_list:
        for maturity in maturity_list:
            sub_df = df_port_agg[(df_port_agg.variable == variable) & 
                                 (df_port_agg.maturity == maturity) &
                                 (df_port_agg.level == level)]
            sub_df = pd.merge(sub_df, ff, left_on = "date", right_index = True)
            
            if sub_df.shape[0] > 0:
                for reg_var in reg_var_list:
                    reg_df = sub_df[[reg_var, "MKT", "SMB", "HML", "CMA", "RMW"]]
                    regressors_list = ["~1", "~MKT", "~MKT+SMB+HML","~MKT+SMB+HML+CMA", " ~MKT+SMB+HML+CMA+RMW"]
                    regressors_num_list = [0,1,3,4,5]
                    for i, regressors in enumerate(regressors_list):
                        results_list.append(
                                smf.ols(formula = reg_var + regressors, 
                                        data = reg_df * 12).fit(cov_type='HC3'))
                        name_list.append((level, variable, maturity, reg_var, regressors_num_list[i]))                

# Constructing a dataset with alphas:
reg_res_df = pd.DataFrame({"name":name_list,
                           "alpha":[x.params[0] if len(x.params) >= 1 else None for x in results_list],
                           "alpha_se":[x.bse[0] if len(x.params) >= 1 else None for x in results_list],
                           "beta_MKT":[x.params[1] if len(x.params) >= 2 else None for x in results_list],
                           "beta_MKT_se":[x.bse[1] if len(x.bse) >= 2 else None for x in results_list],
                           "beta_SMB":[x.params[2] if len(x.params) >= 3 else None for x in results_list],
                           "beta_SMB_se":[x.bse[2] if len(x.bse) >= 3 else None for x in results_list],
                           "beta_HML":[x.params[3] if len(x.params) >= 4 else None for x in results_list],
                           "beta_HML_se":[x.bse[3] if len(x.bse) >= 4 else None for x in results_list],
                           "beta_CMA":[x.params[4] if len(x.params) >= 5 else None for x in results_list],
                           "beta_CMA_se":[x.bse[4] if len(x.bse) >= 5 else None for x in results_list],
                           "beta_RMW":[x.params[5] if len(x.params) >= 6 else None for x in results_list],
                           "beta_RMW_se":[x.bse[5] if len(x.bse) >= 6 else None for x in results_list],
                           "R2":[x.rsquared for x in results_list]})

name_df = reg_res_df.name.apply(pd.Series)
name_df.columns = ["level","variable", "maturity", "port", "FF"]
reg_res_df = pd.concat([name_df, reg_res_df], axis = 1)
reg_res_df.drop(columns = "name", inplace = True)
reg_res_df.to_csv("estimated_data/disaster_sorts/reg_results.csv", index = False)

# Running similar regressions for Cremers factor
df_port_agg = pd.read_csv("estimated_data/disaster_sorts/port_sort_cremers_jump_ret.csv").rename(columns = {"Unnamed: 0":"date"})
df_port_agg["date"] = pd.to_datetime(df_port_agg["date"])

df_port_agg["ew_diff"] = df_port_agg["ew_5"] - df_port_agg["ew_1"]
df_port_agg["vw_diff"] = df_port_agg["vw_5"] - df_port_agg["vw_1"]
df_port_agg = pd.merge(df_port_agg, FF, left_on = "date", right_index = True)

results_list = []
name_list = []
regressors_list = ["~1", "~MKT", "~MKT+SMB+HML","~MKT+SMB+HML+CMA", " ~MKT+SMB+HML+CMA+RMW"]
regressors_num_list = [0,1,3,4,5]
for port in reg_var_list:
    for i_regressor, regressor in enumerate(regressors_list):
        results_list.append(smf.ols(formula = port + regressor, data = df_port_agg.set_index("date")*12).fit(cov_type='HC3'))
        name_list.append((port, regressors_num_list[i_regressor]))  

# Constructing a dataset with alphas:
cremers_reg_res_df = pd.DataFrame({"name":name_list,
                           "alpha":[x.params[0] if len(x.params) >= 1 else None for x in results_list],
                           "alpha_se":[x.bse[0] if len(x.params) >= 1 else None for x in results_list],
                           "beta_MKT":[x.params[1] if len(x.params) >= 2 else None for x in results_list],
                           "beta_MKT_se":[x.bse[1] if len(x.bse) >= 2 else None for x in results_list],
                           "beta_SMB":[x.params[2] if len(x.params) >= 3 else None for x in results_list],
                           "beta_SMB_se":[x.bse[2] if len(x.bse) >= 3 else None for x in results_list],
                           "beta_HML":[x.params[3] if len(x.params) >= 4 else None for x in results_list],
                           "beta_HML_se":[x.bse[3] if len(x.bse) >= 4 else None for x in results_list],
                           "beta_CMA":[x.params[4] if len(x.params) >= 5 else None for x in results_list],
                           "beta_CMA_se":[x.bse[4] if len(x.bse) >= 5 else None for x in results_list],
                           "beta_RMW":[x.params[5] if len(x.params) >= 6 else None for x in results_list],
                           "beta_RMW_se":[x.bse[5] if len(x.bse) >= 6 else None for x in results_list],
                           "R2":[x.rsquared for x in results_list]})

name_df = cremers_reg_res_df.name.apply(pd.Series)
name_df.columns = ["port","FF"]
cremers_reg_res_df = pd.concat([name_df, cremers_reg_res_df], axis = 1)
cremers_reg_res_df.drop(columns = "name", inplace = True)
cremers_reg_res_df.to_csv("estimated_data/disaster_sorts/reg_results_cremers.csv", index = False)




#################################################################
## Making tables
#################################################################
#reg_res_df = pd.read_csv("estimated_data/disaster_sorts/reg_results.csv")
#variable = "D_clamp"
#
#for days in [-99,30,180]:
#    def construct_df_from_rows(row_list, colnames, row_names):
#        df = pd.DataFrame(columns = colnames)
#        for i_row, row in enumerate(row_list):
#            df = df.append(pd.DataFrame(dict(zip(colnames, list(row))), index = [row_names[i_row]]))
#        return df
#            
#    # Table with mean returns and other statistics on portfolios:
#    # Statistics from the regression on a constant: mean and SE
#    port_list = ["ew_1", "ew_2", "ew_3", "ew_4", "ew_5", "ew_diff"]
#    mean_stat_df = reg_res_df[
#            (reg_res_df.FF == 0) & 
#            (reg_res_df.variable == variable) &
#            (reg_res_df.days == days) &
#            reg_res_df.port.isin(port_list)] \
#            .drop(columns = ["days", "variable", "FF"]) \
#            .dropna(axis = 1)
#    
#    mean_list = np.array(mean_stat_df.alpha)
#    se_list = np.array(mean_stat_df.alpha_se)
#    
#    # Calculating standard deviation and sharpe ratio:
#    N = df_port_agg.date.drop_duplicates().shape[0]
#    sd = np.array(df_port_agg[
#            (df_port_agg.variable == variable) &
#            (df_port_agg.days == days)][port_list].std())*np.sqrt(12)
#    sharpe = mean_list/sd
#    
#    # Constructing a table:
#    char_list = ["Mean", "SE", "SD", "Sharpe"]
#    sum_stats = construct_df_from_rows(
#            [mean_list, se_list, sd, sharpe], 
#            port_list, char_list)
#    
#    path = "estimated_data/disaster_sorts/tables/port_chars_" + sort_type + "_" + variable + "_" + str(days) + ".tex"
#    f = open(path, "w")
#    f.write("\\begin{tabular}{lcccccc}\n")
#    f.write("\\toprule \n")
#    f.write(" & EW1 & EW2 & EW3 & EW4 & EW5 & EW1-EW5 \\\\ \n")
#    f.write("\hline \\\\[-1.8ex] \n")
#    for i_row in range(sum_stats.shape[0]):
#        vars_to_write = [char_list[i_row]] + list(sum_stats.iloc[i_row])
#        f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#        
#    f.write("\\bottomrule \n")
#    f.write("\end{tabular} \n")  
#    f.close()
#    
#    
#    # Constructing table with regressions on FF factors:
#    for FF_num in [1,3,5]:
#        mean_stat_df = reg_res_df[
#                (reg_res_df.FF == FF_num) & 
#                (reg_res_df.variable == variable) &
#                (reg_res_df.days == days) &
#                reg_res_df.port.isin(port_list)] \
#                .drop(columns = ["days", "variable", "FF"]) \
#                .dropna(axis = 1).set_index("port").T
#        
#        char_list = ["$\\alpha$", "", "MKT", "", "SMB", "", "HML", "", "CMA", "", "RMW", ""]
#        path = "estimated_data/disaster_sorts/tables/port_ff_regs_" + sort_type + "_" + variable + "_" + str(days) + "_" + str(FF_num) + ".tex"
#        f = open(path, "w")
#        f.write("\\begin{tabular}{lcccccc}\n")
#        f.write("\\toprule \n")
#        f.write(" & EW1 & EW2 & EW3 & EW4 & EW5 & EW1-EW5 \\\\ \n")
#        f.write("\hline \\\\[-1.8ex] \n")
#        for i_factor in range(FF_num + 1):
#            vars_to_write = [char_list[2*(i_factor)]] + list(mean_stat_df.iloc[2*(i_factor)])
#            f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*vars_to_write))
#            
#            vars_to_write = [char_list[2*(i_factor) + 1]] + list(mean_stat_df.iloc[2*(i_factor) + 1])
#            f.write("{} & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#            
#        f.write("\hline \\\\[-1.8ex] \n")
#        vars_to_write = ["$R^2$"] + list(mean_stat_df.iloc[-1])
#        f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#        
#        f.write("\\bottomrule \n")
#        f.write("\end{tabular}")  
#        f.close()
#        
#
#df_reg_sub = reg_res_df[
#        (reg_res_df.maturity == "level") & 
#        (reg_res_df.variable == "D_clamp") &
#        (reg_res_df.port == "ew_diff")][["level", "FF","alpha","alpha_se"]]
#df_reg_sub.sort_values(["FF","level"], inplace = True)
#
#col1_list = ["Average", "", "CAPM","", "3 Factors","", "3 Factors + CMA","", "5 Factors",""]
#
#path = "SS_tables/compare_alphas_individual_spx.tex"
#f = open(path, "w")
#f.write("\\begin{tabular}{lcc}\n")
#f.write("\\toprule \n")
#f.write(" & Individual & SPX \\\\ \n")
#f.write("\hline \\\\[-1.8ex] \n")
#i_col1 = 0
#for i_factor in [0,1,3,4,5]:
#    vars_to_write = [col1_list[i_col1]] + list(df_reg_sub[df_reg_sub.FF == i_factor].alpha)
#    f.write("{} & {:.3f} & {:.3f} \\\\ \n".format(*vars_to_write))    
#    i_col1+=1
#    
#    vars_to_write = [col1_list[i_col1]]+ list(df_reg_sub[df_reg_sub.FF == i_factor].alpha_se)
#    f.write("{} & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#    i_col1+=1
#    
#f.write("\\bottomrule \n")
#f.write("\end{tabular}")  
#f.close()   
#        
#        
#        
#######################################################################
## Calculating value weighted portfolios
##df_port = pd.read_csv("estimated_data/disaster_sorts/port_sort_const_agg_emil.csv")
##df_port = df_port[["permno", "form_date", "('D', -99)"]].rename(columns = {"form_date":"date", "('D', -99)": "port"})
##df_port["date"] = pd.to_datetime(df_port["date"])
##
##crsp_ret = pd.read_csv("crsp_ret.csv")
##crsp_ret = crsp_ret[["permno", "date_eom", "ret", "permco_mktcap"]].rename(columns = {"date_eom":"date", "permco_mktcap":"mktcap"})
##crsp_ret["date"] = pd.to_datetime(crsp_ret["date"])
##crsp_ret["next_mon_ret"] = crsp_ret.sort_values(["permno", "date"]).groupby(["permno"])["ret"].shift(-1)
##
### Merging datasets:
##df_port = pd.merge(df_port, crsp_ret[["permno", "date", "mktcap", "next_mon_ret"]],
##                   on = ["date", "permno"], how = "left")
##df_port = df_port.rename(columns = {"next_mon_ret":"ret"})
##
### Calculating value weighted returns 
##def wavg(group, avg_name, weight_name):
##    d = group[avg_name]
##    w = group[weight_name]
##    try:
##        return (d * w).sum() / w.sum()
##    except ZeroDivisionError:
##        return np.nan
##    
##port_ret_vw = df_port.groupby(["port", "date"]).apply(lambda x: wavg(x, "ret", "mktcap")).rename("ret").reset_index()
##port_ret_vw = pd.pivot_table(port_ret_vw, index = "date", columns = "port", values = "ret")
##port_ret_vw.columns = ["vw_" + str(x+1) for x in range(5)]
##port_ret_vw["vw_diff"] = port_ret_vw["vw_1"] - port_ret_vw["vw_5"]
##smf.ols(formula = "vw_diff ~ 1", data = port_ret_vw * 12).fit().summary()
##
##port_ret_vw = pd.merge(port_ret_vw, ff, left_index=True, right_index=True)
##
##smf.ols(formula = "vw_1 ~ MKT", data = port_ret_vw * 12).fit().summary()
##smf.ols(formula = "vw_5 ~ MKT", data = port_ret_vw * 12).fit().summary()
##
##
##
##
##
##
##port_ret_ew = df_port.groupby(["port", "date"])["ret"].mean().reset_index()
##port_ret_ew = pd.pivot_table(port_ret_ew, index = "date", columns = "port", values = "ret")
##port_ret_ew["ew_diff"] = port_ret_ew.loc[:,1] - port_ret_ew.loc[:,5]
##smf.ols(formula = "ew_diff ~ 1", data = port_ret_ew * 12).fit().summary()
##
##
##
### Looking at scatter plots:
##date_list = np.unique(df_port.date)
##corr_1_list = []
##corr_5_list = []
##for date in date_list:
##    sub_df = df_port[df_port.date == date]
##    corr_1_list.append(np.corrcoef(sub_df[sub_df.port == 1].mktcap, sub_df[sub_df.port == 1].ret)[0,1])
##    corr_5_list.append(np.corrcoef(sub_df[sub_df.port == 5].mktcap, sub_df[sub_df.port == 5].ret)[0,1])
##
##plt.plot(date_list, corr_1_list)
##plt.plot(date_list, corr_5_list)
##
##

# Comparing the noisiness of weekly measure with monthly
df = pd.read_csv("estimated_data/disaster_risk_measures/disaster_risk_measures.csv")
df["date"] = pd.to_datetime(df["date"])

df1 = df[(df.agg_freq == "date_mon") & (df.variable == "D_clamp") & (df.level == "Ind") &
         (df.maturity == "level")]
df2 = df[(df.agg_freq == "date_week") & (df.variable == "D_clamp") & (df.level == "Ind") &
         (df.maturity == "level")]

df1 = df1[["date", "value"]].set_index("date")
df2 = df2[["date", "value"]].set_index("date")

df1[(df1.index < "2004-01-01")].plot()
df2[(df2.index < "2004-01-01")].plot()

df1[(df1.index >= "2004-01-01") & (df1.index < "2010-01-01")].plot()
df2[(df2.index >= "2004-01-01") & (df2.index < "2010-01-01")].plot()

df1[(df1.index >= "2010-01-01")].plot()
df2[(df2.index >= "2010-01-01")].plot()

# Looking at the correlations of betas estimated w.r.t. weekly
# and monthly measures
bw = pd.read_csv("estimated_data/disaster_risk_betas/disaster_risk_betas_week.csv")
bm = pd.read_csv("estimated_data/disaster_risk_betas/disaster_risk_betas.csv")

bw = bw.rename(columns = {"beta_Ind_D_clamp_level":"beta_w"})
bw["date"] = pd.to_datetime(bw["date"])
bw["date_mon"] = bw["date"] + pd.offsets.MonthEnd(0)

bm = bm[["permno", "date_eom", "beta_Ind_D_clamp_level"]].rename(columns = {"beta_Ind_D_clamp_level":"beta_m", "date_eom":"date"})
bm["date"] = pd.to_datetime(bm["date"])




b = pd.merge(bm, bw, on = ["permno", "date"])

# Calculate number of joint observations:
joint_obs = b.groupby("permno").apply(lambda df: df.dropna().shape[0])

# Leaving only permnos with at least 24 months of data:
permno_corr = b.groupby("permno").apply(lambda df: df[["beta_m", "beta_w"]].corr().loc["beta_m", "beta_w"])

permno_corr = pd.merge(joint_obs.rename("obs"), permno_corr.rename("corr"), left_index = True, right_index = True)
permno_corr[permno_corr.obs >= 12].mean()

obsm = bm.groupby("permno").apply(lambda df: df.dropna().shape[0])


# Doing simulation for error-in-variables:
def generate_AR_1(T, N, phi): # phi is AR1, phi = sqrt(rho) in the problem set
    x = np.zeros((T,N))

    # Initializing AR(1) process:
    for n in range(N):
        nu = normal(0, 1, 2).reshape((2,1))
        F = np.array([[phi, 0],[0, 0]])
        V_p = inv(np.eye(4) - np.kron(F,F)).dot(np.ones((4,1)))
        V = V_p.reshape((2,2))
        CV = cholesky(V)
        z = CV.dot(nu)
        x_1 = z[0]

        x[0, n] = x_1

    # Generating subsequent random innovations to AR(1) process
    e = normal(0, 1, (N*(T-1))).reshape((T-1, N))

    # looping through periods 2,...,T to fill AR(1) process
    for n in range(N):
        for t in range(1,T):
            x[t, n] = phi * x[t-1, n] + e[t-1,n]
            
    return x


def simulate(T, varx, beta, phi):
    # Generate D variable
    D = generate_AR_1(T, 1, phi)
    D_err = varx*np.random.normal(0,1,T)
    
    # Generate return according to 'true' model:
    r = beta*D + np.random.normal(0,1,T)
    
    # Estimating a regression:
    reg_res = sm.OLS(r, sm.add_constant(D + D_err)).fit()
    return reg_res.params[1], reg_res.bse[1]










