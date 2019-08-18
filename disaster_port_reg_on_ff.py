"""

"""

import numpy as np
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

########################################################v
# Running FF portfolio regressions on FF factors:

#sort_type = "agg"
sort_type = "ind"

df_port_agg = pd.read_csv("estimated_data/disaster_sorts/port_sort_ret_" + sort_type + ".csv").rename(columns = {"Unnamed: 0":"date"})
df_port_agg["date"] = pd.to_datetime(df_port_agg["date"])
df_port_agg["ew_diff"] = df_port_agg["ew_1"] - df_port_agg["ew_5"]
df_port_agg["vw_diff"] = df_port_agg["vw_1"] - df_port_agg["vw_5"]

# Recession indicator:
df_port_agg["rec"] = np.where(
        (df_port_agg["date"] >= "2001-04-01") & (df_port_agg["date"] < "2001-12-01") |
        (df_port_agg["date"] >= "2008-01-01") & (df_port_agg["date"] < "2009-07-01"),1,0)

# Loading FF:
ff = crsp_comp.load_FF()

variable_list = np.unique(df_port_agg["variable"])
days_list = np.unique(df_port_agg["days"])
reg_var_list = ["ew_" + str(x+1) for x in range(5)] + ["ew_diff"] + ["vw_" + str(x+1) for x in range(5)] + ["vw_diff"]

name_list = []
results_list = []
for variable in variable_list:
    for days in days_list:
        sub_df = df_port_agg[(df_port_agg.variable == variable) & 
                             (df_port_agg.days == days)]
        sub_df = pd.merge(sub_df, ff, left_on = "date", right_index = True)
        
        if sub_df.shape[0] > 0:
            for reg_var in reg_var_list:
                reg_df = sub_df[[reg_var, "MKT", "SMB", "HML", "CMA", "RMW"]]
                results_list.append(
                        smf.ols(formula = reg_var + " ~ 1", 
                                data = reg_df * 12).fit())
                name_list.append((variable, days, reg_var, 0))                

                results_list.append(
                        smf.ols(formula = reg_var + " ~ MKT", 
                                data = reg_df * 12).fit())
                name_list.append((variable, days, reg_var, 1))
                
                results_list.append(
                        smf.ols(formula = reg_var + " ~ MKT + SMB + HML", 
                                data = reg_df * 12).fit())
                name_list.append((variable, days, reg_var, 3))
                
                results_list.append(
                        smf.ols(formula = reg_var + " ~ MKT + SMB + HML + CMA", 
                                data = reg_df * 12).fit())
                name_list.append((variable, days, reg_var, 4))
    
                results_list.append(
                        smf.ols(formula = reg_var + " ~ MKT + SMB + HML + CMA + RMW", 
                                data = reg_df * 12).fit())
                name_list.append((variable, days, reg_var, 5))


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
name_df.columns = ["variable", "days", "port", "FF"]
reg_res_df = pd.concat([name_df, reg_res_df], axis = 1)
reg_res_df.drop(columns = "name", inplace = True)
reg_res_df.to_csv("estimated_data/disaster_sorts/reg_results_" + sort_type + ".csv", index = False)

################################################################
# Making tables
################################################################
variable = "D_clamp"

for days in [-99,30,180]:
    def construct_df_from_rows(row_list, colnames, row_names):
        df = pd.DataFrame(columns = colnames)
        for i_row, row in enumerate(row_list):
            df = df.append(pd.DataFrame(dict(zip(colnames, list(row))), index = [row_names[i_row]]))
        return df
            
    # Table with mean returns and other statistics on portfolios:
    # Statistics from the regression on a constant: mean and SE
    port_list = ["ew_1", "ew_2", "ew_3", "ew_4", "ew_5", "ew_diff"]
    mean_stat_df = reg_res_df[
            (reg_res_df.FF == 0) & 
            (reg_res_df.variable == variable) &
            (reg_res_df.days == days) &
            reg_res_df.port.isin(port_list)] \
            .drop(columns = ["days", "variable", "FF"]) \
            .dropna(axis = 1)
    
    mean_list = np.array(mean_stat_df.alpha)
    se_list = np.array(mean_stat_df.alpha_se)
    
    # Calculating standard deviation and sharpe ratio:
    N = df_port_agg.date.drop_duplicates().shape[0]
    sd = np.array(df_port_agg[
            (df_port_agg.variable == variable) &
            (df_port_agg.days == days)][port_list].std())*np.sqrt(12)
    sharpe = mean_list/sd
    
    # Constructing a table:
    char_list = ["Mean", "SE", "SD", "Sharpe"]
    sum_stats = construct_df_from_rows(
            [mean_list, se_list, sd, sharpe], 
            port_list, char_list)
    
    path = "estimated_data/disaster_sorts/port_chars_" + sort_type + "_" + variable + "_" + str(days) + ".tex"
    f = open(path, "w")
    f.write("\\begin{tabular}{lcccccc}\n")
    f.write("\\toprule \n")
    f.write(" & EW1 & EW2 & EW3 & EW4 & EW5 & EW1-EW5 \\\\ \n")
    f.write("\hline \\\\[-1.8ex] \n")
    for i_row in range(sum_stats.shape[0]):
        vars_to_write = [char_list[i_row]] + list(sum_stats.iloc[i_row])
        f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
        
    f.write("\\bottomrule \n")
    f.write("\end{tabular} \n")  
    f.close()
    
    
    # Constructing table with regressions on FF factors:
    for FF_num in [1,3,5]:
        mean_stat_df = reg_res_df[
                (reg_res_df.FF == FF_num) & 
                (reg_res_df.variable == variable) &
                (reg_res_df.days == days) &
                reg_res_df.port.isin(port_list)] \
                .drop(columns = ["days", "variable", "FF"]) \
                .dropna(axis = 1).set_index("port").T
        
        char_list = ["$\\alpha$", "", "MKT", "", "SMB", "", "HML", "", "CMA", "", "RMW", ""]
        path = "estimated_data/disaster_sorts/port_ff_regs_" + sort_type + "_" + variable + "_" + str(days) + "_" + str(FF_num) + ".tex"
        f = open(path, "w")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule \n")
        f.write(" & EW1 & EW2 & EW3 & EW4 & EW5 & EW1-EW5 \\\\ \n")
        f.write("\hline \\\\[-1.8ex] \n")
        for i_factor in range(FF_num + 1):
            vars_to_write = [char_list[2*(i_factor)]] + list(mean_stat_df.iloc[2*(i_factor)])
            f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*vars_to_write))
            
            vars_to_write = [char_list[2*(i_factor) + 1]] + list(mean_stat_df.iloc[2*(i_factor) + 1])
            f.write("{} & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
            
        f.write("\hline \\\\[-1.8ex] \n")
        vars_to_write = ["$R^2$"] + list(mean_stat_df.iloc[-1])
        f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
        
        f.write("\\bottomrule \n")
        f.write("\end{tabular}")  
        f.close()
        
        
    