"""
Calculating returns on beta sorted portfolios and analyzing their turnover
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
pd.set_option('display.max_columns', None)

os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

####################################################################
#
####################################################################


#level_list = ["ind"]
level_list = ["sp_500_OM", "sp_500_CME"]
days_list = [40, 100]
#days_list = [30, 120]
var_list = ["D_clamp"]

port_mean_list = []
i = 0
for level in level_list:
    for days in days_list:
        for var in var_list:
            i += 1
            port_mean = pd.read_csv("/Users/rsigalov/Dropbox/2019_Revision/estimated_data/portfolios/beta_" + level + "_" + var + "_" + str(days) + "_N_ret.csv").mean()
            ew_mean = np.array(port_mean.iloc[1:6])
            vw_mean = np.array(port_mean.iloc[7:12])
        
            if i == 1:
                port_mean_df = pd.DataFrame(
                        {"port": range(1,6,1), "ret": ew_mean, "type": "EW",
                         "var": var, "level": level, "days": days})
                port_mean_df = port_mean_df.append(pd.DataFrame(
                        {"port": range(1,6,1), "ret": vw_mean, "type": "VW",
                         "var": var, "level": level, "days": days}))
            else:
                port_mean_df = port_mean_df.append(pd.DataFrame(
                        {"port": range(1,6,1), "ret": ew_mean, "type": "EW",
                         "var": var, "level": level, "days": days}))
                port_mean_df = port_mean_df.append(pd.DataFrame(
                        {"port": range(1,6,1), "ret": vw_mean, "type": "VW",
                         "var": var, "level": level, "days": days}))    

pd.pivot_table(port_mean_df, index = "port", columns = ["level", "type", "days"])
pd.pivot_table(port_mean_df.drop("days", axis = 1), index = "port", columns = ["level", "type"]).plot()

# Downloading data on monthly 5 factors from French's website:
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

def load_FF():
    resp = urlopen("http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    f = open("ff.csv", "w")
    indicator = False
    counter = 0
    for line in zipfile.open(zipfile.namelist()[0]).readlines():
        line_text = line.decode('utf-8')
    
        if line_text[0] == ",":
            indicator = True
        
        if line_text[0] == "\r":
            indicator = False
            counter += 1
            
        if not indicator and counter == 2:
            break
        
        if indicator:
            f.write(line_text)
            
    f.close()
    
    ff = pd.read_csv("ff.csv")
    ff = ff.rename({ff.columns[0]: "date"}, axis = 1)
    ff = ff.rename({"Mkt-RF": "MKT"}, axis = 1)
    ff["date"] = pd.date_range(start = "1963-07-01", freq = "M", periods = ff.shape[0])
    ff = ff.set_index("date")/100
    
    return ff

ff = load_FF()

########################################################################
# ff dataframe is ready for work now

# Loading Cremers factors:
cremers_mon = pd.read_csv("data/cremers_factors.csv")
cremers_mon["date"]= pd.to_datetime(cremers_mon["date"])
cremers_mon["date"] = cremers_mon["date"] + pd.offsets.MonthEnd(0)
cremers_mon = (cremers_mon.set_index("date")+1).groupby("date").prod()-1

# Loading portfolio return data and running regressions
level = "ind"
var = "D_clamp"
days = 180

port_ret = pd.read_csv("/Users/rsigalov/Dropbox/2019_Revision/estimated_data/portfolios/beta_" + level + "_" + var + "_" + str(days) + "_N_ret.csv")

port_ret = pd.merge(port_ret.set_index("date"),
                    ff, left_index = True, right_index = True, how = "left")
port_ret = pd.merge(port_ret, cremers.set_index("date"),
                    left_index = True, right_index = True, how = "left")

# Estimating 5 factor model for each portfolio:
vars_to_reg = ["ew_" + str(x) for x in range(1,6,1)] + ["vw_"+str(x) for x in range(1,6,1)]
results_list = []
for var in vars_to_reg:
    results_list.append(smf.ols(formula = var + " ~ MKT + SMB + HML + RMW + CMA", data = port_ret*12).fit())

pd.DataFrame({"port": vars_to_reg,
              "alpha": [x.params[0] for x in results_list],
              "se": [x.bse[0] for x in results_list],
              "R2": [x.rsquared for x in results_list]})

# Estimating 5 factor model for long-short portfolio:
port_ret["diff_ew"] = port_ret["ew_1"] - port_ret["ew_5"]
port_ret["diff_vw"] = port_ret["vw_1"] - port_ret["vw_5"]

vars_to_reg = ["diff_ew", "diff_vw"]
results_list = []
for var in vars_to_reg:
    results_list.append(smf.ols(formula = var + " ~ MKT + SMB + HML + RMW + CMA", data = port_ret*12).fit())

pd.DataFrame({"port": vars_to_reg,
              "alpha": [x.params[0] for x in results_list],
              "se": [x.bse[0] for x in results_list],
              "R2": [x.rsquared for x in results_list]})


# Calculating betas of each portfolio w.r.t. Cremers' JUMP factor
vars_to_reg = ["ew_" + str(x) for x in range(1,6,1)] + ["vw_"+str(x) for x in range(1,6,1)]
results_list = []
for var in vars_to_reg:
    results_list.append(smf.ols(formula = var + " ~ JUMP", data = port_ret*12).fit())

pd.DataFrame({"port": vars_to_reg,
              "alpha": [x.params[0] for x in results_list],
              "alpha_se": [x.bse[0] for x in results_list],
              "beta": [x.params[1] for x in results_list],
              "beta_se": [x.bse[1] for x in results_list],
              "R2": [x.rsquared for x in results_list]})
    
# Calcuating betas fo long-short portfolio wrt Cremers' JUMP factor:
vars_to_reg = ["diff_ew", "diff_vw"]
results_list_diff = []
for var in vars_to_reg:
    results_list_diff.append(smf.ols(formula = var + " ~ JUMP", data = port_ret*12).fit())

pd.DataFrame({"port": vars_to_reg,
              "alpha": [x.params[0] for x in results_list_diff],
              "alpha_se": [x.bse[0] for x in results_list_diff],
              "beta": [x.params[1] for x in results_list_diff],
              "beta_se": [x.bse[1] for x in results_list_diff],
              "R2": [x.rsquared for x in results_list_diff]})
    
smf.ols(formula = "JUMP ~ MKT + SMB + HML + RMW + CMA", data = port_ret*12).fit().summary()
    
    
########################################################################
# Calculate the beta of each of the portfolios with respect to our
# disaster measure and plot it against portfolios' average returns
########################################################################    
level = "ind"
var = "rn_prob_80"
days = 30

# Loading and subsetting data:
port_ret = pd.read_csv("/Users/rsigalov/Dropbox/2019_Revision/estimated_data/portfolios/beta_" + level + "_" + var + "_" + str(days) + "_N_ret.csv")
d_df = pd.read_csv("/Users/rsigalov/Dropbox/2019_Revision/estimated_data/disaster_risk_measures/combined_disaster_df.csv")
d_df = d_df[(d_df.level == level) & 
            (d_df["var"] == var) & 
            (d_df.days == days) & 
            (d_df.agg_type == "mean_all")].set_index("date")[["value"]].rename({"value":"D"}, axis = 1)

# For each portfolio calculating beta w.r.t. specified disaster measure 
# and its average return:
def calc_beta_av_ret(series):
    reg_df = pd.merge(series.rename("port"), d_df, left_index = True, right_index = True)
    results = smf.ols(formula = "port ~ D", data = reg_df).fit()
    
    return (results.params[1], np.mean(series)*12)

vars_to_reg = ["ew_" + str(x) for x in range(1,6,1)] + ["vw_" + str(x) for x in range(1,6,1)]
port_reg_df = port_ret.set_index("date")[vars_to_reg].apply(calc_beta_av_ret, axis = 0).apply(pd.Series)
port_reg_df.columns = ["beta", "mean_ret"]

port_reg_sub_df = port_reg_df.loc[["ew_" + str(x) for x in range(1,6,1)]]
plt.scatter(port_reg_sub_df.beta, port_reg_sub_df.mean_ret)
for i, txt in enumerate(port_reg_sub_df.index):
    plt.annotate(txt, (port_reg_sub_df.beta[i], port_reg_sub_df.mean_ret[i]))

port_reg_sub_df = port_reg_df.loc[["vw_" + str(x) for x in range(1,6,1)]]
plt.scatter(port_reg_sub_df.beta, port_reg_sub_df.mean_ret)
for i, txt in enumerate(port_reg_sub_df.index):
    plt.annotate(txt, (port_reg_sub_df.beta[i], port_reg_sub_df.mean_ret[i]))















