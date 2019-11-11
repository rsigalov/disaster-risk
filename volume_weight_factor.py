"""
This script uses data on volume and open interest to construct
volume and interest weighted level clamped-D factors.
"""

from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import wrds
import time
from arch.univariate import ARX
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
from numpy.linalg import inv
import crsp_comp
import rolling_disaster_betas as roll

import statsmodels.formula.api as smf

def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.975)) & (x >= np.quantile(x, 0.025))])

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    ind_not_extreme = (d <= np.quantile(d, 0.975)) & (d >= np.quantile(d, 0.025))
    d = d[ind_not_extreme]
    w = w[ind_not_extreme]

    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


def main(argv = None):
	# Loading data on interpolated D for all matruties. Work with clamped-D only
	print("Loading interpolated D")
	print("")
	int_d = pd.DataFrame(columns = ["secid", "date", "D_clamp"])

	for days in [30, 60, 90, 120, 150, 180]:
	    print(days)
	    int_d_to_append = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
	    int_d_to_append = int_d_to_append[["secid", "date", "D_clamp"]]
	    int_d_to_append["days"] = days
	    int_d = int_d.append(int_d_to_append)

	int_d["date"] = pd.to_datetime(int_d["date"])
	int_d["date_mon"] = int_d["date"] + pd.offsets.MonthEnd(0)

	# Filtering to have at least 10 observations per month across all maturities
	# and averaging clamped-D across all maturities
	mean_cnt_df = int_d.groupby(["secid", "date_mon"])["D_clamp"].count().reset_index()
	mean_cnt_df = mean_cnt_df.rename({"D_clamp": "cnt"}, axis = 1)
	mean_cnt_df = mean_cnt_df[mean_cnt_df.cnt >= 10]
	int_d_filter = pd.merge(mean_cnt_df, int_d, on = ["secid", "date_mon"], how = "left")
	int_d_mean_mon = int_d_filter.groupby(["secid", "date_mon"])["D_clamp"].mean().reset_index()

	print("Loading Volume")
	print("")
	# Loading data on volume and open interest and calculating mean volume
	vol = pd.read_csv("estimated_data/option_volume/option_volume_1.csv")
	vol = vol.append(pd.read_csv("estimated_data/option_volume/option_volume_2.csv"))
	vol = vol.append(pd.read_csv("estimated_data/option_volume/option_volume_3.csv"))
	vol.rename(columns = {"open_inetrest":"open_interest"}, inplace = True)
	vol["obs_date"] = pd.to_datetime(vol["obs_date"])

	vol_daily_mean = vol.groupby(["secid", "obs_date"])["volume","open_interest"].mean().reset_index()
	vol_daily_mean["date_mon"] = vol_daily_mean["obs_date"] + pd.offsets.MonthEnd(0)
	vol_mon_mean = vol_daily_mean.groupby(["secid", "date_mon"])["volume","open_interest"].mean().reset_index()

	# Merging average disaster measure with volume and open_interest values
	int_d_mean_mon = pd.merge(int_d_mean_mon, vol_mon_mean, on = ["date_mon", "secid"], how = "left")

	# Calculating simple mean and volume/open interest weighted clamped-D and
	# combining them into one dataframe:
	level_op_int_w = int_d_mean_mon.groupby(["date_mon"]).apply(lambda x: wavg(x, "D_clamp", "open_interest"))
	level_volume_w = int_d_mean_mon.groupby(["date_mon"]).apply(lambda x: wavg(x, "D_clamp", "volume"))
	level = int_d_mean_mon.groupby(["date_mon"])["D_clamp"].apply(mean_with_truncation)

	level_df = pd.merge(level.rename("mean"), level_volume_w.rename("vol_weight"),
	                    left_index = True, right_index = True)
	level_df = pd.merge(level_df, level_op_int_w.rename("op_int_weight"),
	                    left_index = True, right_index = True)

	factors = level_df.diff()

	# Saving factors
	factors.to_csv("estimated_data/disaster-risk-series/volume_weight_factors.csv")


	factors = pd.read_csv("estimated_data/disaster-risk-series/volume_weight_factors.csv")
	factors["date_mon"] = pd.to_datetime(factors["date_mon"])
	factors = factors.set_index("date_mon")

	# Estimating rolling betas:
	# Calculating rollin betas with respect to volume weighted thing
	print("Calculating Rolling Betas")
	print("")
	# db = wrds.Connection(wrds_username = "rsigalov")
	db = None
	crsp = crsp_comp.get_monthly_returns(
	    db, start_date = '1986-01-01', end_date = '2017-12-31', balanced = True)

	crsp_with_fac = pd.merge(
	    crsp, factors, left_on = ['date_eom'], right_index = True, how = 'left')

	for i, f in enumerate(factors.columns):
	    print("Factor %d out of %d"%(i+1, len(factors.columns)))
	    beta_f = crsp_with_fac.groupby('permno')['ret', f].\
	                apply(roll.RollingOLS, 24, 18)
	    crsp_with_fac = crsp_with_fac.join(beta_f)

	# Saving betas:
	output_df = crsp_with_fac[['permno', 'date_eom'] + \
	                          ['beta_' + x for x in factors.columns]]

	output_df.to_csv('estimated_data/disaster_risk_betas/disaster_risk_betas_volume.csv', index = False)

	ncuts = 5
	db = None
	# Constructing portfolio sorts:
	columns = ["variable"]
	columns = columns + ["ew_count"] + ["ew_" + str(x+1) for x in range(ncuts)]
	columns = columns + ["vw_count"] + ["vw_" + str(x+1) for x in range(ncuts)]
	port_sort_ret = pd.DataFrame(columns = columns)
	port_sort_bm = pd.DataFrame(columns = columns)
	port_sort_op = pd.DataFrame(columns = columns)

	print("Loading rolling betas")
	roll_betas = pd.read_csv('estimated_data/disaster_risk_betas/disaster_risk_betas_volume.csv')
	roll_betas.rename(columns = {"date_eom": "date"}, inplace = True)
	roll_betas["date"] = pd.to_datetime(roll_betas["date"])
	roll_betas = roll_betas[roll_betas.date >= "1997-07-31"]
	roll_betas = roll_betas[roll_betas.date <= "2015-11-30"]

	print("Computing portfolio characteristics")
	variable_list = [x for x in list(roll_betas.columns) if x not in ["beta_PC1_balanced", "permno", "date"]]
	ports = crsp_comp.monthly_portfolio_sorts(db, roll_betas, variable_list, ncuts)

	# List to save constituents data:
	constituents_df_list = []

	print("Generating output tables")
	for variable in variable_list:
		to_append = ports[variable]["ret"]
		to_append["variable"] = variable
		port_sort_ret = port_sort_ret.append(to_append)

		to_append = ports[variable]["bm"]
		to_append["variable"] = variable
		port_sort_bm = port_sort_bm.append(to_append)

		to_append = ports[variable]["op"]
		to_append["variable"] = variable
		port_sort_op = port_sort_op.append(to_append)

		to_append_const = ports[variable]["constituents"]
		to_append_const["variable"] = variable
		to_append_const.columns = ["form_date", "permno", "port", "variable"]
		constituents_df_list.append(to_append_const)

	constituents_df = pd.concat(constituents_df_list)

	print("Saving Returns, BM and OP")
	port_sort_ret.to_csv("estimated_data/disaster_sorts/port_sort_ret_agg_volume.csv")
	port_sort_bm.to_csv("estimated_data/disaster_sorts/port_sort_bm_agg_volume.csv")
	port_sort_op.to_csv("estimated_data/disaster_sorts/port_sort_op_agg_volume.csv")


	df_port_agg = pd.read_csv("estimated_data/disaster_sorts/port_sort_ret_agg_volume.csv").rename(columns = {"Unnamed: 0":"date"})
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
	reg_var_list = ["ew_" + str(x+1) for x in range(5)] + ["ew_diff"] + ["vw_" + str(x+1) for x in range(5)] + ["vw_diff"]

	name_list = []
	results_list = []
	for variable in variable_list:
	    sub_df = df_port_agg[df_port_agg.variable == variable]
	    sub_df = pd.merge(sub_df, ff, left_on = "date", right_index = True)

	    if sub_df.shape[0] > 0:
	        for reg_var in reg_var_list:
	            reg_df = sub_df[[reg_var, "MKT", "SMB", "HML", "CMA", "RMW"]]
	            results_list.append(
	                    smf.ols(formula = reg_var + " ~ 1",
	                            data = reg_df * 12).fit())
	            name_list.append((variable, reg_var, 0))

	            results_list.append(
	                    smf.ols(formula = reg_var + " ~ MKT",
	                            data = reg_df * 12).fit())
	            name_list.append((variable, reg_var, 1))

	            results_list.append(
	                    smf.ols(formula = reg_var + " ~ MKT + SMB + HML",
	                            data = reg_df * 12).fit())
	            name_list.append((variable, reg_var, 3))

	            results_list.append(
	                    smf.ols(formula = reg_var + " ~ MKT + SMB + HML + CMA",
	                            data = reg_df * 12).fit())
	            name_list.append((variable, reg_var, 4))

	            results_list.append(
	                    smf.ols(formula = reg_var + " ~ MKT + SMB + HML + CMA + RMW",
	                            data = reg_df * 12).fit())
	            name_list.append((variable, reg_var, 5))


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
	name_df.columns = ["variable", "port", "FF"]
	reg_res_df = pd.concat([name_df, reg_res_df], axis = 1)
	reg_res_df.drop(columns = "name", inplace = True)
	reg_res_df.to_csv("estimated_data/disaster_sorts/reg_results_agg_volume.csv", index = False)




if __name__ == "__main__": sys.exit(main(sys.argv))
