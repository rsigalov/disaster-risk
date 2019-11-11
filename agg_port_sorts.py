from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import wrds
import crsp_comp
import rolling_disaster_betas as roll



def main(argv = None):
	# db = wrds.Connection()
	db = None
	ncuts = 5

	# Setting up dataframes to output data on returns, book to market and
	# operatng profitability at formation
	columns = ["variable"]
	columns = columns + ["ew_count"] + ["ew_" + str(x+1) for x in range(ncuts)]
	columns = columns + ["vw_count"] + ["vw_" + str(x+1) for x in range(ncuts)]
	port_sort_ret = pd.DataFrame(columns = columns)
	port_sort_bm = pd.DataFrame(columns = columns)
	port_sort_op = pd.DataFrame(columns = columns)

	print("\n---- Loading rolling betas ----\n")
	roll_betas = pd.read_csv("estimated_data/disaster_risk_betas/disaster_risk_betas.csv")
	roll_betas_week = pd.read_csv("estimated_data/disaster_risk_betas/disaster_risk_betas_week.csv")

	# Get last weekly rolling beta for each month:
	roll_betas_week["date"] = pd.to_datetime(roll_betas_week["date"])
	last_full_week = roll_betas_week[["date"]].drop_duplicates()
	last_full_week["date_mon"] = last_full_week["date"] + pd.offsets.MonthEnd(0)
	last_full_week["date_week_F"] = last_full_week["date"] + pd.DateOffset(days=4)
	last_full_week = last_full_week[last_full_week["date_week_F"] <= last_full_week["date_mon"]]
	last_full_week = last_full_week.sort_values("date")
	last_full_week = last_full_week.groupby("date_mon")["date"].last()
	roll_betas_week = roll_betas_week[roll_betas_week["date"].isin(list(last_full_week))]
	roll_betas_week["date"] = roll_betas_week["date"] + pd.offsets.MonthEnd(0)
	roll_betas_week.columns = ["permno", "date"] + [x + "_week" for x in roll_betas_week.columns[2:]]
	
	# ------ For debugging: limit number of factors
	# roll_betas = roll_betas.iloc[:, range(5)]
	# ------
	
	roll_betas.rename(columns = {"date_eom": "date"}, inplace = True)
	roll_betas["date"] = pd.to_datetime(roll_betas["date"])
	roll_betas = roll_betas[roll_betas.date >= "1997-07-31"]
	roll_betas = roll_betas[roll_betas.date <= "2017-12-31"]

	roll_betas = pd.merge(roll_betas, roll_betas_week, on = ["permno", "date"], how = "left")

	print("\n---- Computing portfolio characteristics ----\n")
	variable_list = [x for x in list(roll_betas.columns) if x not in ["beta_PC1_balanced", "permno", "date"]]
	ports = crsp_comp.monthly_portfolio_sorts(db, roll_betas, variable_list, ncuts)

	# List to save constituents data:
	constituents_df_list = []

	print("\n---- Generating output tables ----\n")
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

	# Dealing with names:
	print("\n---- Replacing names ----\n")
	level_list = ["Ind", "SPX"]
	variable_list = ["D_clamp", "rn_prob_20", "rn_prob_80", "rn_prob_5", "rn_prob_20"]
	maturity_list = ["level",30,60,90,120,150,180]
	beta_freq_list = ["mon", "week"]

	level_dict = {}
	variable_dict = {}
	maturity_dict = {}
	beta_freq_dict = {}

	for level in level_list:
		for variable in variable_list:
			for maturity in maturity_list:
				for freq in beta_freq_list:
					level_dict["beta_" + level + "_" + variable + "_" + str(maturity) + "_" + freq] = level
					variable_dict["beta_" + level + "_" + variable + "_" + str(maturity) + "_" + freq] = variable
					maturity_dict["beta_" + level + "_" + variable + "_" + str(maturity) + "_" + freq] = maturity
					beta_freq_dict["beta_" + level + "_" + variable + "_" + str(maturity) + "_" + freq] = freq

	# # Adding entries for dictionaries for level factors:
	# days_dict["beta_level_D_clamp"] = -99
	# variable_dict["beta_level_D_clamp"] = "D_clamp"
	# level_dict["beta_level_D_clamp"] = "union_cs"

	# # Adding entries for dictionaries for level factors:
	# days_dict["beta_emil"] = -99
	# variable_dict["beta_emil"] = "D"
	# level_dict["beta_emil"] = "emil"

	# # Adding entries for dictionaries for level factors:
	# days_dict["beta_spx_disaster"] = -99
	# variable_dict["beta_spx_disaster"] = "D_spx"
	# level_dict["beta_spx_disaster"] = "spx"

	port_sort_ret["maturity"] = port_sort_ret["variable"]
	port_sort_bm["maturity"] = port_sort_bm["variable"]
	port_sort_op["maturity"] = port_sort_op["variable"]
	constituents_df["maturity"] = constituents_df["variable"]

	port_sort_ret["level"] = port_sort_ret["variable"]
	port_sort_bm["level"] = port_sort_bm["variable"]
	port_sort_op["level"] = port_sort_op["variable"]
	constituents_df["level"] = constituents_df["variable"]

	port_sort_ret["beta_freq"] = port_sort_ret["variable"]
	port_sort_bm["beta_freq"] = port_sort_bm["variable"]
	port_sort_op["beta_freq"] = port_sort_op["variable"]
	constituents_df["beta_freq"] = constituents_df["variable"]

	port_sort_ret = port_sort_ret.replace({"variable": variable_dict, "maturity": maturity_dict, "level": level_dict, "beta_freq": beta_freq_dict})
	port_sort_bm = port_sort_bm.replace({"variable": variable_dict, "maturity": maturity_dict, "level": level_dict, "beta_freq": beta_freq_dict})
	port_sort_op = port_sort_op.replace({"variable": variable_dict, "maturity": maturity_dict, "level": level_dict, "beta_freq": beta_freq_dict})
	constituents_df = constituents_df.replace({"variable": variable_dict, "maturity": maturity_dict, "level": level_dict, "beta_freq": beta_freq_dict})

	# Saving results:
	print("\n---- Saving Returns, BM and OP ----\n")
	port_sort_ret.to_csv("estimated_data/disaster_sorts/port_sort_ret.csv")
	port_sort_bm.to_csv("estimated_data/disaster_sorts/port_sort_bm.csv")
	port_sort_op.to_csv("estimated_data/disaster_sorts/port_sort_op.csv")


	#### Need to understand why WRDS doesn't work before proceeding ####
	# print("\n---- Adding Industry and Market Value for Constituents ----\n")
	# 
	# pivoting data for constituents to save:
	# constituents_df["port"] = constituents_df["port"].astype(float)
	# constituents_df = pd.pivot_table(
	# 	constituents_df, index = ["permno", "form_date"], values = "port", columns = ["level", "variable", "maturity"])
	# constituents_df.columns = constituents_df.columns.to_flat_index()

	# # Adding industry to companies in the portfolio sorts:
	# constituents_df = crsp_comp.get_ff_ind(db, constituents_df.reset_index())

	# # Getting market value for (permno,date):
	# crsp_ret = crsp_comp.get_monthly_returns(db, "1996-01-01", "2017-12-31")
	# crsp_ret = crsp_ret[["date", "permno", "permco_mktcap"]].rename(columns = {"permco_mktcap": "mktcap"})
	# crsp_ret.loc[:, 'date'] = crsp_ret.loc[:, 'date'] + pd.offsets.MonthEnd(0)
	# constituents_df = pd.merge(
	# 	constituents_df, crsp_ret,
	# 	left_on = ["permno", "form_date"], right_on = ["permno", "date"], how = "left")

	# print("\n---- Saving Constituents ----\n")
	# constituents_df.to_csv("estimated_data/disaster_sorts/port_sort_const.csv", index = False)

if __name__ == "__main__": sys.exit(main(sys.argv))
