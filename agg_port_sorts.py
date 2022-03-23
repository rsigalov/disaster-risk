from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
# import wrds
import crsp_comp
import rolling_disaster_betas as roll



# def main(argv = None):

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
roll_betas = pd.read_csv("data/disaster_risk_betas/disaster_risk_betas.csv")
# roll_betas_week = pd.read_csv("data/disaster_risk_betas/disaster_risk_betas_week.csv")

# Get last weekly rolling beta for each month:
# roll_betas_week["date"] = pd.to_datetime(roll_betas_week["date"])
# last_full_week = roll_betas_week[["date"]].drop_duplicates()
# last_full_week["date_mon"] = last_full_week["date"] + pd.offsets.MonthEnd(0)
# last_full_week["date_week_F"] = last_full_week["date"] + pd.DateOffset(days=4)
# last_full_week = last_full_week[last_full_week["date_week_F"] <= last_full_week["date_mon"]]
# last_full_week = last_full_week.sort_values("date")
# last_full_week = last_full_week.groupby("date_mon")["date"].last()
# roll_betas_week = roll_betas_week[roll_betas_week["date"].isin(list(last_full_week))]
# roll_betas_week["date"] = roll_betas_week["date"] + pd.offsets.MonthEnd(0)
# roll_betas_week.columns = ["permno", "date"] + [x + "_week" for x in roll_betas_week.columns[2:]]

# ------ For debugging: limit number of factors
# roll_betas = roll_betas.iloc[:, range(5)]
# ------

roll_betas.rename(columns = {"date_eom": "date"}, inplace = True)
roll_betas["date"] = pd.to_datetime(roll_betas["date"])
roll_betas = roll_betas[roll_betas.date >= "1997-07-31"]
roll_betas = roll_betas[roll_betas.date <= "2021-12-31"]

# roll_betas = pd.merge(roll_betas, roll_betas_week, on = ["permno", "date"], how = "left")

print("\n---- Computing portfolio characteristics ----\n")
# variable_list = [x for x in list(roll_betas.columns) if x not in ["beta_PC1_balanced", "permno", "date"]]
variable_list = [x for x in list(roll_betas.columns) if x not in ["permno", "date"]]
ports = crsp_comp.monthly_portfolio_sorts(None, roll_betas, variable_list, ncuts)

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
# beta_freq_list = ["mon", "week"]

level_dict = {}
variable_dict = {}
maturity_dict = {}
# beta_freq_dict = {}

for level in level_list:
	for variable in variable_list:
		for maturity in maturity_list:
			# for freq in beta_freq_list:
			level_dict["beta_" + level + "_" + variable + "_" + str(maturity)] = level # + "_" + freq] = level
			variable_dict["beta_" + level + "_" + variable + "_" + str(maturity)] = variable # + "_" + freq] = variable
			maturity_dict["beta_" + level + "_" + variable + "_" + str(maturity)] = maturity # + "_" + freq] = maturity
			# beta_freq_dict["beta_" + level + "_" + variable + "_" + str(maturity)] = freq # + "_" + freq] = freq


key_df = port_sort_ret[["variable"]].drop_duplicates().rename(columns={"variable":"old_variable"})
key_df = pd.DataFrame(key_df)
key_df["variable"] = key_df["old_variable"]
key_df["maturity"] = key_df["old_variable"]
key_df["level"] = key_df["old_variable"]
key_df = key_df.replace({"variable": variable_dict, "maturity": maturity_dict, "level": level_dict})

port_sort_ret = pd.merge(
	port_sort_ret.rename(columns={"variable": "old_variable"}), 
	key_df, on="old_variable", how="left")
port_sort_ret.drop(columns="old_variable", inplace=True)
port_sort_bm = pd.merge(
	port_sort_bm.rename(columns={"variable": "old_variable"}), 
	key_df, on="old_variable", how="left")
port_sort_bm.drop(columns="old_variable", inplace=True)
port_sort_op = pd.merge(
	port_sort_op.rename(columns={"variable": "old_variable"}), 
	key_df, on="old_variable", how="left")
port_sort_op.drop(columns="old_variable", inplace=True)
constituents_df = pd.merge(
	constituents_df.rename(columns={"variable": "old_variable"}), 
	key_df, on="old_variable", how="left")
constituents_df.drop(columns="old_variable", inplace=True)

# Saving results:
print("\n---- Saving Returns, BM and OP ----\n")
port_sort_ret.to_csv("data/disaster_sorts/port_beta_sort_ret.csv")
port_sort_bm.to_csv("data/disaster_sorts/port_beta_sort_bm.csv")
port_sort_op.to_csv("data/disaster_sorts/port_beta_sort_op.csv")
constituents_df.to_csv("data/disaster_sorts/port_beta_sort_const.csv")


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

# if __name__ == "__main__": 
# 	sys.exit(main(sys.argv))
