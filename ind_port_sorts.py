"""
This script does portfolio sorts on individual company disaster measures
for different number of days and variables. It combines EW and VW returns,
EW and VW Book-to-Market, EW and VW Operating Profitability for different
maturities and measures into one file and saves them to

	data/disaster_sorts/

folder
"""

from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import psycopg2
import crsp_comp
import data_funcs
from functools import reduce

def connectToWRDS():
	# Setting up the connection to load CRSP and Compustat data (if was not loaded previously)
	with open("account_data/wrds_user.txt") as f:
		wrds_username = f.readline()

	with open("account_data/wrds_pass.txt") as f:
		wrds_password = f.readline()

	conn = psycopg2.connect(
		host="wrds-pgdata.wharton.upenn.edu",
		port = 9737,
		database="wrds",
		user=wrds_username,
		password=wrds_password)

	return conn

def main(argv=None):
	########################################################################
	# Inputs for portfolio construction
	# variable = "D_clamp"
	variable = argv[1]
	days_list = [30, 60, 90, 120, 150, 180]
	ncuts = 5 # number of portfolios
	########################################################################

	conn = connectToWRDS()

	# Loading company monthly averages
	d_mean_mon_days = pd.read_csv(f"data/sorting_variables/monthly_average_{variable}_by_maturity.csv")
	d_mean_mon = pd.read_csv(f"data/sorting_variables/monthly_average_{variable}.csv")
	d_mean_mon_days["date"] = pd.to_datetime(d_mean_mon_days["date"])
	d_mean_mon["date"] = pd.to_datetime(d_mean_mon["date"])

	# Setting up output dataset to append data as we go
	columns = ["days", "variable"]
	columns = columns + ["ew_count"] + ["ew_" + str(x+1) for x in range(ncuts)]
	columns = columns + ["vw_count"] + ["vw_" + str(x+1) for x in range(ncuts)]
	port_sort_ret = pd.DataFrame(columns = columns)
	port_sort_bm = pd.DataFrame(columns = columns)
	port_sort_op = pd.DataFrame(columns = columns)

	# List to save constituents data:
	constituents_df_list = []

	# First do the portfolios for measures separated by days:
	for days in days_list:
		print("\nPortfolios sorts on %d days variable" % days)
		ports = crsp_comp.monthly_portfolio_sorts(
			conn, d_mean_mon_days[d_mean_mon_days.days == days], ["value"], ncuts)

		# Appending observations to output dataframe
		to_append = ports["value"]["ret"]
		to_append["days"] = days
		to_append["variable"] = variable
		port_sort_ret = port_sort_ret.append(to_append)

		to_append = ports["value"]["bm"]
		to_append["days"] = days
		to_append["variable"] = variable
		port_sort_bm = port_sort_bm.append(to_append)

		to_append = ports["value"]["op"]
		to_append["days"] = days
		to_append["variable"] = variable
		port_sort_op = port_sort_op.append(to_append)

		to_append_const = ports["value"]["constituents"]
		to_append_const["days"] = days
		to_append_const["variable"] = variable
		to_append_const.columns = ["form_date", "permno", "port", "days", "variable"]
		constituents_df_list.append(to_append_const)

	# Second, do the portfolios for aggregated measure:
	ports = crsp_comp.monthly_portfolio_sorts(conn, d_mean_mon, ["value"], ncuts)
		
	to_append = ports["value"]["ret"]
	to_append["days"] = -99
	to_append["variable"] = variable
	port_sort_ret = port_sort_ret.append(to_append)

	to_append = ports["value"]["bm"]
	to_append["days"] = -99
	to_append["variable"] = variable
	port_sort_bm = port_sort_bm.append(to_append)

	to_append = ports["value"]["op"]
	to_append["days"] = -99
	to_append["variable"] = variable
	port_sort_op = port_sort_op.append(to_append)

	to_append_const = ports["value"]["constituents"]
	to_append_const["days"] = -99
	to_append_const["variable"] = variable
	to_append_const.columns = ["form_date", "permno", "port", "days", "variable"]
	constituents_df_list.append(to_append_const)

	# Merging all constituents lists together:
	constituents_to_save = pd.concat(constituents_df_list)

	# Saving results:
	print("Saving data")
	port_sort_ret.to_csv(f"data/disaster_sorts/port_sort_ret_ind_{variable}.csv")
	port_sort_bm.to_csv(f"data/disaster_sorts/port_sort_bm_ind_{variable}.csv")
	port_sort_op.to_csv(f"data/disaster_sorts/port_sort_op_ind_{variable}.csv")

	# pivoting data for constituents to save:
	constituents_to_save["port"] = constituents_to_save["port"].astype(float)
	constituents_to_save = pd.pivot_table(
		constituents_to_save, index = ["permno", "form_date"], values = "port", columns = ["variable", "days"])
	constituents_to_save.columns = constituents_to_save.columns.to_flat_index()
	constituents_to_save.to_csv(f"data/disaster_sorts/port_sort_const_ind_{variable}.csv")

	conn.close()

if __name__ == "__main__": 
	sys.exit(main(sys.argv))
