"""
This script does portfolio sorts on individual company disaster measures
for different number of days and variables. It combines EW and VW returns,
EW and VW Book-to-Market, EW and VW Operating Profitability for different
maturities and measures into one file and saves them to

	estimated_data/ind_disaster_sorts/

folder
"""

from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import wrds
import crsp_comp

def main(argv=None):
	# Settin up the connection to load CRSP and Compustat data
	db = wrds.Connection()
	# Main variables
	variable_list = ["D_clamp", "rn_prob_20", "rn_prob_80"]
	days_list = [30,60,90,120,150,180]
	ncuts = 5

	# Setting up output dataset to append data as we go
	columns = ["days", "variable"]
	columns = columns + ["ew_count"] + ["ew_" + str(x+1) for x in range(ncuts)]
	columns = columns + ["vw_count"] + ["vw_" + str(x+1) for x in range(ncuts)]
	port_sort_ret = pd.DataFrame(columns = columns)
	port_sort_bm = pd.DataFrame(columns = columns)
	port_sort_op = pd.DataFrame(columns = columns)

	# Loading CRSP-OM linking table:
	print("Loading data on month-by-month CRSP-OM link")
	crsp_om_link = pd.read_csv("CRSP_OM_link_by_month.csv")
	crsp_om_link["date"] = pd.to_datetime(crsp_om_link["date"])

	# Finally using the average term structure as a sorting variable:
	print("Portfolio sorts on average term structure")
	ts = pd.read_csv("estimated_data/interpolated_D/average_term_structure.csv").rename(columns = {"date_mon":"date"})
	ts["date"] = pd.to_datetime(ts["date"])
	ts_mean = ts.groupby(["secid", "date"])["D_clamp"].mean().reset_index()
	ts_mean = pd.merge(ts_mean, crsp_om_link, on = ["secid", "date"])

	ports = crsp_comp.monthly_portfolio_sorts(db, ts_mean, ["D_clamp"], ncuts)
	to_append = ports["D_clamp"]["ret"]
	to_append["days"] = -99
	to_append["variable"] = "D_clamp"
	port_sort_ret = port_sort_ret.append(to_append)

	to_append = ports["D_clamp"]["bm"]
	to_append["days"] = -99
	to_append["variable"] = "D_clamp"
	port_sort_bm = port_sort_bm.append(to_append)

	to_append = ports["D_clamp"]["op"]
	to_append["days"] = -99
	to_append["variable"] = "D_clamp"
	port_sort_op = port_sort_op.append(to_append)

	for days in days_list:
		print("Doing portfolios sorts on %d days variable" % days)
		disaster_df = pd.read_csv("estimated_data/interpolated_D/mon_ind_disaster_days_" + str(days) + ".csv")
		disaster_df["date"] = pd.to_datetime(disaster_df["date"])
		disaster_df = pd.merge(disaster_df, crsp_om_link, on = ["secid", "date"])
		ports = crsp_comp.monthly_portfolio_sorts(db, disaster_df, variable_list, ncuts)
		# Appending observations to output dataframe
		for variable in variable_list:
			to_append = ports[variable]["ret"]
			to_append["days"] = days
			to_append["variable"] = variable
			port_sort_ret = port_sort_ret.append(to_append)

			to_append = ports[variable]["bm"]
			to_append["days"] = days
			to_append["variable"] = variable
			port_sort_bm = port_sort_bm.append(to_append)

			to_append = ports[variable]["op"]
			to_append["days"] = days
			to_append["variable"] = variable
			port_sort_op = port_sort_op.append(to_append)

	# Saving results:
	print("Saving data")
	port_sort_ret.to_csv("estimated_data/disaster_sorts/port_sort_ret.csv")
	port_sort_bm.to_csv("estimated_data/disaster_sorts/port_sort_bm.csv")
	port_sort_op.to_csv("estimated_data/disaster_sorts/port_sort_op.csv")
			
			
if __name__ == "__main__": sys.exit(main(sys.argv))

