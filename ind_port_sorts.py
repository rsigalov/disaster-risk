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
import data_funcs

# def main(argv=None):
# Settin up the connection to load CRSP and Compustat data
db = wrds.Connection()
# Main variables
variable_list = ["D_clamp", "rn_prob_20", "rn_prob_80"]
days_list = [30,60,90,120,150,180]
ncuts = 5

# Loading data on CS interpolated D that is already merged with CRSP PERMNO.
# Filtering companies to have at leat 5/10 observations to include in a
# portfolio
int_d_5 = pd.DataFrame(
	columns = ["secid", "permno", "date", "days"] + variable_list)
int_d = pd.DataFrame(
	columns = ["secid", "permno", "date", "days"] + variable_list)

for i, days in enumerate(days_list):
    print("Loading data on %d days variable" % days)
    print("")
    int_d_tmp = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
    int_d_tmp = int_d_tmp[int_d_tmp.permno.notnull()]
    int_d_tmp["days"] = days
    int_d = int_d.append(int_d_tmp)
    
    int_d_tmp = data_funcs.filter_min_obs_per_month(int_d_tmp, "D_clamp", 5)
    int_d_tmp["days"] = days
    int_d_5 = int_d_5.append(int_d_tmp)

# Filtering the series for the aggregated measure by requiring a company
# to have at least 10 observations a month:
print("Filtering and Averaging the data")
print("")
int_d_10 = data_funcs.filter_min_obs_per_month(int_d, "D_clamp", 10)

# Averaging on (1) secid-month-days level and (2) secid-month level:
d_mean_mon_days = int_d_5.groupby(["secid", "date_eom", "days"])[variable_list].mean()
d_mean_mon = int_d_10.groupby(["secid", "date_eom"])[variable_list].mean()

# Getting last permno for a given month:
permno_last_5 = int_d_5.groupby(["secid", "date_eom", "days"])["permno"].last()
permno_last_10 = int_d_10.groupby(["secid", "date_eom"])["permno"].last()

# Merging permno to disaster measure:
d_mean_mon_days = pd.merge(
	d_mean_mon_days, permno_last_5, left_index = True, right_index = True).reset_index()
d_mean_mon = pd.merge(
	d_mean_mon, permno_last_10, left_index = True, right_index = True).reset_index()

d_mean_mon_days = d_mean_mon_days.rename(columns = {"date_eom":"date"})
d_mean_mon = d_mean_mon.rename(columns = {"date_eom":"date"})

print("Starting to form portfolios")
print("")
# Setting up output dataset to append data as we go
columns = ["days", "variable"]
columns = columns + ["ew_count"] + ["ew_" + str(x+1) for x in range(ncuts)]
columns = columns + ["vw_count"] + ["vw_" + str(x+1) for x in range(ncuts)]
port_sort_ret = pd.DataFrame(columns = columns)
port_sort_bm = pd.DataFrame(columns = columns)
port_sort_op = pd.DataFrame(columns = columns)

# First do the portfolios for measures separated by days:
db = wrds.Connection()

for days in days_list:
	print("Doing portfolios sorts on %d days variable" % days)
	ports = crsp_comp.monthly_portfolio_sorts(
		db, d_mean_mon_days[d_mean_mon_days.days == days], variable_list, ncuts)
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

# Second, do the portfolios for aggregated measure:
ports = crsp_comp.monthly_portfolio_sorts(db, d_mean_mon, variable_list, ncuts)
for variable in variable_list:
	to_append = ports[variable]["ret"]
	to_append["days"] = -99
	to_append["variable"] = variable
	port_sort_ret = port_sort_ret.append(to_append)

	to_append = ports[variable]["bm"]
	to_append["days"] = -99
	to_append["variable"] = variable
	port_sort_bm = port_sort_bm.append(to_append)

	to_append = ports[variable]["op"]
	to_append["days"] = -99
	to_append["variable"] = variable
	port_sort_op = port_sort_op.append(to_append)

# Saving results:
print("Saving data")
port_sort_ret.to_csv("estimated_data/disaster_sorts/port_sort_ret_ind.csv")
port_sort_bm.to_csv("estimated_data/disaster_sorts/port_sort_bm_ind.csv")
port_sort_op.to_csv("estimated_data/disaster_sorts/port_sort_op_ind.csv")


# if __name__ == "__main__": sys.exit(main(sys.argv))
