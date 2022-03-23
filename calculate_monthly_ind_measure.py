from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import psycopg2
import crsp_comp
import data_funcs
from functools import reduce

def main(argv=None):
	########################################################################
	# Inputs for portfolio construction
	variable = argv[1] #"D_clamp"
    # variable = "D_clamp"
	days_list = [30, 60, 90, 120, 150, 180]
	########################################################################

	# Loading OptionMetrics-CRSP linking table to assign permnos
	om_crsp_link = pd.read_csv("data/linking_files/om_crsp_wrds_linking_table.csv")
	om_crsp_link = om_crsp_link[(om_crsp_link["sdate"].notnull()) & (om_crsp_link["edate"].notnull())]
	om_crsp_link["sdate"] = pd.to_datetime(om_crsp_link["sdate"].astype(int), format="%Y%m%d")
	om_crsp_link["edate"] = pd.to_datetime(om_crsp_link["edate"].astype(int), format="%Y%m%d")
	int_d_list = []

	# Loading interpolated dindividual disaster measures at different maturities
	for i, days in enumerate(days_list):
		print(f"Loading data on {days} days variable\n")
		int_d_tmp = pd.read_csv(f"data/interpolated_D/interpolated_disaster_individual_{days}.csv")
		int_d_tmp = int_d_tmp[int_d_tmp["variable"] == variable]

		# Merging with secid-permno linking table:
		int_d_tmp = pd.merge(int_d_tmp, om_crsp_link, on="secid", how="left")
		int_d_tmp = int_d_tmp[
			(int_d_tmp["date"] >= int_d_tmp["sdate"]) & 
			(int_d_tmp["date"] <= int_d_tmp["edate"]) &
			(int_d_tmp["score"] <= 5)]

		int_d_tmp.drop(columns=["sdate", "edate", "score"], inplace=True)
		int_d_tmp.rename(columns={"PERMNO": "permno"}, inplace=True)
		int_d_tmp = int_d_tmp[int_d_tmp.permno.notnull()]
		int_d_list.append(int_d_tmp)

	int_d = pd.concat(int_d_list, ignore_index=True)
	del(int_d_list)

	# Filtering in two different ways:
	#   (1) Require at least 5 observations for a maturity sepcific measure
	#   (2) Require at least 10 observations for a level measure
	print("Filtering and Averaging the data")
	print("")
	int_d_5 = data_funcs.filter_min_obs_per_month(int_d, "value", 5)
	d_mean_mon_days = int_d_5.groupby(["secid", "date_eom", "days"])["value"].mean()
	# Getting last permno for a given month:
	permno_last_5 = int_d_5.groupby(["secid", "date_eom", "days"])["permno"].last()
	del(int_d_5)

	int_d_10 = data_funcs.filter_min_obs_per_month(int_d, "value", 10)
	d_mean_mon = int_d_10.groupby(["secid", "date_eom"])["value"].mean()
	# Getting last permno for a given month:
	permno_last_10 = int_d_10.groupby(["secid", "date_eom"])["permno"].last()
	del(int_d_10)
	del(int_d)

	# Merging permno to disaster measure:
	d_mean_mon_days = pd.merge(
		d_mean_mon_days, permno_last_5, left_index = True, right_index = True).reset_index()
	d_mean_mon = pd.merge(
		d_mean_mon, permno_last_10, left_index = True, right_index = True).reset_index()

	d_mean_mon_days = d_mean_mon_days.rename(columns = {"date_eom": "date"})
	d_mean_mon = d_mean_mon.rename(columns = {"date_eom": "date"})

	# Saving company specific monthly averaged measures
	d_mean_mon_days.to_csv(f"data/sorting_variables/monthly_average_{variable}_by_maturity.csv", index=False)
	d_mean_mon.to_csv(f"data/sorting_variables/monthly_average_{variable}.csv", index=False)

if __name__ == "__main__":
    sys.exit(main(sys.argv))