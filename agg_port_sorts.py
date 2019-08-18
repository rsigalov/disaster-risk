from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import wrds
import crsp_comp
import rolling_disaster_betas as roll



def main(argv = None):
	db = wrds.Connection()
	ncuts = 5

	columns = ["variable"]
	columns = columns + ["ew_count"] + ["ew_" + str(x+1) for x in range(ncuts)]
	columns = columns + ["vw_count"] + ["vw_" + str(x+1) for x in range(ncuts)]
	port_sort_ret = pd.DataFrame(columns = columns)
	port_sort_bm = pd.DataFrame(columns = columns)
	port_sort_op = pd.DataFrame(columns = columns)

	print("Loading rolling betas")
	roll_betas = pd.read_csv("estimated_data/disaster_risk_betas/disaster_risk_betas.csv")
	# roll_betas = pd.read_csv("estimated_data/disaster_risk_betas/disaster_risk_betas_extrapolation.csv")
	roll_betas.rename(columns = {"date_eom": "date"}, inplace = True)
	roll_betas["date"] = pd.to_datetime(roll_betas["date"])
	roll_betas = roll_betas[roll_betas.date >= "1997-07-31"]

	print("Computing portfolio characteristics")
	variable_list = [x for x in list(roll_betas.columns) if x not in ["beta_PC1_balanced", "permno", "date"]]
	ports = crsp_comp.monthly_portfolio_sorts(db, roll_betas, variable_list, ncuts)

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

	# Dealing with names:
	print("Replacing names")
	level_list = ["ind_cs", "ind_all", "crsp_cs", "union_cs"]
	variable_list = ["D_clamp", "rn_prob_20", "rn_prob_80"]
	days_list = [30,60,90,120,150,180]

	level_dict = {}
	variable_dict = {}
	days_dict = {}

	for variable in variable_list:
		for days in days_list:
			for level in level_list:
				level_dict["beta_" + level + "_" + variable + "_" + str(days)] = level
				variable_dict["beta_" + level + "_" + variable + "_" + str(days)] = variable
				days_dict["beta_" + level + "_" + variable + "_" + str(days)] = days

	# Adding entries for dictionaries for level factors:
	days_dict["beta_level_D_clamp"] = -99
	variable_dict["beta_level_D_clamp"] = "D_clamp"
	level_dict["beta_level_D_clamp"] = "union_cs"

	port_sort_ret["days"] = port_sort_ret["variable"]
	port_sort_bm["days"] = port_sort_bm["variable"]
	port_sort_op["days"] = port_sort_op["variable"]

	port_sort_ret["level"] = port_sort_ret["variable"]
	port_sort_bm["level"] = port_sort_bm["variable"]
	port_sort_op["level"] = port_sort_op["variable"]

	port_sort_ret = port_sort_ret.replace({"variable": variable_dict, "days": days_dict, "level": level_dict})
	port_sort_bm = port_sort_bm.replace({"variable": variable_dict, "days": days_dict, "level": level_dict})
	port_sort_op = port_sort_op.replace({"variable": variable_dict, "days": days_dict, "level": level_dict})

	# Saving results:
	print("Saving results")

	port_sort_ret.to_csv("estimated_data/disaster_sorts/port_sort_ret_agg.csv")
	port_sort_bm.to_csv("estimated_data/disaster_sorts/port_sort_bm_agg.csv")
	port_sort_op.to_csv("estimated_data/disaster_sorts/port_sort_op_agg.csv")


if __name__ == "__main__": sys.exit(main(sys.argv))
