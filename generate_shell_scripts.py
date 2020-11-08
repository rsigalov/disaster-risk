from __future__ import print_function
from __future__ import division
import sys

# Standard libraries
import numpy as np
import pandas as pd
import wrds

# Library to parse from the command line
from optparse import OptionParser

# To shuffle companies in random order
import random

def main(argv = None):
	########################################################################
	# List ot parameters inside the script since don't run 
	# this file many times
	
	# First and last years of upload:
	begin_year = 2018
	end_year = 2019

	# suffix for the output files
	base_name = "new_release"

	# Number of companies in a group. For couple of years 90 is fine. However,
	# to estimate the whole series it is recommended to use 30 to minimize 
	# possible issues with WRDS SQL server
	group_size = 90

	# Delete intermediate files with raw option data?
	delete_raw_option_files = True
	########################################################################

	# Opening connection to WRDS SQL server
	db = wrds.Connection(wrds_username = "rsigalov")

	# OptionMetrics provides a type of underlying on which the option
	# is traded. In some cases, however, it incorrectly identifies it.
	# In particular, some underlyings that are not classified as
	# common shares in fact fall under the common share definition from
	# CRSP. Therefore, we use the information from both OM and CRSP to
	# identify the set of underlying that are common shares.
	 
	# 1. Use OM common shares identifier
	query = """
	select secid, cusip, issue_type
	from OPTIONM.SECURD
	"""
	query = query.replace('\n', ' ').replace('\t', ' ')
	df_issue_type = db.raw_sql(query)

	om_cs_list = df_issue_type[
		(df_issue_type.issue_type == "0") | 
		(df_issue_type.issue_type == 0)]["secid"]

	# 2. Use CRSP common share identifier
	query = """
	select distinct
		permno
	from crspa.msenames
	where shrcd in (10,11) and exchcd in (1,2,3)
	"""
	query = query.replace('\n', ' ').replace('\t', ' ')
	df_crsp_cs = db.raw_sql(query)

	# Getting all secids from the linking table that have
	# a match in CRSP
	om_crsp_link = pd.read_csv("om_crsp_wrds_linking_table.csv")
	om_crsp_link = om_crsp_link[om_crsp_link.score == 1]
	crsp_cs_list = om_crsp_link[om_crsp_link["PERMNO"].isin(df_crsp_cs["permno"])]["secid"]

	# Joining the two lists and dropping duplicates:
	companies_to_estimate = list(om_cs_list) + list(crsp_cs_list)
	companies_to_estimate = list(set(companies_to_estimate))

	########################################################################

	# Shuffling companies before generating shell files. Since some
	# companies have more options (in terms of strikes or maturity)
	# we want to make the expected running time of each part similar
	random.seed(19960202)
	random.shuffle(companies_to_estimate)

	# Splitting into 3 groups to estimate on 3 AWS machines:
	companies_to_estimate_1 = []
	companies_to_estimate_2 = []
	companies_to_estimate_3 = []

	for i_secid, secid in enumerate(companies_to_estimate):
	    if i_secid % 3 == 0:
	        companies_to_estimate_1.append(secid)
	    elif i_secid % 3 == 1:
	        companies_to_estimate_2.append(secid)
	    else:
	        companies_to_estimate_3.append(secid)

	# Group 1
	secid_to_estimate = companies_to_estimate_1
	top_comps = len(secid_to_estimate)
	group_size = 90
	groups = int(top_comps/group_size)
	start_index = 1
	name_prefix = base_name + "_part_1"

	f = open("shell_scripts/sh_" + base_name + "_1", "w")
	for i_group in range(groups):
		f.write("sudo python loading_data.py -s ")
		f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
		f.write(" -b " + str(begin_year) + " -e " + str(2019) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		if delete_raw_option_files:
			f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
			f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
		f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_new_release.txt\n")
		f.write("\n")

	    
	residual_companies = companies_to_estimate_1[(i_group+1)*group_size:]

	# Group 2
	secid_to_estimate = companies_to_estimate_2
	top_comps = len(secid_to_estimate)
	group_size = 90
	groups = int(top_comps/group_size)
	start_index = 1
	name_prefix = base_name + "_part_2"

	f = open("shell_scripts/sh_" + base_name + "_2", "w")
	for i_group in range(groups):
		f.write("sudo python loading_data.py -s ")
		f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
		f.write(" -b " + str(begin_year) + " -e " + str(2019) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		if delete_raw_option_files:
			f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
			f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
		f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_new_release.txt\n")
		f.write("\n")
	    
	residual_companies = companies_to_estimate_2[(i_group+1)*group_size:]

	# Group 3
	secid_to_estimate = companies_to_estimate_3
	top_comps = len(secid_to_estimate)
	group_size = 90
	groups = int(top_comps/group_size)
	start_index = 1
	name_prefix = base_name + "_part_3"

	f = open("shell_scripts/sh_" + base_name + "_3", "w")
	for i_group in range(groups):
		f.write("sudo python loading_data.py -s ")
		f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
		f.write(" -b " + str(begin_year) + " -e " + str(2019) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		if delete_raw_option_files:
			f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
			f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
		f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_new_release.txt\n")
		f.write("\n")
	    
	residual_companies = companies_to_estimate_3[(i_group+1)*group_size:]

	start_index = 1
	name_prefix = base_name + "_residual"

	f.write("sudo python loading_data.py -s ")
	f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
	f.write(" -b " + str(begin_year) + " -e " + str(2019) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
	f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
	f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
	if delete_raw_option_files:
		f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
		f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
	f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_new_release.txt\n")
	f.write("\n")

if __name__ == "__main__":
	sys.exit(main(sys.argv))




