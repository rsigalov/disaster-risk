from __future__ import print_function
from __future__ import division
import sys

# Standard libraries
import numpy as np
import pandas as pd
import psycopg2

# Library to parse from the command line
# from optparse import OptionParser

# To shuffle companies in random order
import random

def main(argv = None):
	########################################################################
	# List ot parameters inside the script since don't run 
	# this file many times
	
	# First and last years of upload:
	begin_year = 2021
	end_year = 2021

	# suffix for the output files
	base_name = "march_2022_update"

	# Number of companies in a group. For couple of years 90 is fine. However,
	# to estimate the whole series it is recommended to use 30 to minimize 
	# possible issues with WRDS SQL server
	group_size = 90

	# parallel processes
	p = 4

	# Delete intermediate files with raw option data?
	delete_raw_option_files = False
	########################################################################

	# Opening connection to WRDS SQL server
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

	# Looping over years and getting all secids. Later will filter
	# them to common share
	secid_df_list = []
	for year in range(begin_year, end_year + 1, 1):
		query = f"select distinct secid from OPTIONM.OPPRCD{year}"
		secid_df_list.append(pd.read_sql_query(query, conn))

	secid_df = pd.concat(secid_df_list, ignore_index=True)
	secid_df = secid_df.drop_duplicates()
	print(secid_df.head())

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
	df_issue_type = pd.read_sql_query(query, conn)

	om_cs_list = df_issue_type[
		(df_issue_type.issue_type == "0") | 
		(df_issue_type.issue_type == 0)]["secid"]

	# 2. Use CRSP common share identifier
	query = """
	select distinct
		permno
	from crsp.msenames
	where shrcd in (10,11) and exchcd in (1,2,3)
	"""
	query = query.replace('\n', ' ').replace('\t', ' ')
	df_crsp_cs = pd.read_sql_query(query, conn)

	# Getting all secids from the linking table that have
	# a match in CRSP
	om_crsp_link = pd.read_csv("data/linking_files/om_crsp_wrds_linking_table.csv")
	om_crsp_link = om_crsp_link[om_crsp_link.score == 1]
	crsp_cs_list = om_crsp_link[om_crsp_link["PERMNO"].isin(df_crsp_cs["permno"])]["secid"]

	# Joining the two lists and dropping duplicates:
	cs_secids = list(om_cs_list) + list(crsp_cs_list)
	cs_secids = list(set(cs_secids))

	companies_to_estimate = list(secid_df[secid_df["secid"].isin(cs_secids)]["secid"])

	conn.close()

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
	residual_companies = []
	secid_to_estimate = companies_to_estimate_1
	top_comps = len(secid_to_estimate)
	group_size = 90
	groups = int(top_comps/group_size)
	start_index = 1
	name_prefix = base_name + "_part_1"

	with open("shell_scripts/sh_" + base_name + "_1.sh", "w") as f:
		f.write("python load_zcb.py\n")
		for i_group in range(groups):
			f.write("python -W ignore loading_data.py -s ")
			f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
			f.write(" -b " + str(begin_year) + " -e " + str(end_year) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
			f.write(f"julia -p {p} fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
			f.write(f"julia -p {p} est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
			if delete_raw_option_files:
				f.write("rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
				f.write("rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
			f.write(f"echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + f"' >> tracking_file_{base_name}.txt\n")
			f.write("\n")

	residual_companies += companies_to_estimate_1[(i_group+1)*group_size:]

	# Group 2
	secid_to_estimate = companies_to_estimate_2
	top_comps = len(secid_to_estimate)
	group_size = 90
	groups = int(top_comps/group_size)
	start_index = 1
	name_prefix = base_name + "_part_2"

	with open("shell_scripts/sh_" + base_name + "_2.sh", "w") as f: 
		f.write("python load_zcb.py\n")
		for i_group in range(groups):
			f.write("python -W ignore loading_data.py -s ")
			f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
			f.write(" -b " + str(begin_year) + " -e " + str(end_year) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
			f.write(f"julia -p {p} fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
			f.write(f"julia -p {p} est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
			if delete_raw_option_files:
				f.write("rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
				f.write("rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
			f.write(f"echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + f"' >> tracking_file_{base_name}.txt\n")
			f.write("\n")
		
	residual_companies += companies_to_estimate_2[(i_group+1)*group_size:]

	# Group 3
	secid_to_estimate = companies_to_estimate_3
	top_comps = len(secid_to_estimate)
	group_size = 90
	groups = int(top_comps/group_size)
	start_index = 1
	name_prefix = base_name + "_part_3"

	with open("shell_scripts/sh_" + base_name + "_3.sh", "w") as f: 
		f.write("python load_zcb.py\n")
		for i_group in range(groups):
			f.write("python -W ignore loading_data.py -s ")
			f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
			f.write(" -b " + str(begin_year) + " -e " + str(end_year) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
			f.write(f"julia -p {p} fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
			f.write(f"julia -p {p} est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
			if delete_raw_option_files:
				f.write("rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
				f.write("rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
			f.write(f"echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + f"' >> tracking_file_{base_name}.txt\n")
			f.write("\n")
		
		residual_companies += companies_to_estimate_3[(i_group+1)*group_size:]

		start_index = 1
		name_prefix = base_name + "_residual"

		f.write("python -W ignore loading_data.py -s ")
		f.write(",".join([str(x) for x in residual_companies]) + " ")
		f.write(" -b " + str(begin_year) + " -e " + str(end_year) + " -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write(f"julia -p {p} fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		f.write(f"julia -p {p} est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
		if delete_raw_option_files:
			f.write("rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
			f.write("rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
		f.write(f"echo 'Done with residual group' >> tracking_file_{base_name}.txt\n")
		f.write("\n")

if __name__ == "__main__":
	sys.exit(main(sys.argv))




