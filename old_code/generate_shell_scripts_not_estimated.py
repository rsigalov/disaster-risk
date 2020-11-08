#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates shell files to estimate disaster measures for
SECIDs that I didn't originally include: some of them because they 
are not common stock, some for some other reason...
"""

import numpy as np
import pandas as pd
import random

df = pd.read_csv("not_estimated_secid.csv")
df = df[df.index_flag == 0]

################################################################
# Specifying company order in which to estimate parameters
################################################################

companies_to_estimate = np.array(df["secid"].drop_duplicates())
companies_to_estimate = [int(x) for x in companies_to_estimate]

# Shuffle companies to balance load since some companies have
# more options and years available and some companies have a few
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
        
################################################################
# Creating shell file for each of the 
################################################################

secid_to_estimate = companies_to_estimate_1
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "missing_part1"

f = open("shell_scripts/option_script_missing_1", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_missing.txt\n")
    f.write("\n")
    
f.close()

residual_companies = companies_to_estimate_1[(i_group+1)*group_size:]


secid_to_estimate = companies_to_estimate_2
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "missing_part2"

f = open("shell_scripts/option_script_missing_2", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_missing.txt\n")
    f.write("\n")
    
f.close()

residual_companies += companies_to_estimate_2[(i_group+1)*group_size:]


secid_to_estimate = companies_to_estimate_3
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "missing_part3"

f = open("shell_scripts/option_script_missing_3", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix + "_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl " + name_prefix + "_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix + "_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix + "_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_missing.txt\n")
    f.write("\n")
    
f.close()

# Dealing with residual companies:
residual_companies += companies_to_estimate_3[(i_group+1)*group_size:]
start_index = 1
name_prefix = "missing_residual"

f = open("shell_scripts/option_script_missing_residual", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

f.write("sudo python loading_data.py -s ")
f.write(",".join([str(x) for x in residual_companies]) + " ")
f.write(" -b 1996 -e 2017 -o " + name_prefix +"\n")
f.write("sudo julia -p 7 fit_smiles.jl " + name_prefix + "\n")
f.write("sudo julia -p 7 est_parameters.jl " + name_prefix + "\n")
f.write("sudo rm data/raw_data/opt_data_" + name_prefix + ".csv\n")
f.write("sudo rm data/raw_data/dist_data_" + name_prefix + ".csv\n")
f.write("echo 'Done with residual ' >> tracking_file_missing.txt\n")
f.write("\n")
    
f.close()

########################################################################
# Dealing with not estimated indices
########################################################################

df = pd.read_csv("not_estimated_secid.csv")
df = df[df.index_flag == 1]

companies_to_estimate = np.array(df["secid"].drop_duplicates())
companies_to_estimate = [int(x) for x in companies_to_estimate]
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


secid_to_estimate = companies_to_estimate_1
top_comps = len(secid_to_estimate)
group_size = 10
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "index_part1"

f = open("shell_scripts/option_script_index_1", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data_index.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles_index.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_index.txt\n")
    f.write("\n")
    
f.close()

residual_companies = companies_to_estimate_1[(i_group+1)*group_size:]


secid_to_estimate = companies_to_estimate_2
top_comps = len(secid_to_estimate)
group_size = 10
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "index_part2"

f = open("shell_scripts/option_script_index_2", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data_index.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles_index.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_index.txt\n")
    f.write("\n")
    
f.close()

residual_companies += companies_to_estimate_2[(i_group+1)*group_size:]


secid_to_estimate = companies_to_estimate_3
top_comps = len(secid_to_estimate)
group_size = 10
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "index_part3"

f = open("shell_scripts/option_script_index_3", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data_index.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles_index.jl " + name_prefix + "_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl " + name_prefix + "_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix + "_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix + "_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file_index.txt\n")
    f.write("\n")
    
f.close()

# Dealing with residual companies:
residual_companies += companies_to_estimate_3[(i_group+1)*group_size:]
start_index = 1
name_prefix = "index_residual"

f = open("shell_scripts/option_script_index_residual", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

f.write("sudo python loading_data_index.py -s ")
f.write(",".join([str(x) for x in residual_companies]) + " ")
f.write(" -b 1996 -e 2017 -o " + name_prefix +"\n")
f.write("sudo julia -p 7 fit_smiles_index.jl " + name_prefix + "\n")
f.write("sudo julia -p 7 est_parameters.jl " + name_prefix + "\n")
f.write("sudo rm data/raw_data/opt_data_" + name_prefix + ".csv\n")
f.write("sudo rm data/raw_data/dist_data_" + name_prefix + ".csv\n")
f.write("echo 'Done with residual ' >> tracking_file_index.txt\n")
f.write("\n")
    
f.close()





