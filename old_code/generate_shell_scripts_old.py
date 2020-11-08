"""
 it create a shell file with commands to download option data, fit smiles
and estimate parameters. Finally, it cleans the directory and deletes raw option
data so it doesn't take space
"""

import numpy as np
import pandas as pd
import random
import wrds
import os

db = wrds.Connection()
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

################################################################
# Loading data on option ranking:
################################################################

rank = pd.read_csv("data/secid_option_ranking.csv")

################################################################
# Specifying company order in which to estimate parameters
################################################################

companies_to_estimate = list(rank.secid)

# Shuffle companies to balance load since some companies have
# more options and years available and some companies have a few
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
name_prefix = "final_part1"

f = open("shell_scripts/option_script_1", "w")

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
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()

secid_to_estimate = companies_to_estimate_2
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part2"

f = open("shell_scripts/option_script_2", "w")

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
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()


secid_to_estimate = companies_to_estimate_3
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part3"

f = open("shell_scripts/option_script_3", "w")

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
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()


################################################################
# Generaing test script that estimates parameters for APPLE
# and MSFT for 2014 and 2015
################################################################
f = open("shell_scripts/test_script", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

f.write("sudo python loading_data.py -s ")
f.write("107525")
f.write(" -b 2015 -e 2015 -o test_run \n")
f.write("sudo julia -p 3 fit_smiles.jl test_run \n")
f.write("sudo julia -p 3 est_parameters.jl test_run \n")
f.write("sudo rm data/raw_data/opt_data_test_run.csv\n")
f.write("sudo rm data/raw_data/dist_data_test_run.csv\n")
f.write("\n")
    
f.close()


################################################################
# Generating scripts for estimation of risk neutral probabilities
# in a different way (without annualization or anything)
################################################################

secid_to_estimate = companies_to_estimate_1
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part1"

f = open("shell_scripts/option_script_rn_prob_1", "w")

for i_group in range(groups):
    f.write("sudo julia -p 7 est_parameters_rn_prob.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()

secid_to_estimate = companies_to_estimate_1
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part2"

f = open("shell_scripts/option_script_rn_prob_2", "w")

for i_group in range(groups):
    f.write("sudo julia -p 7 est_parameters_rn_prob.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()

secid_to_estimate = companies_to_estimate_1
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part3"

f = open("shell_scripts/option_script_rn_prob_3", "w")

for i_group in range(groups):
    f.write("sudo julia -p 7 est_parameters_rn_prob.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()




################################################################
# Generating scripts for downloading data and calculating
# annualized sigma near the money (in the right way)
################################################################

secid_to_estimate = companies_to_estimate_1
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part1"

f = open("shell_scripts/option_script_sigma_NTM_1", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia calc_sigma_MTN.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()

secid_to_estimate = companies_to_estimate_2
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part2"

f = open("shell_scripts/option_script_sigma_NTM_2", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia calc_sigma_MTN.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()


secid_to_estimate = companies_to_estimate_3
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1
name_prefix = "final_part3"

f = open("shell_scripts/option_script_sigma_NTM_3", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo julia calc_sigma_MTN.jl " + name_prefix +"_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_" + name_prefix +"_" + str(i_group + start_index) + ".csv\n")
    f.write("echo 'Done with group " + str(i_group + 1) + "/" + str(groups) + "' >> tracking_file.txt\n")
    f.write("\n")
    
f.close()






