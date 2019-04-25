"""
Choosing companies to load the data on
"""
import numpy as np
import pandas as pd
import wrds
import os

db = wrds.Connection()
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

start_year = 1996
end_year = 2005
df_rank_1 = pd.read_csv("data/opt_rank_volume_number_" + str(start_year) + "_" + str(end_year) + ".csv")

start_year = 2006
end_year = 2011
df_rank_2 = pd.read_csv("data/opt_rank_volume_number_" + str(start_year) + "_" + str(end_year) + ".csv")

start_year = 2012
end_year = 2017
df_rank_3 = pd.read_csv("data/opt_rank_volume_number_" + str(start_year) + "_" + str(end_year) + ".csv")

#df_rank = df_rank_1.append(df_rank_2)
#df_rank = df_rank.append(df_rank_3)

# Calculating share of overall volume in each of periods:
cnt_1 = np.sum(df_rank_1.cnt)
volume_1 = np.sum(df_rank_1.volume)

cnt_2 = np.sum(df_rank_2.cnt)
volume_2 = np.sum(df_rank_2.volume)

cnt_3 = np.sum(df_rank_3.cnt)
volume_3 = np.sum(df_rank_3.volume)

df_rank_1["cnt_share"] = df_rank_1.cnt/cnt_1
df_rank_1["volume_share"] = df_rank_1.volume/volume_1

df_rank_2["cnt_share"] = df_rank_2.cnt/cnt_2
df_rank_2["volume_share"] = df_rank_2.volume/volume_2

df_rank_3["cnt_share"] = df_rank_3.cnt/cnt_3
df_rank_3["volume_share"] = df_rank_3.volume/volume_3

df_rank = df_rank_1.append(df_rank_2)
df_rank = df_rank.append(df_rank_3)


df_sum = df_rank.groupby("secid")["cnt_share", "volume_share"].sum()
df_rank_new = df_sum[["cnt_share", "volume_share"]].rank(ascending = False)

df_rank_new = df_rank_new.sort_values("volume_share", ascending = True)

list_to_estimate = list(df_rank_new.index[(df_rank_new.cnt_share <= 1000) & (df_rank_new.volume_share <= 1000)])

# Splitting the list into three groups:
secid_list_1 = []
secid_list_2 = []
secid_list_3 = []

for i in range(int(np.floor(len(list_to_estimate)/3))):
    secid_list_1.append(list_to_estimate[3*i])
    secid_list_2.append(list_to_estimate[3*i+1])
    secid_list_3.append(list_to_estimate[3*i+2])


############################################################

secid_to_estimate = secid_list_1
top_comps = len(secid_to_estimate)
group_size = 10
groups = int(top_comps/group_size)
start_index = 1

f = open("short_script_1", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters_short.jl equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data_new/opt_data_equity_short_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data_new/dist_data_equity_short_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()


secid_to_estimate = secid_list_2
top_comps = len(secid_to_estimate)
group_size = 10
groups = int(top_comps/group_size)
start_index = 101

f = open("short_script_2", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters_short.jl equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data_new/opt_data_equity_short_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data_new/dist_data_equity_short_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()


secid_to_estimate = secid_list_3
top_comps = len(secid_to_estimate)
group_size = 10
groups = int(top_comps/group_size)
start_index = 201

f = open("short_script_3", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2017 -o equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters_short.jl equity_short_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data_new/opt_data_equity_short_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data_new/dist_data_equity_short_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()














