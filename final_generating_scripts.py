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

############################################################
# Getting top companies by the number of option entries
# in the specified time period
############################################################
for t in range(start_year, end_year + 1, 1):
    print(t)
    query = """
        select secid, count(strike_price) as cnt 
        from OPTIONM.OPPRCD2008
        group by secid order by cnt desc
    """
    query = query.replace('\n', ' ').replace('\t', ' ')
    query = query.replace('OPTIONM.OPPRCD2008', 'OPTIONM.OPPRCD' + str(t))
    df_comps_to_append = db.raw_sql(query)
    if t == start_year:
        df_comps = df_comps_to_append
    else:
        df_comps = df_comps.append(df_comps_to_append)

df_comps = df_comps.groupby("secid")["cnt"].sum()
df_comps = df_comps.sort_values(ascending = False)
df_comps = df_comps.reset_index()

# 1. Removing the index:
query = """
select secid, issue_type
from OPTIONM.SECURD
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_issue_type = db.raw_sql(query)

# Merging on TOTAL number of options in 2017:
df_comps = pd.merge(df_comps, df_issue_type, on = "secid")
df_comps = df_comps[df_comps["issue_type"] == "0"]
df_comps = df_comps.drop("issue_type", axis = 1)

# Calculating the ranking for secids based on number of options
df_comps = df_comps.sort_values("cnt", ascending = False)
df_comps["number_rank"] = list([x + 1 for x in range(df_comps.shape[0])])

############################################################
# Getting top companies by the volume of option trading
############################################################
query = """
select secid, extract(year from date) as year, sum(volume) as volume, 
sum(open_interest) as op_int
from OPTIONM.OPVOLD
where cp_flag in ('C', 'P')
and date >= _start_date_ and date <= _end_date_
group by extract(year from date), secid
"""
query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace("_start_date_", "'" + str(start_year) + "-01-01'")
query = query.replace("_end_date_", "'" + str(end_year) + "-12-31'")
df_volume = db.raw_sql(query)
df_volume = df_volume.sort_values("volume", ascending = False)

# Getting info on the type of the company:
query = """
select secid, issue_type
from OPTIONM.SECURD
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_issue_type = db.raw_sql(query)

# Merging on TOTAL number of options in 2017:
df_volume = pd.merge(df_volume, df_issue_type, on = "secid")
df_volume = df_volume[df_volume["issue_type"] == "0"]
df_volume = df_volume.drop("issue_type", axis = 1)
df_volume = df_volume.sort_values("volume", ascending = False)

# Calculating the ranking for secids based volume traded
df_volume = df_volume.drop("op_int", axis = 1)
df_volume = df_volume.groupby("secid")["volume"].sum()
df_volume = df_volume.reset_index().sort_values("volume", ascending = False)
df_volume["volume_rank"] = list([x + 1 for x in range(df_volume.shape[0])])

###############################################################
# Merging results: secid, ranking in volume, ranking in number
###############################################################
df_rank = pd.merge(df_comps, df_volume, on = "secid", how = "outer")

# Saving result
df_rank.to_csv("data/opt_rank_volume_number_" + str(start_year) + "_" + str(end_year) + ".csv", index = False)

###############################################################
# Getting all data together:
###############################################################
start_year = 1996
end_year = 2005
df_rank_1 = pd.read_csv("data/opt_rank_volume_number_" + str(start_year) + "_" + str(end_year) + ".csv")

start_year = 2006
end_year = 2011
df_rank_2 = pd.read_csv("data/opt_rank_volume_number_" + str(start_year) + "_" + str(end_year) + ".csv")

start_year = 2012
end_year = 2017
df_rank_3 = pd.read_csv("data/opt_rank_volume_number_" + str(start_year) + "_" + str(end_year) + ".csv")


###############################################################
# Leaving out already estimated ones:
###############################################################
# 1. For period 1996-2005
secid_both_top = df_rank_3[(df_rank_3["number_rank"] <= 2000) & 
                               (df_rank_3["volume_rank"] <= 2000)]["secid"]
print(len(secid_both_top))

to_estimate_1996_2005 = list(df_rank_1[df_rank_1["secid"].isin(secid_both_top)]["secid"])

downloaded_1996_2005_file_list = list(range(81, 100 + 1, 1)) + list(range(101, 169, 1)) + list(range(401, 489, 1)) + list(range(701, 727+1,1))

for i in downloaded_1996_2005_file_list:
    print(i)
    if i == 81:
        df = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    elif i != 99:
        df = df.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

already_estimated = list(set(df["secid"]))

to_estimate_1996_2005 = np.setdiff1d(np.array(secid_both_top), np.array(already_estimated))
print(len(to_estimate_1996_2005))

# 2. For period 2006-2011

to_estimate_2006_2011 = list(df_rank_2[df_rank_2["secid"].isin(secid_both_top)]["secid"])
print(len(to_estimate_2006_2011))

downloaded_2006_2011_file_list = list(range(21, 60 + 1, 1)) + list(range(201, 262 + 1, 1)) + list(range(501, 580 + 1, 1)) + list(range(801, 823+1,1))

for i in downloaded_2006_2011_file_list:
    print(i)
    if i == 21:
        df = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    elif i != 38:
        df = df.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

already_estimated = list(set(df["secid"]))

to_estimate_2006_2011 = np.setdiff1d(np.array(to_estimate_2006_2011), np.array(already_estimated))
print(len(to_estimate_2006_2011))

# 3. For period 2012-2017

to_estimate_2012_2017 = list(df_rank_3[df_rank_3["secid"].isin(secid_both_top)]["secid"])
print(len(to_estimate_2012_2017))

downloaded_2012_2017_file_list = list(range(61, 80 + 1, 1)) + list(range(301, 371 + 1, 1)) + list(range(601, 652 + 1, 1)) + list(range(901, 921+1,1))

for i in downloaded_2012_2017_file_list:
    print(i)
    if i == 61:
        df = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    else:
        df = df.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

already_estimated = list(set(df["secid"]))

to_estimate_2012_2017 = np.setdiff1d(np.array(to_estimate_2012_2017), np.array(already_estimated))
print(len(to_estimate_2012_2017))

###############################################################
# Creating a shell file to estimate:
secid_to_estimate = to_estimate_1996_2005
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1001

f = open("final_script_13", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2005 -o equity_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl equity_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl equity_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()


secid_to_estimate = to_estimate_2006_2011
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1101

f = open("final_script_14", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 2006 -e 2011 -o equity_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl equity_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl equity_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()



secid_to_estimate = to_estimate_2012_2017
top_comps = len(secid_to_estimate)
group_size = 30
groups = int(top_comps/group_size)
start_index = 1201

f = open("final_script_15", "w")

f.write("sudo python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("sudo python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 2012 -e 2017 -o equity_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 fit_smiles.jl equity_" + str(i_group + start_index) + "\n")
    f.write("sudo julia -p 7 est_parameters.jl equity_" + str(i_group + start_index) + "\n")
    f.write("sudo rm data/raw_data/opt_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("sudo rm data/raw_data/dist_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()















