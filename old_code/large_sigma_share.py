"""
This script calculates the share of observations that have large NTM sigma:
    observations where 5sigma > 1
    observations where 2sigma > 1
    observations where 1sigma > 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from sklearn.decomposition import PCA
import os
import wrds
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')


# Getting the names of all files that start from svi_params_*
# and loading the data
path = "data/"
files = []
for i in os.listdir(path):
    if 'svi_params' in i:
        files.append(i)
        
for (i_filename, filename) in enumerate(files[0:200]):
    print(i_filename)
    if i_filename == 0:
        svi = pd.read_csv(path + filename)
    else:
        svi = svi.append(pd.read_csv(path + filename))

for (i_filename, filename) in enumerate(files[200:400]):
    print(i_filename)
    if i_filename == 0:
        svi2 = pd.read_csv(path + filename)
    else:
        svi2 = svi2.append(pd.read_csv(path + filename))

for (i_filename, filename) in enumerate(files[400:len(files)]):
    print(i_filename)
    if i_filename == 0:
        svi3 = pd.read_csv(path + filename)
    else:
        svi3 = svi3.append(pd.read_csv(path + filename))

svi = svi.append(svi2)
svi = svi.append(svi3)


# Subsetting the data needed:
svi = svi[["secid", "obs_date", "spot", "F", "T", "sigma_NTM", "min_K", "max_K"]]

# grouping by date and calculating the share of observations with large sigmas:
svi["obs_date"] = pd.to_datetime(svi["obs_date"])
daily_obs = svi[svi["T"] <= 0.5].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs"}, axis = 1)
sigma_5_big_1_obs_date = svi[(5*svi["sigma_NTM"] >= 1) & (svi["T"] <= 0.5)].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_5"}, axis = 1)
sigma_2_big_1_obs_date = svi[(2*svi["sigma_NTM"] >= 1) & (svi["T"] <= 0.5)].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_2"}, axis = 1)
sigma_1_big_1_obs_date = svi[(1*svi["sigma_NTM"] >= 1) & (svi["T"] <= 0.5)].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_1"}, axis = 1)

daily_obs = pd.merge(daily_obs, sigma_5_big_1_obs_date)
daily_obs = pd.merge(daily_obs, sigma_2_big_1_obs_date)
daily_obs = pd.merge(daily_obs, sigma_1_big_1_obs_date)

# Calculating shares:
daily_obs["share_5"] = daily_obs["obs_5"]/daily_obs["obs"]
daily_obs["share_2"] = daily_obs["obs_2"]/daily_obs["obs"]
daily_obs["share_1"] = daily_obs["obs_1"]/daily_obs["obs"]

# Plotting the shares:
daily_obs.set_index("obs_date")[["share_5", "share_2", "share_1"]].plot(figsize = (8,6), title = "Share of observations with x*sigma >= 1")

# Trying to identify only the largest stocks (say 500):
db = wrds.Connection()

query = """
select 
    secid, 
    extract(year from date) as year, 
    sum(volume) as volume
from OPTIONM.OPVOLD
where cp_flag in ('C', 'P')
and date >= '1996-01-01' and date <= '2017-12-31'
group by extract(year from date), secid
"""

query = query.replace('\n', ' ').replace('\t', ' ')
df_volume = db.raw_sql(query)

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

# Calculating sum of volume traded for each year:
df_volume_year = df_volume.groupby("year")["volume"].sum().reset_index()

# Merging on volume data frame and calculating the share of each secid in year
# volume traded:
df_volume = pd.merge(df_volume, df_volume_year, on  = "year", how = "left")
df_volume["share"] = df_volume["volume_x"]/df_volume["volume_y"]

# Summing shares of volume traded and ordering
df_sum_shares = df_volume.groupby("secid")["share"].sum().reset_index().sort_values("share", ascending = False)

# Take first n companies and plot the figure from before
n = 100
secid_list = df_sum_shares["secid"][0:n]
svi_sub = svi[svi["secid"].isin(secid_list)]

daily_obs = svi_sub[svi_sub["T"] <= 0.5].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs"}, axis = 1)
sigma_5_big_1_obs_date = svi_sub[(5*svi_sub["sigma_NTM"] >= 1) & (svi_sub["T"] <= 0.5)].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_5"}, axis = 1)
sigma_2_big_1_obs_date = svi_sub[(2*svi_sub["sigma_NTM"] >= 1) & (svi_sub["T"] <= 0.5)].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_2"}, axis = 1)
sigma_1_big_1_obs_date = svi_sub[(1*svi_sub["sigma_NTM"] >= 1) & (svi_sub["T"] <= 0.5)].groupby("obs_date")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_1"}, axis = 1)

daily_obs = pd.merge(daily_obs, sigma_5_big_1_obs_date, how = "outer")
daily_obs = pd.merge(daily_obs, sigma_2_big_1_obs_date, how = "outer")
daily_obs = pd.merge(daily_obs, sigma_1_big_1_obs_date, how = "outer")

# Calculating shares:
daily_obs["share_5"] = daily_obs["obs_5"]/daily_obs["obs"]
daily_obs["share_2"] = daily_obs["obs_2"]/daily_obs["obs"]
daily_obs["share_1"] = daily_obs["obs_1"]/daily_obs["obs"]

# Plotting the shares:
daily_obs.set_index("obs_date")[["share_5", "share_2", "share_1"]].plot(figsize = (8,6), title = "Share of observations with x*sigma >= 1, top-100 secid by volume")



###################################################################################
# Looking at S&P 500 data
###################################################################################

# Loading data on index (estimate on EC2):
svi_spx = pd.read_csv("data/svi_params_spx.csv")

svi_spx = svi_spx[["secid", "obs_date", "spot", "F", "T", "sigma_NTM", "min_K", "max_K"]]

svi_spx["date_trunc"] = pd.to_datetime(svi_spx["obs_date"]) - pd.offsets.MonthBegin(1)
daily_obs = svi_spx[svi_spx["T"] <= 0.5].groupby("date_trunc")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs"}, axis = 1)
sigma_5_big_1_obs_date = svi_spx[(5*svi_spx["sigma_NTM"] >= 1) & (svi_spx["T"] <= 0.5)].groupby("date_trunc")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_5"}, axis = 1)
sigma_2_big_1_obs_date = svi_spx[(2*svi_spx["sigma_NTM"] >= 1) & (svi_spx["T"] <= 0.5)].groupby("date_trunc")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_2"}, axis = 1)
sigma_1_big_1_obs_date = svi_spx[(1*svi_spx["sigma_NTM"] >= 1) & (svi_spx["T"] <= 0.5)].groupby("date_trunc")["sigma_NTM"].count().reset_index().rename({"sigma_NTM": "obs_1"}, axis = 1)

daily_obs = pd.merge(daily_obs, sigma_5_big_1_obs_date, how = "outer")
daily_obs = pd.merge(daily_obs, sigma_2_big_1_obs_date, how = "outer")
daily_obs = pd.merge(daily_obs, sigma_1_big_1_obs_date, how = "outer")
daily_obs.loc[daily_obs["obs_5"].isnull(), "obs_5"] = 0
daily_obs.loc[daily_obs["obs_2"].isnull(), "obs_2"] = 0
daily_obs.loc[daily_obs["obs_1"].isnull(), "obs_1"] = 0

# Calculating shares:
daily_obs["share_5"] = daily_obs["obs_5"]/daily_obs["obs"]
daily_obs["share_2"] = daily_obs["obs_2"]/daily_obs["obs"]
daily_obs["share_1"] = daily_obs["obs_1"]/daily_obs["obs"]

daily_obs.set_index("date_trunc")[["share_5", "share_2", "share_1"]].plot(figsize = (8,6), title = "Share of observations with x*sigma >= 1, SPX")

# Looking into integration issues:
df_spx = pd.read_csv("output/var_ests_spx.csv")

D_spx = calc_Ds(df_spx, 60/365)

# Calculating share of monthly observations where some of the variables are missing:
D_spx["date_trunc"] = pd.to_datetime(D_spx["date"]) - pd.offsets.MonthBegin(1)

num_unique_dates = len(np.unique(D_spx["date_trunc"]))
date_list = []
not_null_share_matrix = np.zeros((num_unique_dates, 6))
pos_share_matrix = np.zeros((num_unique_dates, 6))
i = 0

for (name, subdf) in D_spx.groupby("date_trunc"):
    date_list.append(name)
    not_null_share_matrix[i,:] = np.array(subdf.notnull().mean()[2:8])
    pos_share_matrix[i,:] = np.array((subdf > 0).mean()[2:8])
    i += 1

# Putting into a data frame:
df_share_not_null = pd.DataFrame(not_null_share_matrix)
df_share_not_null.columns = ["D" + x for x in measure_list]
df_share_not_null["date_trunc"] = date_list
df_share_not_null = df_share_not_null.set_index("date_trunc")
df_share_not_null.plot(figsize = (8,6), title = "Share of non-missing values, SPX")

df_share_pos = pd.DataFrame(pos_share_matrix)
df_share_pos.columns = ["D" + x for x in measure_list]
df_share_pos["date_trunc"] = date_list
df_share_pos = df_share_pos.set_index("date_trunc")
df_share_pos.plot(figsize = (8,6), title = "Share of positive values, SPX")


# Getting some integrals that diverge:
df_spx[df_spx.V.isnull()][["date", "T", "V", "IV"]]


df_spx[df_spx["date"] == "2017-03-29"][["date", "T", "V", "IV"]]
df_spx[df_spx["date"] == "2017-03-30"][["date", "T", "V", "IV"]]
df_spx[df_spx["date"] == "2017-03-31"][["date", "T", "V", "IV"]]




#######################################################





