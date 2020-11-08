"""
File to generate scripts for execution on AWS. First, this program downloads the
list of options for each year with number of strikes and total volume. Then it ranks
them. Then it create a shell file with commands to download option data, fit smiles
and estimate parameters. Finally, it cleans the directory and deletes raw option
data so it doesn't take space
"""

import numpy as np
import pandas as pd
import wrds
import os

db = wrds.Connection()
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

# Specifying the years of the ranking
start_year = 1996
end_year = 2017

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
df_rank.to_csv("data/secid_option_ranking.csv", index = False)