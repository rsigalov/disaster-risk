"""
This scripts calculates the minimum and maximum maturity for each day amongs smiles
where after filtering there are at least X options. This file takes the same SQL
query that is used for filtering and downloading raw option data, modifies it and
uses it to calculate the minimum and maximum maturities
"""

from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import wrds


def main(argv=None):

    db = wrds.Connection()
    min_options = 5 # Require 5 options to be present in a smile
    min_smiles_in_month = 5

    df_store = pd.DataFrame(columns = ["secid", "mon","year", "cnt"])

    for year in range(2014,2017+1,1):
        print("Year %d" % year)
        # Reading SQL script and plugging in characteristics:
        f = open("check_num_smiles.sql", "r") 
        query = f.read()
        query = query.replace("_data_base_", "OPTIONM.OPPRCD" + str(year))
        query = query.replace("_start_date_", "'" + str(year) + "-01-01'")
        query = query.replace("_end_date_", "'" + str(year) + "-12-31'")
        query = query.replace("_min_options_", str(min_options))
        query = query.replace("_min_smiles_per_month_", str(min_smiles_in_month))
        query = query.replace('\n', ' ').replace('\t', ' ')
        # Running query on WRDS
        df_store = df_store.append(db.raw_sql(query))

    df_store.to_csv("tracking_number_of_secids_4.csv", index = False)


if __name__ == "__main__": sys.exit(main(sys.argv))

# ), cnt_smiles as (
#     select 
#         secid, mon, count(date) as cnt
#     from min_max_mat
#     where min_maturity <= 30
#     and max_maturity >= 30
#     group by secid, mon
# )
# select 
#     mon, count(secid) num_firms
# from cnt_smiles
# where cnt >= _min_smiles_per_month_
# group by mon
# order by mon


        # # Dealing with dates
        # min_max_maturity_df["date"] = pd.to_datetime(min_max_maturity_df["date"] )
        # min_max_maturity_df["date_mon"] = min_max_maturity_df["date"]  + pd.offsets.MonthEnd(0)
        # # Calculating the number of firms in each month that have at
        # # least <min_obs_per_mon> eligible smiles that month that
        # # straddle 30 days
        # min_max_maturity_df.groupby(["secid", "date_mon"])





# secid = 101293
# min_options = 5
# min_max_maturity_df = pd.DataFrame(columns = ["secid", "date", "min_maturity", "max_maturity"])

# for year in range(1996,2017+1,1):
#     print("Year %d" % year)
#     f = open("check_num_smiles.sql", "r")
#     query = f.read()
#     query = query.replace("_data_base_", "OPTIONM.OPPRCD" + str(year)).replace("_secid_", str(secid))
#     query = query.replace("_start_date_", "'" + str(year) + "-01-01'").replace("_secid_", str(secid))
#     query = query.replace("_end_date_", "'" + str(year) + "-12-31'")
#     query = query.replace("_min_options_", str(min_options))
#     query = query.replace('\n', ' ').replace('\t', ' ')
#     min_max_maturity_df = min_max_maturity_df.append(db.raw_sql(query))

# min_max_maturity_df.set_index("date")["min_maturity"].plot()
# plt.axhline(30, color = "black", alpha = 0.8)

# min_max_maturity_df[min_max_maturity_df.min_maturity <= 30].shape

# # Comparing this with number of days for which have interpolated series:
# int_d = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_days_30.csv")
# int_d_sub = int_d[int_d.secid == secid]
# int_d_sub[~int_d_sub.rn_prob_80.isnull()].shape

# # Comparing this number with just filtered series from .jl script:
# raw_opt = pd.read_csv("data/raw_data/opt_data_test_num_smiles.csv")
# raw_opt = raw_opt.drop_duplicates()
# raw_opt["date"] = pd.to_datetime(raw_opt["date"])
# raw_opt["exdate"] = pd.to_datetime(raw_opt["exdate"])
# num_opts = raw_opt.groupby(["secid", "date", "exdate"])["cp_flag"].count().reset_index().rename({"cp_flag":"num"}, axis = 1)
# num_opts = num_opts[num_opts.num >= 5]
# num_opts["dtm"] = [x.days - 1 for x in num_opts["exdate"] - num_opts["date"]]
# min_mat = num_opts.groupby(["secid","date"])["dtm"].min().reset_index()
# min_mat[(min_mat.secid == secid) & (min_mat.dtm <= 30)].shape

# # Looking at raw_svi data with additional filters:
# raw_svi = pd.read_csv("data/raw_data/svi_params_test_num_smiles.csv")
# raw_svi["dtm"] = raw_svi["T"] * 365
# min_mat = raw_svi.groupby(["secid", "obs_date"])["dtm"].min().reset_index()
# min_mat[(min_mat.secid == secid) & (min_mat.dtm <= 30)].shape

# # Where is the discrepancy coming from
# min_mat["date"] = pd.to_datetime(min_mat["date"])
# min_max_maturity_df["date"] = pd.to_datetime(min_max_maturity_df["date"])
# comp_df = pd.merge(min_mat[(min_mat.secid == secid) & (min_mat.dtm <= 30)],
#          min_max_maturity_df[min_max_maturity_df.min_maturity <= 30],
#          on = ["date"], how = "outer")
# comp_df[comp_df.min_maturity.isnull()]
























