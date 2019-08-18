from __future__ import print_function
from __future__ import division
import sys
import pandas as pd
import numpy as np
import os
import time
from functools import reduce


def interpolate_subdf(subdf, variable, days):
    y = np.array(subdf[variable])
    not_missing = ~np.isnan(y)

    if np.sum(not_missing) == 0:
        return np.nan
    else:
        x = np.array(subdf["T"])[not_missing]
        y = y[not_missing]
        return np.interp(days/365, x, y, left = np.nan, right = np.nan)

def main(argv = None):

    days = float(argv[1])

    # Loading data on standard variables:
    os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision")
    file_list = os.listdir("estimated_data/V_IV/")
    # file_list = [x for x in file_list if "var_ests_missing" in x]
    file_list = [x for x in file_list if "var_ests_index" in x]

    for i, file_name in enumerate(file_list):
        print(i)
        if i == 0:
            df = pd.read_csv("estimated_data/V_IV/" + file_name)
        else:
            df = df.append(pd.read_csv("estimated_data/V_IV/" + file_name))

    df["date"] = pd.to_datetime(df["date"])
    df["D_in_sample"] = df["V_in_sample"] - df["IV_in_sample"]
    df["D_clamp"] = df["V_clamp"] - df["IV_clamp"]

    start = time.time()
    vars_to_interpolate = ["D_clamp", "D_in_sample", "rn_prob_20", "rn_prob_40",
                           "rn_prob_60", "rn_prob_80"]

    df_list = []

    for variable in vars_to_interpolate:
        print("Started interpolating %s, %.4fs elapsed"%(variable,time.time() - start))
        df_list.append(df.groupby(["secid", "date"]).apply(lambda x: interpolate_subdf(x, variable, days)).rename(variable).reset_index())

    print("Total %.4f elapsed" %(time.time() - start))

    df_to_save = reduce(lambda df1, df2: pd.merge(df1, df2, on = ["secid", "date"]), df_list)
    # path_to_save = "estimated_data/interpolated_D/int_ind_disaster_missing_days_" + str(int(days)) + ".csv"
    path_to_save = "estimated_data/interpolated_D/int_ind_disaster_index_days_" + str(int(days)) + ".csv"
    df_to_save.to_csv(path_to_save, index = False)

if __name__ == "__main__": sys.exit(main(sys.argv))

#start = time.time()
#int_d = df.groupby(["secid", "date"]).apply(lambda x: interpolate_subdf(x, "D_clamp", 30)).rename("D_clamp").reset_index()
#print("%.4f elapsed" %(time.time() - start))
