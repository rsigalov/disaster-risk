"""
Similar file for indices. The only difference is the risk-neutral probability
of a decline that now takes values in -5%, -10%, -15%, -20%, as opposed to
-20%, -40%, -60%, -80%.
"""

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


def doesContainAppendixFromList(x, appendix_list):
    '''helper function to filter file names'''
    for appendix in appendix_list:
        if appendix in x:
            return True
    
    return False


def main(argv = None):

    days = float(argv[1])
    is_index = int(argv[2])   # Indicees have different variables to interpolate 
    append_to_save = argv[3]  # Append to the output file
    appendix_list = argv[4:]  # Potentially combine data from several appendices

    # Loading data on standard variables:
    file_list = os.listdir("data/output/")

    # Filtering files to specified appendices:
    file_list = [x for x in file_list if doesContainAppendixFromList(x, appendix_list)]

    print("\n---- Loading Data ----\n")
    df = pd.concat([pd.read_csv("data/output/" + file_name) for file_name in file_list], ignore_index=True)

    print("\n---- Removing duplicates ----\n")
    # At initial stages of the project we loaded and estimated data for some
    # companies multiple times. Hence, we need to leave only the
    # unique observations.
    df = df.drop_duplicates()

    df["date"] = pd.to_datetime(df["date"])
    df["D_in_sample"] = df["V_in_sample"] - df["IV_in_sample"]
    df["D_clamp"] = df["V_clamp"] - df["IV_clamp"]

    start = time.time()
    if is_index == 1:
        vars_to_interpolate = [
            "D_clamp", "D_in_sample", "rn_prob_5", "rn_prob_10","rn_prob_15", "rn_prob_20"]
    elif is_index == 0:
        vars_to_interpolate = [
            "D_clamp", "D_in_sample", "rn_prob_5", "rn_prob_10","rn_prob_15", "rn_prob_20"]
    else:
        raise ValueError

    df_list = []

    print("\n---- Interpolating Disaster Measures ----\n")
    for variable in vars_to_interpolate:
        print("Started interpolating %s, %.4fs elapsed"%(variable,time.time() - start))
        df_list.append(df.groupby(["secid", "date"]).apply(lambda x: interpolate_subdf(x, variable, days)).rename(variable).reset_index())

    print("\n----Total %.4f elapsed ----\n" %(time.time() - start))
    print("\n---- Saving Reslts ----\n")
    df_to_save = reduce(lambda df1, df2: pd.merge(df1, df2, on = ["secid", "date"]), df_list)
    
    path_to_save = f"data/interpolated_D/int_spx_disaster_{append_to_save}_days_{str(int(days))}.csv"
    
    df_to_save.to_csv(path_to_save, index = False)

if __name__ == "__main__": sys.exit(main(sys.argv))