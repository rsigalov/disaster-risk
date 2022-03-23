'''
Need to add documentation
'''

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
    '''helper function to filter file names with a list'''
    for appendix in appendix_list:
        if appendix in x:
            return True
    
    return False


def main(argv = None):

    days = int(argv[1])
    is_index = int(argv[2])   # Indices have different variables to interpolate 
    append_to_save = argv[3]  # Append to the output file
    appendix_list = argv[4:]  # Potentially combine data from several appendices
    print("Parameters for interpolation:")
    print("  is_index      ", is_index)
    print("  append_to_save", append_to_save)
    print("  appendix_list ", ", ".join(appendix_list))

    # List of all estimate files:
    # appendix_list = ["var_ests_final_part", "var_ests_march_2021_update_part", "var_ests_march_2022_update_part", "var_ests_missing_part", "var_ests_new_release_part"]

    start = time.time()

    # Use files with the following suffix:
    # base_name = "new_release"
    # base_name = "missing"
    # base_name = "index"

    # Loading data on standard variables:
    # os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision")
    file_list = os.listdir("data/output/")

    # Filtering files to specified appendices:
    file_list = [x for x in file_list if doesContainAppendixFromList(x, appendix_list)]
    file_list = sorted(file_list)
    # file_list = file_list[0:5] # for testing

    # print(f"\n---- Loading Data, {(time.time() - start):.4f}s elapsed ----\n")
    # df = pd.concat([pd.read_csv("data/output/" + file_name) for file_name in file_list], ignore_index=True)
    # print(f"Observations BEFORE removing duplicates: {df.shape[0]}")

    # print(f"\n---- Removing duplicates, {(time.time() - start):.4f}s elapsed ----\n")
    # # At initial stages of the project we loaded and estimated data for some
    # # companies multiple times. Hence, we need to leave only the
    # # unique observations.
    # df = df.drop_duplicates()
    # print(f"Observations AFTER removing duplicates: {df.shape[0]}")

    # df["date"] = pd.to_datetime(df["date"])
    # df["D_in_sample"] = df["V_in_sample"] - df["IV_in_sample"]
    # df["D_clamp"] = df["V_clamp"] - df["IV_clamp"]

    # Variable to loop over
    if is_index == 1:
        vars_to_interpolate = ["D_clamp", "D_in_sample", "rn_prob_5", "rn_prob_20", "V_clamp", "IV_clamp"]
    elif is_index == 0:
        vars_to_interpolate = ["D_clamp", "D_in_sample", "rn_prob_20", "rn_prob_80", "V_clamp", "IV_clamp"]
    else:
        raise ValueError
        
    df_list = []

    print(f"\n---- Interpolating Disaster Measures, {(time.time() - start):.4f}s elapsed ----\n")
    i = 1
    total_size = len(file_list)
    for file in file_list:
        print(f"File {i} out of {total_size}: {file}")
        df = pd.read_csv("data/output/" + file)

        # Older files may have different column names:
        df = df.rename(columns={'rn_prob_20ann': 'rn_prob_20','rn_prob_40ann': 'rn_prob_40','rn_prob_60ann': 'rn_prob_60','rn_prob_80ann': 'rn_prob_80'})

        df["date"] = pd.to_datetime(df["date"])
        df["D_in_sample"] = df["V_in_sample"] - df["IV_in_sample"]
        df["D_clamp"] = df["V_clamp"] - df["IV_clamp"]
        for variable in vars_to_interpolate:
            df_to_append = df.groupby(["secid", "date"]).apply(lambda x: interpolate_subdf(x, variable, days)).rename("value").reset_index()
            df_to_append["days"] = days
            df_to_append["variable"] = variable
            df_list.append(df_to_append)
    
        i += 1

    print("\n----Total %.4f elapsed ----\n" % (time.time() - start))
    print("\n---- Saving Reslts ----\n")
    df_to_save = pd.concat(df_list, ignore_index=True)

    print("\n---- Removing duplicates ----\n")
    df_to_save = df_to_save.drop_duplicates()
    
    path_to_save = f"data/interpolated_D/interpolated_disaster_{append_to_save}_{days}.csv"
    
    df_to_save.to_csv(path_to_save, index = False)

if __name__ == "__main__": 
    sys.exit(main(sys.argv))

