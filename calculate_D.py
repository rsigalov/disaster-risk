import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from sklearn.decomposition import PCA
import os
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')


# Loading all data:
file_list_to_load  = [x for x in os.listdir("output") if "var_ests_equity_" in x]

measure_list = ["","_in_sample", "_5_5", "_otm_in_sample",
                "_otm_5_5", "_otm1_in_sample"]

for i_file, filename in enumerate(file_list_to_load):
    if i_file % 10 == 0:
        print("%s out of %s" % (str(i_file), str(len(file_list_to_load))))
    
    if i_file == 0:
        df = pd.read_csv("output/" + filename)
        df = df[["secid", "date", "T", "rn_prob_40ann", "rn_prob_2sigma"] + 
                ["IV" + x for x in measure_list] +
                ["V" + x for x in measure_list]]
        
        for measure in measure_list:
            df["D" + measure] = df["V" + measure] - df["IV" + measure]
        
        df = df.drop(["V" + x for x in measure_list], axis = 1)
        df = df.drop(["IV" + x for x in measure_list], axis = 1)
    else:
        df_to_append = pd.read_csv("output/" + filename)
        df_to_append = df_to_append[["secid", "date", "T", "rn_prob_40ann", "rn_prob_2sigma"] + 
                ["IV" + x for x in measure_list] +
                ["V" + x for x in measure_list]]
        
        for measure in measure_list:
            df_to_append["D" + measure] = df_to_append["V" + measure] - df_to_append["IV" + measure]
            
        df_to_append = df_to_append.drop(["V" + x for x in measure_list], axis = 1)
        df_to_append = df_to_append.drop(["IV" + x for x in measure_list], axis = 1)
        df = df.append(df_to_append)

        
        
# Calculating interpolated D:
T_30_days = 30/365

for i_measure in range(len(measure_list)):
    print(measure_list[i_measure])
    df_short = df[["secid", "date", "T", "D" + measure_list[i_measure]]]
    
    secid_list = []
    date_list = []
    D_list = []
    
    for (name, sub_df) in df_short.groupby(["secid", "date"]):           
        secid_list.append(name[0])
        date_list.append(name[1])
        
        # Removing NaNs from interpolation function:
        x = sub_df["T"]
        y = sub_df["D" + measure_list[i_measure]]
        stacked = np.vstack((x.T, y.T)).T

        stacked_filter = stacked[~np.isnan(stacked).any(axis = 1)]
        
        x_new = stacked_filter[:,0]
        y_new = stacked_filter[:,1]
        
        if x_new.shape[0] > 0:
            D_list.append(np.interp(T_30_days, x_new, y_new))
        else:
            D_list.append(np.nan)
    
    D_df_to_merge = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
    D_df_to_merge = D_df_to_merge.sort_values(["secid", "date"])
    
    # Getting the first date of the month average the series:
    D_df_to_merge = D_df_to_merge.replace([np.inf, -np.inf], np.nan)
    D_df_to_merge.columns = ["secid", "date", "D" + measure_list[i_measure]]
    
    if i_measure == 0:
        D_df = D_df_to_merge
    else:
        D_df = pd.merge(D_df, D_df_to_merge, on = ["secid", "date"])
        
# Saving data:
#D_df.to_csv("estimated_d_part_1.csv")        

# Filtering data. Require a secid to have at least 15 observations in a month to
# be averaged to a monthly level. Require an secid to be present in 80% of the
# sample (80% of 264 months) to be included in PCA analysis
# Interpolating to get a 30 day D measure:
D_df["date"] = pd.to_datetime(D_df["date"])


# Averaging within a month for each secid:
D_df["date_trunc"] = D_df["date"] - pd.offsets.MonthBegin(1)

def min_month_obs(x):
    return x["D_5_5"].count() > 15

D_filter = D_df.groupby(["secid", "date_trunc"]).filter(min_month_obs)

D_mon_mean = D_filter.groupby(["secid", "date_trunc"])["D_5_5"].mean().reset_index()

## Plotting for some companies
#secid_list = np.unique(D_df["secid"])
#for i in range(50):
#    D_sub = D_mon_mean[D_mon_mean.secid == secid_list[i]]
#    plt.plot(D_sub.date_trunc, D_sub.D, label = str(secid_list[i]))

num_months = len(np.unique(D_mon_mean["date_trunc"]))
def min_sample_obs(x):
    return x["D_5_5"].count() > num_months*0.8

D_mon_mean_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)
len(np.unique(D_mon_mean_filter["secid"]))


df_pivot = pd.pivot_table(D_mon_mean_filter, values = "D_5_5", 
                          index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
df_pivot = df_pivot.sort_values("date_trunc")
df_pivot.index = pd.to_datetime(df_pivot.index)

# Replacing missing values with average across non-missing values:
for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = np.nanmean(df_pivot.iloc[:,i])

# Replacing negative values with zeros:
for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(df_pivot.iloc[:,i]) < 0, i] = 0
    
# Removing truncating extremely large values:
for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(df_pivot.iloc[:,i]) > 10, i] = 10

X_D = np.array(df_pivot)   

w = eigs(np.corrcoef(X_D.T),1)[1][:,0].astype(float).flatten()
w = w/np.sum(w)
Dw = (X_D @ w.reshape(-1,1)).flatten()
plt.plot(Dw)




################################################################
# Looking at risk-neutral probability of some percent decline
################################################################
# Averaging within a month for each secid:
pdf = df[["secid", "date", "rn_prob_40ann", "rn_prob_2sigma"]]
pdf["date"] = pd.to_datetime(pdf["date"])
pdf["date_trunc"] = pdf["date"] - pd.offsets.MonthBegin(1)

def min_month_obs(x):
    return x["rn_prob_40ann"].count() > 15

pdf_filter = pdf.groupby(["secid", "date_trunc"]).filter(min_month_obs)

pdf_mon_mean = pdf_filter.groupby(["secid", "date_trunc"])["rn_prob_40ann"].mean().reset_index()

# Plotting for some companies
secid_list = np.unique(pdf["secid"])
for i in range(25):
    pdf_sub = pdf_mon_mean[pdf_mon_mean.secid == secid_list[i]]
    plt.plot(pdf_sub.date_trunc, pdf_sub["rn_prob_40ann"], label = str(secid_list[i]))

num_months = len(np.unique(pdf_mon_mean["date_trunc"]))
def min_sample_obs(x):
    return x["rn_prob_40ann"].count() > num_months * 0.8

pdf_mon_mean_filter = pdf_mon_mean.groupby("secid").filter(min_sample_obs)
len(np.unique(D_mon_mean_filter["secid"]))


df_pivot = pd.pivot_table(pdf_mon_mean_filter, values = "rn_prob_40ann", 
                          index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
df_pivot = df_pivot.sort_values("date_trunc")
df_pivot.index = pd.to_datetime(df_pivot.index)

# Replacing missing values with average across non-missing values:
for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = np.nanmean(df_pivot.iloc[:,i])

# Replacing negative values with zeros:
#for i in range(len(df_pivot.columns)):
#    df_pivot.iloc[np.array(df_pivot.iloc[:,i]) < 0, i] = 0
    
# Removing truncating extremely large values:
#for i in range(len(df_pivot.columns)):
#    df_pivot.iloc[np.array(df_pivot.iloc[:,i]) > 10, i] = 10

X_D = np.array(df_pivot)   

w = eigs(np.corrcoef(X_D.T),1)[1][:,0].astype(float).flatten()
w = w/np.sum(w)
Dw_p = (X_D @ w.reshape(-1,1)).flatten()
plt.plot(Dw)

plt.plot(Dw*5)
plt.plot(Dw_p)



























