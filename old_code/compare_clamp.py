#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare estimation results for clamped vs. extrapolated SVi curves:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from sklearn.decomposition import PCA
import statsmodels.api as sm
import os
from scipy.sparse.linalg import eigs
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

# List of files in data/output_new/
directory = "data/output_new/"
file_list = os.listdir(directory)
file_list = [x for x in file_list if "var_ests_equity_short" in x]

for i_file, filename in enumerate(file_list):
    print("%d out of %d" % (i_file, len(file_list)))
    if i_file == 0:
        df = pd.read_csv(directory + filename)
    else:
        df = df.append(pd.read_csv(directory + filename))

# Apple from extrapolated SVI volatility smile:
df.date = pd.to_datetime(df.date)


############################################################
# Estimate D's for extrapolated SVI
############################################################
def calc_Ds(df, T):

    measure_list = ["_clamp"]
    
    for i_measure in range(len(measure_list)):
        print(i_measure)
        df_short = df[["secid", "date", "T", "V" + measure_list[i_measure], 
                       "IV" + measure_list[i_measure]]]
        df_short["D"] = df_short["V" + measure_list[i_measure]] - df_short["IV" + measure_list[i_measure]]
        df_short = df_short.drop(["V" + measure_list[i_measure], "IV" + measure_list[i_measure]], axis = 1)
        
        secid_list = []
        date_list = []
        D_list = []
        
        for (name, sub_df) in df_short.groupby(["secid", "date"]):                
            secid_list.append(name[0])
            date_list.append(name[1])
            
            # Removing NaNs for interpolation function:
            x = sub_df["T"]
            y = sub_df["D"]
            stacked = np.vstack((x.T, y.T)).T
    
            stacked_filter = stacked[~np.isnan(stacked).any(axis = 1)]
            
            x_new = stacked_filter[:,0]
            y_new = stacked_filter[:,1]
            
            # Interpolating. Since we removed NaN if can't interpolate the function
            # will simply return NaNs
            if len(x_new) == 0:
                D_list.append(np.nan)
            else:
                D_list.append(np.interp(T, x_new, y_new))
        
        D_df_to_merge = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
        D_df_to_merge = D_df_to_merge.sort_values(["secid", "date"])
        
        # Getting the first date of the month average the series:
        D_df_to_merge = D_df_to_merge.replace([np.inf, -np.inf], np.nan)
        D_df_to_merge.columns = ["secid", "date", "D" + measure_list[i_measure]]
        
        if i_measure == 0:
            D_df = D_df_to_merge
        else:
            D_df = pd.merge(D_df, D_df_to_merge, on = ["secid", "date"])
                
    return D_df
    

# Interpolating to get a 30 day D measure:
D = calc_Ds(df, 30/365)
D["date"] = pd.to_datetime(D["date"])

D.to_csv("clamp_D_30_days.csv", index = False)
D = pd.read_csv("clamp_D_30_days.csv")

# Averaging within a month for each secid:
D["date_trunc"] = D["date"] - pd.offsets.MonthBegin(1)

def min_month_obs(x):
    return x["D_clamp"].count() > 15

D_filter = D.groupby(["secid", "date_trunc"]).filter(min_month_obs)

D_mon_mean = D_filter.groupby(["secid", "date_trunc"])["D_clamp"].mean().reset_index()

# Plotting for some companies
secid_list = np.unique(D["secid"])
for i in range(30):
    D_sub = D_mon_mean[D_mon_mean.secid == secid_list[i]]
    plt.plot(D_sub.date_trunc, D_sub.D_clamp, label = str(secid_list[i]))

num_months = len(np.unique(D_mon_mean["date_trunc"]))
def min_sample_obs(x):
    return x["D_clamp"].count() > num_months*0.8

D_mon_mean_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)
len(np.unique(D_mon_mean_filter["secid"]))



pca = PCA(n_components = 5)
df_pivot = pd.pivot_table(D_mon_mean_filter, values = "D_clamp", 
                          index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
df_pivot = df_pivot.sort_values("date_trunc")
df_pivot.index = pd.to_datetime(df_pivot.index)

for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = np.nanmean(df_pivot.iloc[:,i])


X_D = np.array(df_pivot)   

# Standardize the data:
for i in range(X_D.shape[1]):
    X_D[:,i] = (X_D[:,i] - np.mean(X_D[:,i]))/np.std(X_D[:,i])

pca.fit(X_D)


pca.explained_variance_ratio_[0]

plt.hist(pca.transform(X_D)[:,0], bins = 15)
plt.hist(pca.transform(X_D)[:,1], bins = 15)

mean_D = D_filter.groupby(["date_trunc"])["D_clamp"].mean()

plt.plot(mean_D.index, pca.components_[0]/np.std(pca.components_[0]), label = "PC1")
plt.plot(mean_D.index, pca.components_[1]/np.std(pca.components_[1]), label = "PC2")
plt.plot(mean_D.index, np.array(mean_D)/np.std(np.array(mean_D)), label = "Mean")
plt.legend()




# Constructing factor:
plt.plot(X_D @ pca.components_[0].reshape(-1,1))



########################################################
# Comparing extracting actual PC1 and index that Emil
# suggests to construct:
df_pivot = pd.pivot_table(D_mon_mean_filter, values = "D_clamp", 
                          index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
df_pivot = df_pivot.sort_values("date_trunc")
df_pivot.index = pd.to_datetime(df_pivot.index)

for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = np.nanmean(df_pivot.iloc[:,i])

X_D = np.array(df_pivot) 

# Standardize the data:
for i in range(X_D.shape[1]):
    X_D[:,i] = (X_D[:,i] - np.mean(X_D[:,i]))/np.std(X_D[:,i])

# Standard PCA approach:
pc1 = np.linalg.eig(X_D @ X_D.T)[1][:,0].astype(float) # Calculating PC1
pc1_loadings = np.linalg.eig(X_D.T @ X_D)[1][:,0].astype(float) # Loadings on PC1

# Using weights on the PC1 to multiply D:
weight_index = (X_D @ pc1_loadings.reshape(-1,1)).flatten()

plt.plot(df_pivot.index, (-1) * pc1, label = "-PC1")
plt.plot(df_pivot.index, weight_index, label = "-D*w")
plt.plot(df_pivot.index, mean_D, label = "-mean_D")
plt.legend()

(pc1, weight_index)


(X_D.T @ pc1.reshape(-1,1)).flatten() - pc1_loadings




pc1 = eigs(X_D @ X_D.T,1)[1][:,0].astype(float).flatten()
pc1_loadings = eigs(X_D.T @ X_D,1)[1][:,0].astype(float).flatten()

plt.plot(pc1)
plt.plot(X_D @ pc1_loadings.reshape(-1,1))


(X_D.T @ pc1) + pc1_loadings

v =  eigs(np.corrcoef(X_D.T),1)[1][:,0].astype(float).flatten()
w = loadings_corr/np.sum(loadings_corr)
Dw = (X_D @ w.reshape(-1,1)).flatten()

# Combining in one data frame:
Emils_D = pd.read_csv("Emils_D.csv")
Emils_D.Date = pd.to_datetime(Emils_D.Date, format = "%m/%d/%y")
Emils_D.D = Emils_D.D/np.std(np.array(Emils_D.D)) 

Clamp_D = pd.DataFrame({"date": mean_D.index,
                        "D*w": Dw/np.std(Dw), _
                        "Mean(D)": np.array(mean_D)/np.std(np.array(mean_D))})
Clamp_D.date = Clamp_D.date + pd.offsets.MonthEnd(1)
    
Emils_D = pd.merge(Emils_D, Clamp_D, left_on = "Date", right_on = "date")
Emils_D = Emils_D.set_index("Date").drop("date", axis = 1)
Emils_D.plot(figsize = (8,6))

Emils_D.corr()








####################################################
# Loading Emil's D:
Emils_D = pd.read_csv("Emils_D.csv")
Emils_D.Date = pd.to_datetime(Emils_D.Date, format = "%m/%d/%y")
Emils_D.D = Emils_D.D/np.std(np.array(Emils_D.D)) 


# Comparing them on the same graph:
plt.plot(mean_D.index, pca.components_[0]/np.std(pca.components_[0]), label = "PC1")
plt.plot(mean_D.index, pca.components_[1]/np.std(pca.components_[1]), label = "PC2")
plt.plot(mean_D.index, np.array(mean_D)/np.std(np.array(mean_D)), label = "Mean")
plt.plot(Emils_D.Date, Emils_D.D/np.std(np.array(Emils_D.D)), label = "Emil's D")
plt.legend()

# Calculating correlations
Clamp_D = pd.DataFrame({"date": mean_D.index,
                        "PC1": pca.components_[0]/np.std(pca.components_[0]), 
                        "PC2": pca.components_[1]/np.std(pca.components_[1]),
                        "mean": np.array(mean_D)/np.std(np.array(mean_D))})
Clamp_D.date = Clamp_D.date + pd.offsets.MonthEnd(1)
    
Emils_D = pd.merge(Emils_D, Clamp_D, left_on = "Date", right_on = "date")

Emils_D = Emils_D.set_index("Date").drop("date", axis = 1)
Emils_D.plot()

Emils_D.corr()

####################################################
# Companies with largest loadings:
secid_list = list(df_pivot.columns)
pc0_loadings = pca.transform(X_D.T)[:,0]
pc1_loadings = pca.transform(X_D.T)[:,1]

df_loadings = pd.DataFrame({"secid": secid_list,
                            "PC0_loading": pc0_loadings,
                            "PC1_loading": pc1_loadings})
    
df_loadings.to_csv("loadings_clamps.csv")

pc0_loadings @ pca.compo

####################################################
# Looking at subsample of data:
D_mon_mean_filter_sub = D_mon_mean_filter[D_mon_mean_filter.date_trunc < "2020-01-01"]
pca = PCA(n_components = 5)
df_pivot = pd.pivot_table(D_mon_mean_filter_sub, values = "D_clamp", 
                          index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
df_pivot = df_pivot.sort_values("date_trunc")
df_pivot.index = pd.to_datetime(df_pivot.index)

for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = np.nanmean(df_pivot.iloc[:,i])
#    df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = 0


X_D = np.array(df_pivot)   
pca.fit(X_D.T)


pca.explained_variance_ratio_[0]

plt.hist(pca.transform(X_D.T)[:,0], bins = 15)
plt.hist(pca.transform(X_D.T)[:,1], bins = 15)

mean_D = D_filter[D_filter.date_trunc < "2020-01-01"].groupby(["date_trunc"])["D_clamp"].mean()

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(mean_D.index, pca.components_[0]/np.std(pca.components_[0]), label = "PC1")
ax.plot(mean_D.index, pca.components_[1]/np.std(pca.components_[1]), label = "PC2")
ax.plot(mean_D.index, np.array(mean_D)/np.std(np.array(mean_D)), label = "Mean")
ax.axhline(0, color = "black")
ax.legend()



















# What are the companies that have positive vs. negative loading on the factor:
secid_pos_PC1 = list(df_pivot.columns[pca.transform(X_D.T)[:,0] > 0.35])

for secid in secid_pos_PC1:
    df_to_plot = D_mon_mean_filter[D_mon_mean_filter["secid"] == secid]
    plt.plot(df_to_plot.date_trunc, df_to_plot.D_clamp)
    
    
for secid in list(df_pivot.columns[pca.transform(X_D.T)[:,0] < 0]):
    df_to_plot = D_mon_mean_filter[D_mon_mean_filter["secid"] == secid]
    plt.plot(df_to_plot.date_trunc, df_to_plot.D_clamp)








list(pca.transform(X_D.T)[:,0][pca.transform(X_D.T)[:,0] > 0])






################################################
# What is the fraction of variance explained by
# the mean.

# 1. Calculate loadings on mean D
mean_D = D.groupby(["date_trunc"])["D_clamp"].mean()
loadings = []

for i in range(len(df_pivot.columns)):
    D_series = df_pivot.iloc[:,i]
    loadings.append(sm.OLS(mean_D, D_series).fit().params.iloc[0])

mean_D = np.array(mean_D).reshape(-1, 1)
loadings = np.array(loadings).reshape(-1, 1)

explained = mean_D @ loadings.T
residual = X_D - explained
variance = np.trace(residual.T @ residual)


i = 1
D_series = df_pivot.iloc[:,i]
model = sm.OLS(mean_D, D_series).fit()


X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
loading = sm.OLS(mean_D, D_series).fit().params[0]





#for days in [30, 60, 90, 120]:
#    D = calc_Ds(df, days/365)
#    D["date"] = pd.to_datetime(D["date"])
#    D["date_trunc"] = D["date"] - pd.offsets.MonthBegin(1)
#    D_mon_mean = D.groupby("date_trunc")["D_clamp"].mean()
#    plt.plot(D_mon_mean, label = str(days))
#plt.legend()



#def plot_missing_positive_shares(df):

measure_list = ["", "_in_sample", "_5_5", "_otm_in_sample", "_otm_5_5", "_otm1_in_sample"]
for (i_measure, measure) in enumerate(measure_list):
    print(i_measure)
    cnt_all = D_ext.groupby("date")["D" + measure].size().reset_index().rename(columns={"D" + measure: "cnt_all"})
    cnt_not_null = D_ext.groupby("date")["D" + measure].count().reset_index().rename(columns={"D" + measure: "cnt_not_null"})
    cnt_pos = D_ext[D_ext["D" + measure] > 0].groupby("date")["D" + measure].count().reset_index().rename(columns={"D" + measure: "cnt_pos"})
    
    cnt_merged = pd.merge(cnt_all, cnt_not_null)
    cnt_merged = pd.merge(cnt_merged, cnt_pos)
    
    cnt_merged["date"] = pd.to_datetime(cnt_merged["date"])
    
    # Calculating shares:
    cnt_merged["s_not_null" + measure] = cnt_merged["cnt_not_null"]/cnt_merged["cnt_all"]
    cnt_merged["s_pos" + measure] = cnt_merged["cnt_pos"]/cnt_merged["cnt_all"]
    
    # Averaging within a month:
    cnt_merged["date_trunc"] = cnt_merged["date"] - pd.offsets.MonthBegin(1)
    cnt_mon = cnt_merged.groupby("date_trunc")[["s_not_null" + measure, "s_pos" + measure]].mean()
    
    if i_measure == 0:
        shares_all = cnt_merged[["date", "cnt_all", "s_not_null" + measure, "s_pos"  + measure]]
    else:
        shares_all = pd.merge(shares_all, cnt_merged[["date", "s_not_null" + measure, "s_pos"  + measure]], on = "date")
        
    if i_measure == 0:
        shares_all_mon = cnt_mon
    else:
        shares_all_mon = pd.merge(shares_all_mon, cnt_mon, on = "date_trunc")
            
    

    
    
shares_all_mon[["s_not_null" + x for x in measure_list]].plot(figsize = (10,8))
shares_all_mon[["s_pos" + x for x in measure_list]].plot(figsize = (10,8))



plot_missing_positive_shares(D_ext)
plot_missing_positive_shares(D_clamp)



D_ext.set_index("date").drop("secid", axis=1).plot(figsize = (8,6), title = "CVS extrapolated 2014")
D_clamp.set_index("date").drop("secid", axis=1).plot(figsize = (8,6), title = "CVS clamped 2014")

plt.plot(D_clamp["date"], D_clamp["D"], label = "D clamped")
plt.plot(D_clamp["date"], D_clamp["D_in_sample"], label = "D-in-sample, extrapolated")
plt.title("AAPL 2017")
plt.legend()








