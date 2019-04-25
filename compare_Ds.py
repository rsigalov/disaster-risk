"""
Comparing measures D constructed from different integation
limit of V and IV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from sklearn.decomposition import PCA
import os
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

############################################################################
# 1. Loading data
############################################################################
for i in range(301, 371 + 1, 1):
    print(i)
    if i == 301:
        df = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    elif i != 38:
        df = df.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))
        
df = df.drop_duplicates()

############################################################################
# 2. Interpolativ D = V-IV to get a 30 day measure of disaster
############################################################################
T_30_days = 30/365
measure_list = ["", "_in_sample", "_5_5", "_otm", "_otm_in_sample",
                "_otm_5_5", "_otm1", "_otm1_in_sample", "_otm1_5_5"]

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
        
        # Removing NaNs from interpolation function:
        x = sub_df["T"]
        y = sub_df["D"]
        stacked = np.vstack((x.T, y.T)).T

        stacked_filter = stacked[~np.isnan(stacked).any(axis = 1)]
        
        x_new = stacked_filter[:,0]
        y_new = stacked_filter[:,1]
        
        D_list.append(np.interp(T_30_days, x_new, y_new))
    
    D_df_to_merge = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
    D_df_to_merge = D_df_to_merge.sort_values(["secid", "date"])
    
    # Getting the first date of the month average the series:
    D_df_to_merge = D_df_to_merge.replace([np.inf, -np.inf], np.nan)
    D_df_to_merge.loc[D_df_to_merge["D"] > 10, "D"] = np.nan # Handler for extermely large values
    D_df_to_merge.loc[D_df_to_merge["D"] < 0, "D"] = 0 # Handler for negative values
    D_df_to_merge.columns = ["secid", "date", "D" + measure_list[i_measure]]
    
    if i_measure == 0:
        D_df = D_df_to_merge
    else:
        D_df = pd.merge(D_df, D_df_to_merge, on = ["secid", "date"])
        
############################################################################
# 3. Calculating average daily correlation matrix between measures. First,
# calculate correlation matrix for each company, then average across companies
############################################################################
secid_list = []
D_measure_cov_list = []
for (name, sub_df) in D_df.groupby(["secid"]):
    secid_list.append(name)
    D_measure_cov_list.append(np.array(sub_df[["D" + x for x in measure_list]].corr()))

av_daily_D_corr = reduce(lambda x,y: x+ y, D_measure_cov_list)/len(D_measure_cov_list)

plot_heat_map_list([av_daily_D_corr], measure_list, measure_list, "")
     
############################################################################
# 4. Averaging measure within each (company, month)
############################################################################
D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_trunc"] = D_df["date"] - pd.offsets.MonthBegin(1)
# Now average each measure within firm, month and calculate correlations:
monthly_D_df = D_df.groupby(["secid", "date_trunc"])[["D" + x for x in measure_list]].mean().reset_index()
        
## Trying to leave only companie that have at least 15 observations in a given month:
#def filter_func(x):
#    return x['D'].count() >= 15
#
#monthly_D_df = D_df.groupby(["secid", "date_trunc"]).filter(filter_func).groupby(["secid", "date_trunc"])[["D" + x for x in measure_list]].mean().reset_index()
#

############################################################################
# 5. Now calculating montly correlation between measures:      
############################################################################
secid_list = []
D_measure_cov_list = []
for (name, sub_df) in monthly_D_df.groupby(["secid"]):
    secid_list.append(name)
    to_append = np.array(sub_df[["D" + x for x in measure_list]].corr())
    to_append[~np.isfinite(to_append)] = 0
    D_measure_cov_list.append(to_append)

av_monthly_D_corr = reduce(lambda x,y: x+ y, D_measure_cov_list)/len(D_measure_cov_list)   
        
plot_heat_map_list([av_monthly_D_corr], measure_list, measure_list, "")
        
############################################################################
# 5.5 Taking only the companies that are present in the whole sample:
############################################################################

############################################################################
# 6. Getting first principal component of monthly series:
############################################################################
pc1_list = []
pc1_exp_var = []

# If want to subsample by dates:
#monthly_D_df_sub = monthly_D_df[monthly_D_df["date_trunc"] >= "2009-01-01"]
monthly_D_df_sub = monthly_D_df

for i_measure in range(len(measure_list)):
    print(i_measure)
    
    pca = PCA(n_components=3)
    df_pivot = pd.pivot_table(monthly_D_df_sub, values = "D" + measure_list[i_measure], 
                              index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
    df_pivot = df_pivot.sort_values("date_trunc")
    df_pivot.index = pd.to_datetime(df_pivot.index)
    
    X_D = np.array(df_pivot)        
    X_D[~np.isfinite(X_D)] = 0
    pca.fit(X_D.T)
    
    pc1_list.append(pca.components_[0])
    pc1_exp_var.append(pca.explained_variance_ratio_[0])
        

plt.plot(pc1_list[0])
plt.plot(pc1_list[1])
plt.plot(pc1_list[2])
plt.plot(pc1_list[3])
plt.plot(pc1_list[4])
plt.plot(pc1_list[5])
plt.plot(pc1_list[6])
plt.plot(pc1_list[7])
plt.plot(pc1_list[8])

# Plotting all PCs together (without Emil's series):
PC1_D = pd.DataFrame({"date": np.unique(monthly_D_df_sub["date_trunc"].sort_values())})

for i_measure in range(len(measure_list)):
    PC1_D["D"+measure_list[i_measure]] = pc1_list[i_measure]
        
PC1_D = PC1_D.set_index("date")
corr_D = np.array(PC1_D.corr())
plot_heat_map_list([corr_D], list(PC1_D.columns), list(PC1_D.columns), "")     
PC1_D.plot(figsize = (8,6))
        
        
# 7. Correlation with Emil's measure
Emils_D = pd.read_csv("output/Emils_D.csv", names = ["date", "Emils_D"])
Emils_D["date"] = pd.to_datetime(Emils_D["date"])
Emils_D["date"] = Emils_D["date"] - pd.offsets.MonthBegin(1)

PC1_D = pd.DataFrame({"date": np.unique(monthly_D_df_sub["date_trunc"].sort_values())})

for i_measure in range(len(measure_list)):
    PC1_D["D"+measure_list[i_measure]] = pc1_list[i_measure]
        
PC1_D = pd.merge(Emils_D[Emils_D["date"] >= min(PC1_D["date"])], PC1_D, on = "date", how = "outer")        
PC1_D = PC1_D.set_index("date")
corr_emil_D = np.array(PC1_D.corr())
        
plot_heat_map_list([corr_emil_D], list(PC1_D.columns), list(PC1_D.columns), "")     

# 8. Comparing Emil's D measure and baseline D measure that I calculate on graph:
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(PC1_D.index, PC1_D["Emils_D"]*9, label = "9 * Emil's D")
ax.plot(PC1_D.index, PC1_D["D"], label = "Roman's D")
ax.legend()
ax.set_title("Comparing Emil's D and D from 0 to infinity")
plt.show()

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(PC1_D.index, PC1_D["Emils_D"]*9, label = "9 * Emil's D")
ax.plot(PC1_D.index, PC1_D["D_in_sample"], label = "Roman's D in sample")
ax.legend()
ax.set_title("Comparing Emil's D and D in sample")
plt.show()

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(PC1_D.index, PC1_D["Emils_D"]*9, label = "9 * Emil's D")
ax.plot(PC1_D.index, PC1_D["D_otm1"], label = "Roman's D in sample")
ax.legend()
ax.set_title("Comparing Emil's D and D [0, spot(1-sigma)]")
plt.show()

# 9. Comparing all PC1's on a single plot:
PC1_D_plot = PC1_D.copy()
PC1_D_plot["Emils_D"] = PC1_D_plot["Emils_D"] * 50
PC1_D_plot.plot(figsize = (8,6))
plt.show()


####################################################################
# Combining all estimated pricnpical components in one series:
####################################################################
# 1. Loading saved files:
D_df_1 = pd.read_csv("output/D_df_1996_2005_30.csv").drop(["Unnamed: 0", "date_trunc"], axis = 1)
D_df_2 = pd.read_csv("output/D_df_2006_2011_30.csv").drop(["Unnamed: 0", "date_trunc"], axis = 1)
D_df_3 = pd.read_csv("output/D_df_2012_2017_30.csv").drop("Unnamed: 0", axis = 1)

D_df_all = D_df_1.append(D_df_2).append(D_df_3)

# 2. Generating monthly averages:
D_df_all["date"] = pd.to_datetime(D_df_all["date"])
D_df_all["date_trunc"] = D_df_all["date"] - pd.offsets.MonthBegin(1)
monthly_D_df = D_df_all.groupby(["secid", "date_trunc"])[["D" + x for x in measure_list]].mean().reset_index()

# ???. What if I just estimate average across firms and months of D:
mean_D = D_df_all.groupby("date_trunc")[["D" + x for x in measure_list]].mean()
mean_D.plot(figsize = (8,6))

####################################################################
# Getting data on probability of rare disaster:
####################################################################
# 1. Loading data:
for i in range(101, 169 + 1, 1):
    print(i)
    if i == 101:
        df1 = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    else:
        df1 = df1.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))
        
df1 = df1.drop_duplicates()

for i in range(201, 262 + 1, 1):
    print(i)
    if i == 201:
        df2 = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    else:
        df2 = df2.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))
        
df2 = df2.drop_duplicates()

for i in range(301, 371 + 1, 1):
    print(i)
    if i == 301:
        df3 = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    else:
        df3 = df3.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))
        
df3 = df3.drop_duplicates()

# 2. Combining all of them together in one dataframe:
df_all = df1.append(df2).append(df3)

# 3. Getting average 
df_prob = df_all[["secid", "date", "rn_prob_2sigma", "rn_prob_40ann"]]
df_prob["date"] = pd.to_datetime(df_prob["date"])
df_prob["date_trunc"] = df_prob["date"] - pd.offsets.MonthBegin(1)

monthly_prob = df_prob.groupby(["date_trunc"])[["rn_prob_2sigma", "rn_prob_40ann"]].mean()

# 4. Getting only firms-months that have >= 15 observations:
def min_month_obs(x):
    return x["D"].count() > 15

D_df_filter = D_df_all.groupby(["secid", "date_trunc"]).filter(min_month_obs) #[["D" + x for x in measure_list]].mean().reset_index()

secid_list = []
num_secid = len(np.unique(D_df_filter["secid"]))
D_measure_cov_list = np.zeros((9,9,num_secid))
i = 0
for (name, sub_df) in D_df_filter.groupby(["secid"]):
    D_measure_cov_list[:,:,i] = np.array(sub_df[["D" + x for x in measure_list]].corr())
    i = i + 1

av_corr = np.nanmean(D_measure_cov_list, axis = 2)
pd.DataFrame(av_corr)

D_df_filter_mo_mean = D_df_filter.groupby(["secid", "date_trunc"])[["D" + x for x in measure_list]].mean().reset_index()

secid_list = []
num_secid = len(np.unique(D_df_filter_mo_mean["secid"]))
D_measure_cov_list = np.zeros((9,9,num_secid))
i = 0
for (name, sub_df) in D_df_filter_mo_mean.groupby(["secid"]):
    D_measure_cov_list[:,:,i] = np.array(sub_df[["D" + x for x in measure_list]].corr())
    i = i + 1

av_corr = np.nanmean(D_measure_cov_list, axis = 2)
pd.DataFrame(av_corr)


# 5. Getting average monthly for each firm and calculating average correlation:
D_df_filter["date"] = pd.to_datetime(D_df_filter["date"])
D_df_filter["date_trunc"] = D_df_filter["date"] - pd.offsets.MonthBegin(1)
av_mon_D = D_df_filter.groupby(["date_trunc"])[["D" + x for x in measure_list]].median()
av_mon_D.plot(figsize = (8,6))

# 7. Getting firms that are present in 80% of all months:
# Getting number of unique months:
num_months = len(np.unique(D_df_filter_mo_mean["date_trunc"]))
def min_sample_obs(x):
    return x["D"].count() > 90 # 140 = 80% * 176months in sample

D_df_mo_filter_filter = D_df_filter_mo_mean.groupby("secid").filter(min_sample_obs)
len(np.unique(D_df_mo_filter_filter["secid"]))

# Doing PC1 and filling missing values with means
pc1_list = []
pc1_exp_var = []

# If want to subsample by dates:
D_df_sub = D_df_mo_filter_filter[D_df_mo_filter_filter["date_trunc"] >= "1996-01-01"]

for i_measure in range(len(measure_list)):
    print(i_measure)
    
    pca = PCA(n_components = 1)
    df_pivot = pd.pivot_table(D_df_sub, values = "D" + measure_list[i_measure], 
                              index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
    df_pivot = df_pivot.sort_values("date_trunc")
    df_pivot.index = pd.to_datetime(df_pivot.index)
    
    for i in range(len(df_pivot.columns)):
        df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = np.nanmean(df_pivot.iloc[:,i])
    
    X_D = np.array(df_pivot)   
    pca.fit(X_D.T)
    
    pc1_list.append(pca.components_[0])
    pc1_exp_var.append(pca.explained_variance_ratio_[0])


PC1_D = pd.DataFrame({"date": np.unique(D_df_sub["date_trunc"].sort_values())})

for i_measure in range(len(measure_list)):
    PC1_D["D"+measure_list[i_measure]] = pc1_list[i_measure]
        
PC1_D = PC1_D.set_index("date")
corr_D = np.array(PC1_D.corr())
plot_heat_map_list([corr_D], list(PC1_D.columns), list(PC1_D.columns), "")     
PC1_D.plot(figsize = (8,6))
        

####################################################
db = wrds.Connection(wrds_username = "rsigalov")

df_prices = pd.DataFrame({"secid": [], "date": [], "exdate": [], "cp_flag": [],
                          "strike_price":[],"impl_volatility":[],"mid_price":[],
                          "under_price":[]})
year_start = 1996
year_end = 2005

secid_list = [8957.0,100906.0,100958.0,100967.0,100973.0,101080.0,101087.0,101163.0,101177.0,101428.0,101488.0,101674.0,101761.0,101798.0,101817.0,101857.0,101977.0,101979.0,102109.0,102205.0,102235.0,102263.0,102271.0,102276.0,102288.0,102294.0,102324.0,102383.0,102392.0,102433.0]
year_list = list(range(year_start, year_end + 1, 1))

num_secid = len(secid_list)
num_years = len(year_list)

i_secid = 0
for secid in secid_list:
    i_secid += 1
    i_year = 0
    secid = str(secid)
    
    for year in year_list:
        i_year += 1
        print("Secid %s, %s/%s. Year %s, %s/%s" % (str(secid), str(i_secid), 
                                                   str(num_secid), str(year), 
                                                   str(i_year), str(num_years)))
        
        start_date = "'" + str(year) + "-01-01'"
        end_date = "'" + str(year) + "-12-31'"
        data_base = "OPTIONM.OPPRCD" + str(year)
        
        f = open("load_option_list.sql", "r")
        query = f.read()
        
        query = query.replace('\n', ' ').replace('\t', ' ')
        query = query.replace('_start_date_', start_date).replace('_end_date_', end_date)
        query = query.replace('_secid_', secid)
        query = query.replace('_data_base_', data_base)
        
        df_option_i = db.raw_sql(query)
        df_prices = df_prices.append(df_option_i)
        
############################################################
# Getting all the data:
############################################################
        
file_list_to_load = os.listdir("output") 
file_list_to_load  = [x for x in os.listdir("output")  if "var_ests_equity_" in x]

downloaded_1996_2005_file_list = list(range(81, 100 + 1, 1)) + list(range(101, 169, 1)) + list(range(401, 489, 1)) + list(range(701, 727+1,1)) + list(range(1001, 1006+1, 1))

for i in downloaded_1996_2005_file_list:
    print(i)
    if i == 81:
        df1 = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    elif i != 99:
        df1 = df1.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

downloaded_2006_2011_file_list = list(range(21, 60 + 1, 1)) + list(range(201, 262 + 1, 1)) + list(range(501, 580 + 1, 1)) + list(range(801, 823+1,1)) + list(range(1101, 1109 + 1, 1))

for i in downloaded_2006_2011_file_list:
    print(i)
    if i == 21:
        df2 = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    elif i != 38:
        df2 = df2.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

downloaded_2012_2017_file_list = list(range(61, 80 + 1, 1)) + list(range(301, 371 + 1, 1)) + list(range(601, 652 + 1, 1)) + list(range(901, 921+1,1)) + list(range(1201, 1210 + 1, 1))

for i in downloaded_2012_2017_file_list:
    print(i)
    if i == 61:
        df3 = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    else:
        df3 = df3.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

df = df1.append(df2).append(df3)

T_30_days = 30/365
#measure_list = ["", "_in_sample"]
measure_list = ["_5_5", "_otm_in_sample", "_otm_5_5", "_otm1_in_sample"]

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
            D_list.append(np.interp(T_30_days, x_new, y_new))
    
    D_df_to_merge = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
    D_df_to_merge = D_df_to_merge.sort_values(["secid", "date"])
    
    # Getting the first date of the month average the series:
    D_df_to_merge = D_df_to_merge.replace([np.inf, -np.inf], np.nan)
#    D_df_to_merge.loc[D_df_to_merge["D"] > 10, "D"] = np.nan # Handler for extermely large values
#    D_df_to_merge.loc[D_df_to_merge["D"] < 0, "D"] = 0 # Handler for negative values
    D_df_to_merge.columns = ["secid", "date", "D" + measure_list[i_measure]]
    
    if i_measure == 0:
        D_df = D_df_to_merge
    else:
        D_df = pd.merge(D_df, D_df_to_merge, on = ["secid", "date"])

#D_df.to_csv("D_df_all_test_2.csv")

########################################
# Loading data back again:
#D_df_1 = pd.read_csv("D_df_all_test.csv")
#D_df_2 = pd.read_csv("D_df_all_test_2.csv")
#D_df = pd.merge(D_df_1, D_df_2, on = ["secid", "date"], how = "outer")
#D_df = D_df.drop(["Unnamed: 0_x", "Unnamed: 0_y"], axis = 1)
#D_df.to_csv("D_df_all_test.csv")
#

D_df = pd.read_csv("D_df_all_test.csv")


#cnt_all = D_df.groupby("date")["D_in_sample"].size().reset_index().rename(columns={'D_in_sample': 'cnt_all'})
#cnt_not_null = D_df.groupby("date")["D_in_sample"].count().reset_index().rename(columns={'D_in_sample': 'cnt_not_null'})
#cnt_pos = D_df[D_df["D_in_sample"] > 0].groupby("date")["D_in_sample"].count().reset_index().rename(columns={'D_in_sample': 'cnt_pos'})
#
#cnt_merged = pd.merge(cnt_all, cnt_not_null)
#cnt_merged = pd.merge(cnt_merged, cnt_pos)
#
#cnt_merged["date"] = pd.to_datetime(cnt_merged["date"])
#cnt_merged.set_index("date").plot()
#
## Calculating shares:
#cnt_merged["share_not_null"] = cnt_merged["cnt_not_null"]/cnt_merged["cnt_all"]
#cnt_merged["share_positive"] = cnt_merged["cnt_pos"]/cnt_merged["cnt_all"]
#cnt_merged.set_index("date")[["share_not_null", "share_positive"]].plot()
#
## Averaging within a month:
#cnt_merged["date_trunc"] = cnt_merged["date"] - pd.offsets.MonthBegin(1)
#cnt_merged.groupby("date_trunc")[["share_not_null", "share_positive"]].mean().plot()

# Calculate share of null and share of positive for each of the measures:
measure_list = ["", "_in_sample", "_5_5", "_otm_in_sample", "_otm_5_5", "_otm1_in_sample"]
for (i_measure, measure) in enumerate(measure_list):
    print(i_measure)
    cnt_all = D_df.groupby("date")["D" + measure].size().reset_index().rename(columns={"D" + measure: "cnt_all"})
    cnt_not_null = D_df.groupby("date")["D" + measure].count().reset_index().rename(columns={"D" + measure: "cnt_not_null"})
    cnt_pos = D_df[D_df["D" + measure] > 0].groupby("date")["D" + measure].count().reset_index().rename(columns={"D" + measure: "cnt_pos"})
    
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



################################################
# Working with alrgest subset of the data
################################################
D_df = pd.read_csv("D_df_all_test.csv")

D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_trunc"] = D_df["date"] - pd.offsets.MonthBegin(1)

# Removing NaN

# 4. Getting only firms-months that have >= 15 observations:
def min_month_obs(x):
    return x["D"].count() > 10

#measure_list = ["", "_in_sample", "_5_5", "_otm_in_sample", "_otm_5_5", "_otm1_in_sample"]
#for (i_measure, measure) in measure_list:
#    print(i_measure)
measure = ""

D_df_short = D_df[["date", "date_trunc", "secid", "D" + measure]]

# Removing observations with NaN or values < 0. Can consider truncating them to 
# zero, for example:
D_df_short = D_df_short[~D_df_short["D" + measure].isnull()]
D_df_short = D_df_short[D_df_short["D" + measure] > 0]

# Filtering by requiring a company-month to have at least 15 observations:
D_df_filter = D_df_short.groupby(["secid", "date_trunc"]).filter(min_month_obs)

#secid_list = []
#num_secid = len(np.unique(D_df_filter["secid"]))
#D_measure_cov_list = np.zeros((9, 9, num_secid))
#i = 0
#for (name, sub_df) in D_df_filter.groupby(["secid"]):
#    D_measure_cov_list[:,:,i] = np.array(sub_df[["D" + measure]].corr())
#    i = i + 1
#
#av_corr = np.nanmean(D_measure_cov_list, axis = 2)
#pd.DataFrame(av_corr)

#D_df_filter_mo_mean = D_df_filter.groupby(["secid", "date_trunc"])[["D" + x for x in measure_list]].mean().reset_index()
#
#secid_list = []
#num_secid = len(np.unique(D_df_filter_mo_mean["secid"]))
#D_measure_cov_list = np.zeros((9,9,num_secid))
#i = 0
#for (name, sub_df) in D_df_filter_mo_mean.groupby(["secid"]):
#    D_measure_cov_list[:,:,i] = np.array(sub_df[["D" + x for x in measure_list]].corr())
#    i = i + 1
#
#av_corr = np.nanmean(D_measure_cov_list, axis = 2)
#pd.DataFrame(av_corr)


# 5. Getting average monthly for each firm and calculating average correlation:
#D_df_filter["date"] = pd.to_datetime(D_df_filter["date"])
#D_df_filter["date_trunc"] = D_df_filter["date"] - pd.offsets.MonthBegin(1)
av_mon_D = D_df_filter.groupby(["date_trunc"])[["D" + measure]].mean()
#av_mon_D.plot(figsize = (8,6))

# 7. Getting firms that are present in 80% of all months:
# Getting number of unique months:
D_df_filter_mo_mean = D_df_filter.groupby(["secid", "date_trunc"])[["D" + measure]].mean().reset_index()
num_months = len(np.unique(D_df_filter_mo_mean["date_trunc"]))
def min_sample_obs(x):
    return x["D"].count() > int(276*0.8)

D_df_mo_filter_filter = D_df_filter_mo_mean.groupby("secid").filter(min_sample_obs)
len(np.unique(D_df_mo_filter_filter["secid"]))

pca = PCA(n_components = 1)
df_pivot = pd.pivot_table(D_df_mo_filter_filter, values = "D" + measure, 
                          index=["date_trunc"], columns=["secid"], aggfunc=np.sum)
df_pivot = df_pivot.sort_values("date_trunc")
df_pivot.index = pd.to_datetime(df_pivot.index)

for i in range(len(df_pivot.columns)):
    df_pivot.iloc[np.array(np.isnan(df_pivot.iloc[:,i])), i] = np.nanmean(df_pivot.iloc[:,i])

X_D = np.array(df_pivot)   
pca.fit(X_D.T)

plt.plot(pca.components_[0])
pca.explained_variance_ratio_[0]


#PC1_D = pd.DataFrame({"date": np.unique(D_df_sub["date_trunc"].sort_values())})
#
#for i_measure in range(len(measure_list)):
#    PC1_D["D"+measure_list[i_measure]] = pc1_list[i_measure]
#        
#PC1_D = PC1_D.set_index("date")
#corr_D = np.array(PC1_D.corr())
#plot_heat_map_list([corr_D], list(PC1_D.columns), list(PC1_D.columns), "")     
#PC1_D.plot(figsize = (8,6))

############################################
# Loading data on one option
############################################


import wrds
db = wrds.Connection(wrds_username = "rsigalov")


start_date = "'2014-01-01'"
end_date = "'2014-12-31'"
data_base = "OPTIONM.OPPRCD2014"
secid = str(102583)

f = open("load_option_list.sql", "r")
query = f.read()

query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('_start_date_', start_date).replace('_end_date_', end_date)
query = query.replace('_secid_', secid)
query = query.replace('_data_base_', data_base)

df_option_i = db.raw_sql(query)

# saving this data
df_option_i.to_csv("cvs_data_2014.csv")

# loading aapl dist_data:
query = """
select 
    secid, ex_date, amount, distr_type
from OPTIONM.DISTRD
where secid = _secid_
and currency = 'USD'
"""

query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('_secid_', secid)
dist_data = db.raw_sql(query)

dist_data.to_csv("cvs_dist_data.csv")



####################################################
# Testing interpolations:
####################################################
#x = np.array([0.05, 0.08, 0.33, 0.45])
#y = np.array([0.1, 0.2, np.nan, 0.3])

x = np.array([0.05, 0.08, 0.33, 0.45])
y = np.array([0.1, 0.2, np.nan, np.nan])

np.interp(0.1, x, y)

stacked = np.vstack((x.T, y.T)).T

stacked_filter = stacked[~np.isnan(stacked).any(axis = 1)]

x_new = stacked_filter[:,0]
y_new = stacked_filter[:,1]

np.interp(0.1, x_new, y_new)
















