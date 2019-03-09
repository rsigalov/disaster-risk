"""
Disaster risk analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
import os
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

for i in range(1, 21, 1):
    if i == 1:        
        df = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    else:
        df = df.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

df["date"] = pd.to_datetime(df["date"])

#for i in range(11,21,1):
#    if i == 11:        
#        df = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
#    else:
#        df = df.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))
#
#df["date"] = pd.to_datetime(df["date"])

for i in range(21, 41, 1):
    print(i)
    if i == 21:
        df = pd.read_csv("output/var_ests_equity_" + str(i) + ".csv")
    elif i != 38:
        df = df.append(pd.read_csv("output/var_ests_equity_" + str(i) + ".csv"))

############################################
# Calculating share of NaN and Inf

T_days = np.arange(30,390,30)/365

    
df_pca = pd.DataFrame({"date":list(set(df_short["date"]))})
df_pca = df_pca.sort_values("date").reset_index().drop("index",axis = 1)

for t in T_days:
    
    secid_list = []
    date_list = []
    D_list = []
    D_ins_list = []
    
    for name, group in df_short.groupby(["secid", "date"]):
                
        secid_list.append(name[0])
        date_list.append(name[1])
    
        D_list.append(np.interp(T_150_days, group["T"], group["D"]))
        D_ins_list.append(np.interp(T_150_days, group["T"], group["D_in_sample"]))
    
    D_30 = pd.DataFrame({"secid":secid_list, "date":date_list, 
                         "D":D_list, "D_ins":D_ins_list})

    
    # Getting all in matrix form:
    pca = PCA(n_components=1)
    df_pivot = pd.pivot_table(D_30.replace([np.inf, -np.inf, np.nan], 0).dropna(), 
                           values = "D", index=["date"],
                           columns=["secid"], aggfunc=np.sum)
    df_pivot = df_pivot.sort_values("date")
    
    X_D = np.array(df_pivot)        
    X_D[~np.isfinite(X_D)] = 0
    pca.fit(X_D.T[0:100,:])
    plt.plot(pca.components_[0])
    print(pca.explained_variance_ratio_)



plt.plot(D_30[D_30["secid"] == 109117]["D"])
plt.plot(D_30[D_30["secid"] == 109117]["D_ins"])

df[df["secid"] == 101594]["D"].interpolate()


########################################################
# Comparing different measures
########################################################
measure_list = ["", "_in_sample", "_5_5", "_otm", "_otm_in_sample",
                "_otm_5_5", "_otm1", "_otm1_in_sample", "_otm1_5_5"]
#i_measure = 1

pc1_list = []
pc1_exp_var = []

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
    
        D_list.append(np.interp(T_30_days, sub_df["T"], sub_df["D"]))
    
    D_df = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
    D_df = D_df.sort_values(["secid", "date"])
    
    # Interpolating NAs:
    D_df = D_df.groupby("secid").apply(lambda group: group.interpolate(method='index'))
        
    D_df.loc[D_df["D"] < 0, "D"] = 0
    D_df.loc[D_df["D"] > 10, "D"] = 0 # Handler for extermely large values
    
    pca = PCA(n_components=3)
    df_pivot = pd.pivot_table(D_df.replace([np.inf, -np.inf, np.nan], 0).dropna(), 
                           values = "D", index=["date"],
                           columns=["secid"], aggfunc=np.sum)
    df_pivot = df_pivot.sort_values("date")
    df_pivot.index = pd.to_datetime(df_pivot.index)
    
    X_D = np.array(df_pivot)        
    X_D[~np.isfinite(X_D)] = 0
    pca.fit(X_D.T)
    
    pc1_list.append(pca.components_[0])
    pc1_exp_var.append(pca.explained_variance_ratio_[0])


# Plotting components together:
for i in range(len(measure_list)):
    plt.plot(pc1_list[i], label = measure_list[i])

dates = list(set(df["date"]))
dates.sort()

colnames = ["Full", "Full_in_sample", "5to5", "otm", "otm_in_sample",
            "otm_5_to_5", "otm1", "otm1_in_sample", "otm1_5_to_5"]

df_dict = {"date":dates}
for i in range(len(colnames)):
    df_dict[colnames[i]] = pc1_list[i]

df_D_PC1 = pd.DataFrame(df_dict)
np.array(df_D_PC1.corr())



plot_heat_map_list([np.array(df_D_PC1.corr())], colnames, colnames, 
                    "Correlation of differently measured D PC1, 2008",
                    "images/D_corr_2017.pdf")

# The ones that I leave are:
plt.plot(pc1_list[0])
plt.plot(pc1_list[1])
plt.plot(pc1_list[6])
plt.plot(pc1_list[7])

df_D_PC1["date"] = pd.to_datetime(df_D_PC1["date"])
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(df_D_PC1["date"], df_D_PC1["Full"], label = "0 to Infinity")
ax.plot(df_D_PC1["date"], df_D_PC1["Full_in_sample"], label = "Lowest to Highest Strike")
ax.plot(df_D_PC1["date"], df_D_PC1["otm1"], label = "0 to Spot*(1-sigma)")
ax.plot(df_D_PC1["date"], df_D_PC1["otm1_in_sample"], label = "Lowest strike to Spot*(1-sigma)")
ax.legend()
plt.savefig("images/PC1_D_comparison_2017.pdf", bbox_inches='tight', format = 'pdf')


df_D_PC1.set_index("date")[["Full", "Full_in_sample", "otm1", "otm1_in_sample"]].plot()

# Plotting explained variance for each measure




# Calculating the average of the measure within each company over a month:
i_measure = 2
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

    D_list.append(np.interp(T_30_days, sub_df["T"], sub_df["D"]))

D_df = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
D_df = D_df.sort_values(["secid", "date"])

# Getting the first date of the month average the series:
D_df["date_trunc"] = D_df["date"] - pd.offsets.MonthBegin(1)
D_df = D_df.replace([np.inf, -np.inf], 0)
D_df.loc[D_df["D"] > 10, "D"] = np.nan # Handler for extermely large values
D_df.loc[D_df["D"] < 0, "D"] = 0 # Handler for extermely large values

# Averaging D measure for each (secid, month):
av_D_df = D_df.groupby(["secid", "date_trunc"])["D"].mean().reset_index()

# Now taking the principal component:
pca = PCA(n_components = 1)
df_pivot = pd.pivot_table(av_D_df, 
                       values = "D", index=["date_trunc"],
                       columns=["secid"], aggfunc = np.sum)
df_pivot = df_pivot.sort_values("date_trunc")
X_D = np.array(df_pivot)        

pca.fit(X_D.T)

plt.plot(pca.components_[0])
print(pca.explained_variance_ratio_[0])



############################################################
# Calculating correlation between measures within a firm
#measure_list = ["Full", "Full_in_sample", "5to5"]
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
    
        D_list.append(np.interp(T_30_days, sub_df["T"], sub_df["D"]))
    
    D_df_to_merge = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
    D_df_to_merge = D_df_to_merge.sort_values(["secid", "date"])
    
    # Getting the first date of the month average the series:
    D_df_to_merge = D_df_to_merge.replace([np.inf, -np.inf], np.nan)
    D_df_to_merge.loc[D_df_to_merge["D"] > 10, "D"] = np.nan # Handler for extermely large values
    D_df_to_merge.loc[D_df_to_merge["D"] < 0, "D"] = 0 # Handler for extermely large values
    D_df_to_merge.columns = ["secid", "date", "D" + measure_list[i_measure]]
    
    if i_measure == 0:
        D_df = D_df_to_merge
    else:
        D_df = pd.merge(D_df, D_df_to_merge, on = ["secid", "date"])

# For each company calculate correlation between the measures:
secid_list = []
D_measure_cov_list = []
for (name, sub_df) in D_df.groupby(["secid"]):
    secid_list.append(name)
    D_measure_cov_list.append(np.array(sub_df[["D" + x for x in measure_list]].corr()))

# Calculating average correlation matrices for thes companies:
av_D_measure_corr = reduce(lambda x,y: x+ y, D_measure_cov_list)/len(D_measure_cov_list)

D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_trunc"] = D_df["date"] - pd.offsets.MonthBegin(1)
# Now average each measure within firm, month and calculate correlations:
av_D_df = D_df.groupby(["secid", "date_trunc"])[["D" + x for x in measure_list]].mean().reset_index()

secid_list = []
D_measure_cov_list = []
for (name, sub_df) in av_D_df.groupby(["secid"]):
    secid_list.append(name)
    D_measure_cov_list.append(np.array(sub_df[["D" + x for x in measure_list]].corr()))

av_D_measure_corr = reduce(lambda x,y: x+ y, D_measure_cov_list)/len(D_measure_cov_list)
pd.DataFrame(av_D_measure_corr).to_csv("output/monthly_av_companies_D_corr.csv")

################################################
# Looking at the probability of crash:
# 1. Interpolating risk-neutral probability of a crash along the
#    maturity axis to get a 30 day measure
secid_list = []
date_list = []
prob_2_sigma_list = []
prob_40ann_list = []

for (name, sub_df) in df.groupby(["secid", "date"]):                
    secid_list.append(name[0])
    date_list.append(name[1])

    prob_2_sigma_list.append(np.interp(T_30_days, sub_df["T"], sub_df["rn_prob_2sigma"]))
    prob_40ann_list.append(np.interp(T_30_days, sub_df["T"], sub_df["rn_prob_40ann"]))

rn_prob_df = pd.DataFrame({"secid": secid_list, "date": date_list, 
                           "rn_prob_2_sigma": prob_2_sigma_list, "rn_prob_40ann": prob_40ann_list})

# Plotting averages within date across firms for risk-neutral
# probability of a 2-sigma and 40% annualized drops
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(rn_prob_df.groupby("date")["rn_prob_2_sigma"].mean(), label = "2sigma")
ax.plot(rn_prob_df.groupby("date")["rn_prob_40ann"].mean(), label = "40% annualized")
ax.legend()
plt.savefig("images/rn_prob_disaster_2017.pdf", bbox_inches='tight', format = 'pdf')




############################################
# Looking at SPX


df1 = pd.read_csv("output/var_ests_spx_1.csv")
df2 = pd.read_csv("output/var_ests_spx_2.csv")

df_spx = df1.append(df2)
#df_spx["date"] = pd.to_datetime(df_spx["date"])

measure_list = ["", "_in_sample", "_5_5", "_otm", "_otm_in_sample",
                "_otm_5_5", "_otm1", "_otm1_in_sample", "_otm1_5_5"]

T_30_days = 30/365

for i_measure in range(len(measure_list)):
    print(i_measure)
    df_short = df_spx[["secid", "date", "T", "V" + measure_list[i_measure], 
                   "IV" + measure_list[i_measure]]]
    df_short["D"] = df_short["V" + measure_list[i_measure]] - df_short["IV" + measure_list[i_measure]]
    df_short = df_short.drop(["V" + measure_list[i_measure], "IV" + measure_list[i_measure]], axis = 1)
    
    date_list = []
    D_list = []
    
    for (name, sub_df) in df_short.groupby(["date"]):  
        date_list.append(name)
        D_list.append(np.interp(T_30_days, sub_df["T"], sub_df["D"]))
    
    D_df_to_merge = pd.DataFrame({"date":date_list, "D":D_list})
    D_df_to_merge = D_df_to_merge.sort_values(["date"])
    
    # Getting the first date of the month average the series:
    D_df_to_merge = D_df_to_merge.replace([np.inf, -np.inf], np.nan)
    D_df_to_merge.loc[D_df_to_merge["D"] > 10, "D"] = np.nan # Handler for extermely large values
    D_df_to_merge.loc[D_df_to_merge["D"] < 0, "D"] = 0 # Handler for extermely large values
    D_df_to_merge.columns = ["date", "D" + measure_list[i_measure]]
    
    if i_measure == 0:
        D_df = D_df_to_merge
    else:
        D_df = pd.merge(D_df, D_df_to_merge, on = ["date"])


D_df[["D" + x for x in measure_list]].corr().to_csv("output/spx_D_corr.csv")


# Averaging within a month and plotting:
D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_trunc"] = D_df["date"] - pd.offsets.MonthBegin(1)

D_df_av = D_df.groupby("date_trunc")[["D" + x for x in measure_list]].mean()

D_df_av[["D" + x for x in measure_list]].corr().to_csv("output/spx_D_corr_monthly.csv")

#### Comparing with Emil's series:
Emils_D = pd.read_csv("output/Emils_D.csv", names = ["date", "Emils_D"])
Emils_D["date"] = pd.to_datetime(Emils_D["date"])
Emils_D["date"] = Emils_D["date"] - pd.offsets.MonthBegin(1)
Emils_D = Emils_D.set_index("date")

D_short = D_df_av[["D", "D_in_sample"]].reset_index()
D_comp = pd.merge(Emils_D, D_short, left_on = "date", right_on = "date_trunc")

D_comp[["D", "D_in_sample", "Emils_D"]].corr()














def cmap_map(function, cmap):
    
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def plot_heat_map_list(matrix_list, xlabel_list, ylabel_list, matrix_label_list, path_to_save = None):
    num_matrices = len(matrix_list)

    if num_matrices == 1:
        i_corr_mat = 0
        x_tick_num = len(xlabel_list)
        y_tick_num = len(ylabel_list)

        fig, ax = plt.subplots(1,1,figsize=(6,6))

        colormap = cmap_map(lambda x: 0.5 * x + 0.5, matplotlib.cm.Oranges)

        im = ax.imshow(np.array(matrix_list[i_corr_mat]), cmap = colormap)

        ax.set_xticks(np.arange(x_tick_num))
        ax.set_yticks(np.arange(y_tick_num))
        ax.set_yticklabels(xlabel_list)
        ax.set_xticklabels(ylabel_list)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(x_tick_num):
            for j in range(y_tick_num):
                text = ax.text(j, i, round(matrix_list[i_corr_mat][i, j] * 100,1),
                               ha="center", va="center", color="black", weight = 'bold')

        ax.set_title(matrix_label_list)

    else:

        fig, axes = plt.subplots(nrows = num_matrices // 2 + 1, ncols = 2,  figsize = (7, 3.5 * (num_matrices // 2 + 1)))
        axes_list = [item for sublist in axes for item in sublist] 

        colormap = cmap_map(lambda x: 0.5 * x + 0.5, matplotlib.cm.Oranges)

        for i_corr_mat in range(num_matrices):
            ax = axes_list.pop(0)
            im = ax.imshow(np.array(matrix_list[i_corr_mat]), cmap = colormap)

            ax.set_xticks(np.arange(x_tick_num))
            ax.set_yticks(np.arange(y_tick_num))
            ax.set_yticklabels(xlabel_list)
            ax.set_xticklabels(ylabel_list)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(x_tick_num):
                for j in range(y_tick_num):
                    text = ax.text(j, i, round(matrix_list[i_corr_mat][i, j] * 100,1),
                                   ha="center", va="center", color="black", weight = 'bold')

            ax.set_title(matrix_label_list[i_corr_mat])

        for ax in axes_list:
            ax.remove()

    plt.tight_layout()
    if path_to_save is None:
    	plt.show()
    else:
    	plt.savefig(path_to_save, bbox_inches='tight', format = 'pdf')







