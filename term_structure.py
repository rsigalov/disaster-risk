"""
Analysis of term structure of disaster risk
"""
# Libraries to execute the script from terminal
from __future__ import print_function
from __future__ import division
import sys
from optparse import OptionParser

# Libraries for analysis
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# For PCA
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigs

MIN_OBS_PER_MONTH = 5
VARIABLE = "D_clamp"

#def main(argv = None):
#
#	################################################################
#    # Loading data on different maturity interpolated D
#    ################################################################
#    parser = OptionParser()
#
#    def get_comma_separated_args(option, opt, value, parser):
#        setattr(parser.values, option.dest, value.split(','))
#
#    parser.add_option('-v','--variable', action="store",
#        type='string', dest='VARIABLE',
#        help = """Variable to use for level factor construction, e.g. D_Clamp, rn_prob_80""")
#    parser.add_option("-m", "--min_mon_obs", action="store",
#		type="string", dest="MIN_OBS_PER_MONTH",
#		help = "Minimum number of observation per month")
#
#    (options, args) = parser.parse_args()
#    MIN_OBS_PER_MONTH = int(options.MIN_OBS_PER_MONTH)
#    VARIABLE = options.VARIABLE



################################################################
# Loading data on different maturity interpolated D
################################################################
int_d_columns = ["secid", "date"] + [VARIABLE]
int_d = pd.DataFrame(columns = int_d_columns)

for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    int_d_tmp = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_days_" + str(days) + ".csv")
    int_d_tmp = int_d_tmp[int_d_columns]
    int_d_tmp["m"] = days
    int_d = int_d.append(int_d_tmp)

################################################################
# Calculating level factor as taking the cross section average 
# across all companies and all maturities at a given date. 
# (Need to make sure to use companies for which all maturities 
# exist).
################################################################
int_d_var = int_d[["secid","date", "m", VARIABLE]]
int_d_var = int_d_var[~int_d_var[VARIABLE].isnull()]

# Pivoting table to only leave observations with full term structure present:
pivot_mat = pd.pivot_table(int_d_var, index = ["secid", "date"], columns = "m", values = VARIABLE)
pivot_mat = pivot_mat.dropna().reset_index()

# Leaving only (secid, date) with the full term structure, got
# these on the previous step
int_d_var = pd.merge(pivot_mat[["secid", "date"]], int_d_var, 
                     on = ["secid", "date"], how = "left")


################################################################
# Averaging within a month for each company and calculating number 
# of days with observations. Later going to filter by the number 
# of observations in a month, e.g. if the secid has more than 5 
# observations in a given month we include it in the
# sample and use it for calculations later on
################################################################
def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.99)) & (x >= np.quantile(x, 0.01))])

pivot_mat["date"] = pd.to_datetime(pivot_mat["date"])
pivot_mat["date_mon"] = pivot_mat["date"] + pd.offsets.MonthEnd(0)

# Counting number of full observations for each (secid, month).
mean_cnt_df = pivot_mat.groupby(["secid", "date_mon"])["date"].count().reset_index()
mean_cnt_df = mean_cnt_df.rename({"date": "cnt"}, axis = 1)

# Filtering by the minimum number of observations:
mean_cnt_df = mean_cnt_df[mean_cnt_df.cnt >= MIN_OBS_PER_MONTH]
num_secids_per_month = mean_cnt_df.reset_index().groupby("date_mon")["secid"].count()

# Merging this back to dataframe with individual disaster series:
int_d_var["date"] = pd.to_datetime(int_d_var["date"])
int_d_var["date_mon"] = int_d_var["date"] + pd.offsets.MonthEnd(0)
int_d_var = pd.merge(mean_cnt_df, int_d_var, on = ["secid", "date_mon"], how = "left")

# Calculating the level factor. First average the measure for each (secid, month).
# Next, average with truncation at 1% level across companies within each month.
level_factor = int_d_var.groupby(["secid", "date_mon"])[VARIABLE] \
                        .mean()\
                        .reset_index() \
                        .groupby("date_mon")[VARIABLE] \
                        .apply(mean_with_truncation) \
                        .rename("level_factor")
                        
# Averaging within each (secid, month, maturity):
int_d_mon_mat = int_d_var.groupby(["secid", "date_mon", "m"])[VARIABLE].mean().reset_index()


################################################################
# Trying to calculate slope and curvature factors
################################################################
# Estimating level as mean across term structure (TS)
#            slope as mean slope across TS
#            curvature as rate of change of slope across TS
def lsc_from_term_structure(sub_df):
    sub_df = sub_df.sort_values(["m"])
    x = np.array(sub_df[VARIABLE])
    level = np.mean(x)
    slope_x = (x[1:6]-x[0:5])/30
    slope = np.mean(slope_x)
    curve_x = (slope_x[1:5] - slope_x[0:4])/30
    curve = np.mean(curve_x)
    return (level, slope, curve)

lsc_ind_date_der = int_d_var.groupby(["secid", "date"]).apply(lsc_from_term_structure).apply(pd.Series)
lsc_ind_date_der.columns = ["level", "slope", "curve"]
lsc_ind_date_der = lsc_ind_date_der.reset_index()

lsc_ind_date_der["date_mon"] = lsc_ind_date_der["date"] + pd.offsets.MonthEnd(0)
lsc_ind_month_der = lsc_ind_date_der.groupby(["secid","date_mon"])["level", "slope", "curve"].mean()
lsc_ind_month_der = lsc_ind_month_der.reset_index()

lsc_factors_der = lsc_ind_month_der.groupby("date_mon")["level", "slope", "curve"].apply(mean_with_truncation).apply(pd.Series)
for i in range(3):
    lsc_factors_der.iloc[:,i] = lsc_factors_der.iloc[:,i]/np.std(lsc_factors_der.iloc[:,i])


################################################################
# Doing a PCA exercise. Due to a small number of firms with large
# panel this can be problematic.
################################################################    
# First to look at the number of firms that we have let's 
# calculate the share of total months that the firm is present
# in the sample:
total_number_of_months = len(np.unique(int_d_mon_mat["date_mon"]))
secid_share_months = int_d_mon_mat[int_d_mon_mat.m == 30].groupby("secid")["date_mon"].count().sort_values(ascending = False)/total_number_of_months

# Taking firms that are present in at least 50% of the sample.
# There is a total of 126 such firms.
min_month_share = 0.45
secid_list = list(secid_share_months.index[secid_share_months >= min_month_share])
num_secid = len(secid_list)

# Calculating correlation matrix between all these firms:
df_pivot = pd.pivot_table(
        int_d_mon_mat[int_d_mon_mat.secid.isin(secid_list)], 
        index = "date_mon", columns = ["m", "secid"], values = VARIABLE)



w1, pc1 = construct_pc_unbalanced(df_pivot, 1)
w2, pc2 = construct_pc_unbalanced(df_pivot, 2)
w3, pc3 = construct_pc_unbalanced(df_pivot, 3)
pc_df = pd.merge(pc1, pc2, left_index = True, right_index = True)
pc_df = pd.merge(pc_df, pc3, left_index = True, right_index = True)

# Calculating table with summary statistics on loadings for different
# maturities. The idea is to show that the first principal component
# is flat across maturities and second principal component has some
# meaningful differences between maturities, e.g. slope
col_list = ["load_" + str(x) for x in [30, 60, 90, 120, 150, 180]]
w1_mat_split = pd.DataFrame(columns = col_list)
w2_mat_split = pd.DataFrame(columns = col_list)
w3_mat_split = pd.DataFrame(columns = col_list)

for i in range(6):
    w1_mat_split.iloc[:,i] = w1[i*num_secid:(i + 1)*num_secid]
    w2_mat_split.iloc[:,i] = w2[i*num_secid:(i + 1)*num_secid]
    w3_mat_split.iloc[:,i] = w3[i*num_secid:(i + 1)*num_secid]

print("Summary statistics for loadings on PC1")
print("--------------------------------------------------------------")
w1_mat_sum_stat = (w1_mat_split*1000).describe().round(3).loc[["count", "mean", "std", "25%", "50%", "75%"],:]
print(w1_mat_sum_stat)
print("")
print("Summary statistics for loadings on PC2")
print("--------------------------------------------------------------")
w2_mat_sum_stat = (w2_mat_split*1000).describe().round(3).loc[["count", "mean", "std", "25%", "50%", "75%"],:]
print(w2_mat_sum_stat)
print("")
print("Summary statistics for loadings on PC3")
print("--------------------------------------------------------------")
w3_mat_sum_stat = (w3_mat_split*1000).describe().round(3).loc[["count", "mean", "std", "25%", "50%", "75%"],:]
print(w3_mat_sum_stat)
print("")

# Looking at box plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 4))
r = axes[0].boxplot([np.array(w1_mat_split.iloc[:,i]) for i in range(w1_mat_split.shape[1])])
r = axes[1].boxplot([np.array(w2_mat_split.iloc[:,i]) for i in range(w1_mat_split.shape[1])])
r = axes[2].boxplot([np.array(w3_mat_split.iloc[:,i]) for i in range(w1_mat_split.shape[1])])
axes[0].set_title('PC1')
axes[1].set_title('PC2')
axes[2].set_title('PC3')
plt.setp(axes, xticks = [1,2,3,4,5,6],
         xticklabels = [30, 60, 90, 120, 150, 180])
plt.show()


# In order to construct the principal components we need to carefully
# reweight the data so that weights sum up to unity for each date:
# Matrix with dummies if the observation is present:
def construct_pc_unbalanced(df_pivot, pc_num):
    '''
    Gets as input a pivoted dataframe where each column is a time
    series observation and each row is date
    '''
    corr_df = df_pivot.corr()
    eigs_tuple = eigs(np.array(corr_df.fillna(0)),pc_num)
    w_raw = eigs_tuple[1][:,pc_num - 1].astype(float).flatten()
    w_raw = w_raw/np.sum(w_raw)
    
    x = np.array(df_pivot.fillna(0))
    d = np.where(np.array(~df_pivot.isnull()), 1, 0)
    w = np.tile(w_raw.reshape((1,-1)), (df_pivot.shape[0], 1))
    dw = d * w
    dw_sum = np.sum(dw, axis = 1).reshape((-1,1))
    dw_sum = np.tile(dw_sum, (1, x.shape[1]))
    w = w / dw_sum
    pc = np.sum(x * w, axis = 1)  
    
    w_df = pd.DataFrame({"W" + str(pc_num):w_raw}, index = df_pivot.columns)
    pc_df = pd.DataFrame({"PC" + str(pc_num):pc}, index = df_pivot.index)
    return w_df, pc_df


def construct_pcs(df, num_pcs = 3, method = "unbalanced", min_share_obs = 0.5):
    '''
    Function that calculates principal components in differen ways.
      1. PC based on unbalanced panel where observations are
         reweighted to make the weights sum up to 1 at each date
      2. PC based on largely balanced panel, where missing
         variables are filled with the time series mean of a
         particular observation.
         
    Takes as inputs:
        
        df - 
             
        method - balanced vs. unbalanced method

        min_share_obs - minimum share of observations needed to
             be included in PCA
    '''
    
    # First to look at the number of firms that we have let's 
    # calculate the share of total months that the firm is present
    # in the sample:
    total_number_of_months = len(np.unique(df["date_mon"]))
    secid_share_months = df[df.m == 30].groupby("secid")["date_mon"].count().sort_values(ascending = False)/total_number_of_months
    
    secid_list = list(secid_share_months.index[secid_share_months >= min_share_obs])
    
    # Calculating correlation matrix between all these firms:
    df_pivot = pd.pivot_table(
            df[df.secid.isin(secid_list)], 
            index = "date_mon", columns = ["m", "secid"], values = VARIABLE)
    
    print(df_pivot.shape)
    
    if method == "unbalanced":
        pc_list = [] 
        w_list = []
        for i in range(1, num_pcs + 1,1):
            w_i, pc_i = construct_pc_unbalanced(df_pivot, i)
            pc_list.append(pc_i)
            w_list.append(w_i)
            
        # Constructing dataframes with weights and principal components:
        pc_df = reduce(lambda df1, df2: 
            pd.merge(df1,df2,left_index = True, right_index = True), pc_list)
        w_df = reduce(lambda df1, df2: 
            pd.merge(df1,df2,left_index = True, right_index = True), w_list)    
            
        return w_df, pc_df
    elif method == "balanced_fill":
        # First, filling missing values with time series averages
        for i_col in range(len(df_pivot.columns)):
            x = np.array(df_pivot.iloc[:,i_col])
            mean_x = np.nanmean(x)
            df_pivot.iloc[:, i_col].loc[df_pivot.iloc[:,i_col].isnull()] = mean_x
            
        corr_df = np.corrcoef(np.array(df_pivot).T)
        eigenvalue_decompos = eigs(np.array(corr_df), corr_df.shape[0] + 1)
        all_eigs = eigenvalue_decompos[0].astype(float)
        exp_var = all_eigs/np.sum(all_eigs)
        w = eigenvalue_decompos[1].astype(float)[:, range(num_pcs)]
        w = w/np.tile(np.sum(w, axis = 0).reshape((-1,num_pcs)), (w.shape[0], 1))
        pc = (np.array(df_pivot) @ w)
        
        pc_df = pd.DataFrame(pc_post, index = df_pivot.index, columns = ["PC" + str(i+1) for i in range(num_pcs)])
        w_df = pd.DataFrame(w, index = df_pivot.columns)
        
        return w_df, pc_df, exp_var[range(num_pcs)]
    else:
        raise ValueError("method can be 'unbalanced' or 'balanced_fill'")
                

w_df, pc_df = construct_pcs(
        int_d_mon_mat, num_pcs = 3, method = "unbalanced", min_share_obs = 0.5)

w_df, pc_df, exp_var = construct_pcs(
        int_d_mon_mat[int_d_mon_mat.date_mon >= "2003-01-01"], 
        num_pcs = 3, method = "balanced_fill", min_share_obs = 0.8)

################################################################
# Writing a function that takes in a data frame with observations
# normalizes them to have mean zero and unit standard deviation
# and runs a regression on factors supplied by the user. Before
# running the regression the function orthgonalizes the factors
def estimate_regs_on_factor(df, factors):      
    num_factors = factors.shape[1]
    # Normalizing the variables in the data frame:
    for i_col in range(df.shape[1]):
        x = np.array(df.iloc[:,i_col])
        df.iloc[:, i_col] = (x - np.nanmean(x))/np.nanstd(x) 
        
    # Estimating the regression of each series on the first principal
    # component. From that we can calculate residuals and calculate
    # the overall R^2 which will say what is the share of explained
    # variance by the first PC.

    reg_df = pd.merge(df, factors, on = ["date_mon"], how = "left") 
    
    # Sequentially estimating regressions on more and more factors:
    # (1) on factor 1, (2) on factors 1 and 2, (3) on factors 1,2
    # and 3, and so on...
    results_list = [] # list with regression results
    index_list = [] # list with indices of form (secid, m, num_factors)
    for i_col in range(df.shape[1]):
        for i_factors in range(num_factors):
            index_list.append(df.columns[i_col] + (i_factors+1, ))
            Y = df.iloc[:,i_col].rename("Y")
            X = factors.iloc[:, range(i_factors + 1)]      
            # Merging left-hand variable on factors to have the same 
            # time periods
            reg_df = pd.merge(Y, X, left_index = True, right_index = True)
            Y = np.array(reg_df.iloc[:,0])
            X = np.array(reg_df.iloc[:, range(1, i_factors+2,1)])
            X = sm.add_constant(X)
            res = sm.OLS(Y, X, missing = "drop").fit()
            results_list.append(res)

    # Aggregating variance into one measure. First Calculating the
    # sum of all squared deviations from the mean. Since we normalized
    # the mean of each series to be zero, this is just the sum of
    # squares of all the columns. Next calculate the sum of squared
    # residuals in a regression without a constant. Next, compare the
    # improvement in a model with one factor.
    # 
    # Going through regression with different number of factors and
    # and calculating aggregate R squared for each of them. This will
    # give a sense of how much each factor contributes to explaining 
    # the variability in the data        
    TSS = np.nansum(np.power(df, 2)) # TSS - same for all regressions
    R2_agg_list = []
    for i_factors in range(num_factors):
        reg_list_factor = [results_list[x] for x in range(len(results_list)) if (index_list[x][2] - 1 == i_factors)]
        RSS = np.sum([np.sum([x**2 for x in y.resid]) for y in reg_list_factor])
        R2_agg_list.append((TSS - RSS)/TSS)
    
    # Returing statistics on regressions in a dataframe:    
    res_df_to_return = pd.DataFrame({
            "m": [x[0] for x in index_list],
            "secid": [x[1] for x in index_list],
            "pc_num": [x[2] for x in index_list],
            "R2": [x.rsquared for x in results_list],
            "N": [x.nobs for x in results_list]})

    return R2_agg_list, res_df_to_return


R2_agg_list, reg_factors_df = estimate_regs_on_factor(df_pivot, pc_df)
R2_agg_list, reg_factors_df = estimate_regs_on_factor(df_pivot, lsc_factors_der)
R2_agg_list, reg_factors_df = estimate_regs_on_factor(df_pivot, pc_post_df)


def pc_weights_sum_stats(w, path_to_write = None):
    '''
    The input should be a dataframe with multiindex where the first level
    is maturity (e.g. 30 or 120) and second level is secid.
    
    If path_to_write is not None, then the latex table will be constructed
    and saved at a specified path
    '''
    
    # Looping through different principal component
    
    # Looping through different matrities:
    for m in w_df.index.levels[0]:
        
    




################################################################
# Doing an actual PCA for the post 2002 period where there are
# firms that are present in a large part of the sample

# Taking firms that are present in at least 75% of post 2002
# time period. There is a total of 53 such firms.
total_number_of_months = len(np.unique(int_d_mon_mat[int_d_mon_mat.date_mon >= "2003-01-01"]["date_mon"]))
secid_share_months = int_d_mon_mat[(int_d_mon_mat.m == 30) & (int_d_mon_mat.date_mon >= "2003-01-01")].groupby("secid")["date_mon"].count().sort_values(ascending = False)/total_number_of_months

min_month_share = 0.75
secid_list = list(secid_share_months.index[secid_share_months >= min_month_share])
num_secid = len(secid_list)

# Calculating correlation matrix between all these firms:
int_d_mon_mat_sub = int_d_mon_mat[int_d_mon_mat.secid.isin(secid_list)]
int_d_mon_mat_sub = int_d_mon_mat_sub[int_d_mon_mat_sub.date_mon >= "2003-01-01"] #& ([int_d_mon_mat.date_mon >= "2006-01-01"])]
df_pivot = pd.pivot_table(
        int_d_mon_mat_sub, 
        index = "date_mon", columns = ["m", "secid"], values = VARIABLE)

# For the part where I take the second half of the sample, I am going
# to fill the missing values with a time series mean for this particular
# company:
for i_col in range(len(df_pivot.columns)):
    x = np.array(df_pivot.iloc[:,i_col])
    mean_x = np.nanmean(x)
    df_pivot.iloc[:, i_col].loc[df_pivot.iloc[:,i_col].isnull()] = mean_x

corr_df = np.corrcoef(np.array(df_pivot).T)
eigenvalue_decompos = eigs(np.array(corr_df), corr_df.shape[0] + 1)
all_eigs = eigenvalue_decompos[0].astype(float)
exp_var = all_eigs/np.sum(all_eigs)
w_post = eigenvalue_decompos[1].astype(float)[:,[0,1,2]]
w_post = w_post/np.tile(np.sum(w_post, axis = 0).reshape((-1,3)), (w_post.shape[0], 1))
pc_post = (np.array(df_pivot) @ w_post)
pc_post_df = pd.DataFrame(pc_post, index = df_pivot.index, columns = ["PC1_sub", "PC2_sub", "PC3_sub"])

# Now need to compare this with the principal component that I obtained
# in unbalanced panel.
pc_all_df = pd.merge(pc_df, pc_post_df, left_index = True, right_index = True, how = "left")
pc_all_df[["PC1", "PC1_sub"]].plot()
pc_all_df.diff().corr()

# Looking at how the loading are distributed between maturities:
col_list = ["load_" + str(x) for x in [30, 60, 90, 120, 150, 180]]
w1_mat_split = pd.DataFrame(columns = col_list)
w2_mat_split = pd.DataFrame(columns = col_list)
w3_mat_split = pd.DataFrame(columns = col_list)

for i in range(6):
    w1_mat_split.iloc[:,i] = w_post[i*num_secid:(i + 1)*num_secid,0]
    w2_mat_split.iloc[:,i] = w_post[i*num_secid:(i + 1)*num_secid,1]
    w3_mat_split.iloc[:,i] = w_post[i*num_secid:(i + 1)*num_secid,2]

print("Summary statistics for loadings on PC1")
print("-----------------------------------------------------------------------------")
w1_mat_sum_stat = w1_mat_split.describe().loc[["count", "mean", "std", "25%", "50%", "75%"],:]
print(w1_mat_sum_stat)
print("")
print("Summary statistics for loadings on PC2")
print("-----------------------------------------------------------------------------")
w2_mat_sum_stat = w2_mat_split.describe().loc[["count", "mean", "std", "25%", "50%", "75%"],:]
print(w2_mat_sum_stat)
print("")
print("Summary statistics for loadings on PC3")
print("-----------------------------------------------------------------------------")
w3_mat_sum_stat = w3_mat_split.describe().loc[["count", "mean", "std", "25%", "50%", "75%"],:]
print(w3_mat_sum_stat)
print("")

# Looking at violin plot:
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 4))
r = axes[0].boxplot([np.array(w1_mat_split.iloc[:,i]) for i in range(w1_mat_split.shape[1])])
r = axes[1].boxplot([np.array(w2_mat_split.iloc[:,i]) for i in range(w1_mat_split.shape[1])])
r = axes[2].boxplot([np.array(w3_mat_split.iloc[:,i]) for i in range(w1_mat_split.shape[1])])
axes[0].set_title('PC1')
axes[1].set_title('PC2')
axes[2].set_title('PC3')
plt.setp(axes, xticks = [1,2,3,4,5,6],
         xticklabels = [30, 60, 90, 120, 150, 180])
plt.show()








################################################################
# Running individual (secid, month, maturity) disaster measures
# on level factor
################################################################

    
    
#reg_df = pd.merge(int_d_mon_mat, level_factor, on = ["date_mon"], how = "left")
reg_df = pd.merge(int_d_mon_mat, lsc_factors_der, on = ["date_mon"], how = "left") 
                       
secid_list = np.unique(reg_df["secid"])

results_list = []
secid_name_list = []
m_name_list = []
for i_secid, secid in enumerate(secid_list):
    if i_secid % 100 == 0:
        print("secid %d out of %d" % (i_secid, len(secid_list)))
    for m in [30, 60, 90, 120, 150, 180, "All"]:
        secid_name_list.append(secid)
        m_name_list.append(m)
        if m == "All":
            reg_df_sub = reg_df[reg_df.secid == secid]
            results_list.append(smf.ols(formula = VARIABLE + " ~ level + slope + curve", data = reg_df_sub).fit())
        else:    
            reg_df_sub = reg_df[(reg_df.secid == secid) & (reg_df.m == m)]
            results_list.append(smf.ols(formula = VARIABLE + " ~ level + slope + curve", data = reg_df_sub).fit())

reg_res_df = pd.DataFrame({"secid": secid_name_list,
              "m": m_name_list,
              "alpha": [x.params[0] for x in results_list],
              "alpha_se": [x.bse[0] for x in results_list],
              "beta_level": [x.params[1] for x in results_list],
              "beta_level_se": [x.bse[1] for x in results_list],
              "beta_slope": [x.params[2] for x in results_list],
              "beta_slope_se": [x.bse[2] for x in results_list],
              "beta_curve": [x.params[3] for x in results_list],
              "beta_curve_se": [x.bse[3] for x in results_list],
              "R2": [x.rsquared for x in results_list],
              "N": [x.nobs for x in results_list]})
    
    

####################################################################################
#reg_res_df[(reg_res_df.N >= 12*10) & (reg_res_df.m == 30)].R2.plot.hist()
#reg_res_df[(reg_res_df.N >= 12*10) & (reg_res_df.m == "All")].R2.plot.hist()
#    
#
#pd.pivot_table(reg_res_df[(reg_res_df.N >= 12*10) & (reg_res_df.m.isin([30,180]))][["secid","m", "beta_level"]],index = "secid", columns = "m").plot.hist(alpha = 0.5)
#
#
#reg_res_df[(reg_res_df.N >= 12*10) & (reg_res_df.m != "All")].R2.mean()


# Saving results:
int_d_mon_mat.to_csv("estimated_data/term_structure/ind_secid_term_structure_" + VARIABLE + ".csv")
reg_res_df.to_csv("estimated_data/term_structure/regressions_level_factor_" + VARIABLE + ".csv")

#reg_res_df.to_csv("estimated_data/term_structure/regressions_lsc_D_clamp.csv")


#if __name__ == "__main__":
#	sys.exit(main(sys.argv))







###################################################################################
# Some additional stuff to add later
###################################################################################
#
## Regressing the term structure on m and m^2 to get the slope and curvature:
#def get_slope_curvature(sub_df):
#    sub_df["m2"] = np.power(sub_df["m"], 2)
#    res = smf.ols(formula = VARIABLE + " ~ m + m2", data = sub_df).fit()
#    return (res.params[0], res.bse[0], res.params[1], res.bse[1],res.params[2], res.bse[2], res.rsquared)
#
#sub_mon_d = int_d_mon_mat[int_d_mon_mat.secid == 101328]
#slope_curv = sub_mon_d.groupby(["secid", "date_mon"]).apply(get_slope_curvature)
#slope_curv = slope_curv.apply(pd.Series)
#slope_curv.columns = ["alpha", "alpha_se", "beta", "beta_se", "gamma", "gamma_se", "R2"]
#
#
#slope_curv[(np.abs(slope_curv["gamma"]/slope_curv["gamma_se"]) < 2) & (slope_curv["beta"]/slope_curv["beta_se"] > 2)]
#
#slope_curv[slope_curv["gamma"]/slope_curv["gamma_se"] < -2]
#
#
#def get_features(sub_df):
#    value_vec = np.array(sub_df[VARIABLE])
#    slope_vec = np.zeros(value_vec.shape[0]-1)
#    curv_vec = np.zeros(value_vec.shape[0]-2)
#    
#    for i in range(value_vec.shape[0] - 1):
#        slope_vec[i] = (value_vec[i+1]-value_vec[i])/30
#        
#    for i in range(value_vec.shape[0] - 2):
#        curv_vec[i] = (value_vec[i+2]+value_vec[i])/2 - value_vec[i+1] 
#        
#    return (slope_vec, curv_vec)
#
#def classify_ts(sub_df):
#    sub_df = sub_df.sort_values("m")
#    value_vec = np.array(sub_df[VARIABLE])
#    slope_1 = value_vec[2] - value_vec[0]
#    slope_2 = value_vec[5] - value_vec[3]
#    
#    if (slope_1 >= 0) & (slope_2 > 0):
#        return "upward"
#    elif (slope_1 > 0) & (slope_2 <0):
#        return "hump"
#    elif (slope_1 < 0) & (slope_2 <0):
#        return "downward"
#    elif (slope_1 < 0) & (slope_2 >0):
#        return "inverse_hump"
#    else:
#        return np.nan
#
#
#classes = int_d_mon_mat[int_d_mon_mat.secid == 101328].groupby("date_mon").apply(classify_ts)
#for i in range(classes.shape[0]):
#    fig, ax = plt.subplots()
#    date = classes.index[i]
#    ts_type = classes[i]
#    sub_df = int_d_mon_mat[(int_d_mon_mat.secid == 101328) & (int_d_mon_mat.date_mon == date)]
#    ax.plot([30*(x+1) for x in range(6)], np.array(sub_df[VARIABLE]))
#    ax.set_title(ts_type + ", " + str(date))
#    fig.savefig("estimated_data/term_structure/test_classification/plot_" + str(i) + ".png")
#
#
#
#classes_all = int_d_mon_mat.groupby(["secid","date_mon"]).apply(classify_ts).rename("ts_type").reset_index()
#cnt_types = classes_all.groupby(["date_mon", "ts_type"])["secid"].count().rename("cnt").reset_index()
#pivot_types = pd.pivot_table(cnt_types, index = "date_mon", columns = "ts_type", fill_value = 0)
#total_types = classes_all.groupby("date_mon")["secid"].count().rename("total")
#
#share_types = pd.DataFrame(np.array(pivot_types)/np.array(total_types)[:,None])
#share_types.index = total_types.index
#share_types.columns = ["downward", "hump", "inverse_hump", "upward"]
#share_types.plot.area(figsize = (10,7))
#
#pd.pivot_table(cnt_types, index = "date_mon", columns = "ts_type").plot.area(figsize = (10,7))
#pd.pivot_table(cnt_types, index = "date_mon", columns = "ts_type").plot(figsize = (10,7))
#
#pd.pivot_table(cnt_types, index = "date_mon", columns = "ts_type")[[("cnt","upward"),("cnt", "hump")]].plot(figsize = (10,7))
#pd.pivot_table(cnt_types, index = "date_mon", columns = "ts_type")[[("cnt","downward"),("cnt", "inverse_hump")]].plot(figsize = (10,7))
#
#
#####################################################################################
## To show:
#
#lsc_factors_der.plot(figsize = (10,7))
#
#lsc_factors_der.diff().corr()
#
#lsc_factors_der.diff().plot(figsize = (10,7))
#
#lsc_lags = lsc_factors_der.shift(1)
#lsc_lags.columns = ["level_lag", "slope_lag", "curve_lag"]
#lsc_lags = pd.merge(lsc_lags, lsc_factors_der, left_index = True, right_index = True)
#
#smf.ols(formula = "level ~ level_lag + slope_lag + curve_lag", data = lsc_lags.diff()).fit().summary()
##smf.ols(formula = "level ~ level_lag + slope_lag + curve_lag", data = lsc_lags[lsc_lags.index <= "2008-8-31"].diff()).fit().summary()
##smf.ols(formula = "level ~ level_lag + slope_lag + curve_lag", data = lsc_lags[lsc_lags.index > "2009-01-31"].diff()).fit().summary()
#
#plot_df = lsc_lags.diff()
#plot_df = plot_df[plot_df.index != "2001-01-31"]
#plot_out_lehman = plot_df[(plot_df.index >= "2009-02-28") | (plot_df.index <= "2008-08-31")]
#plot_during_lehman = plot_df[(plot_df.index < "2009-02-28") & (plot_df.index > "2008-08-31")]
#
#fig, ax = plt.subplots(figsize = (10,6.5))
#ax.scatter(plot_out_lehman["slope_lag"], plot_out_lehman["level"])
#ax.scatter(plot_during_lehman["slope_lag"], plot_during_lehman["level"], color = "red")
#plt.xlabel("Lag of Delta Slope")
#plt.ylabel("Current Value of Delta Level")
#for i, txt in enumerate(plot_during_lehman.index):
#    ax.annotate(txt.strftime('%Y-%m-%d'), (plot_during_lehman["slope_lag"].iloc[i], plot_during_lehman["level"].iloc[i]))
#
#
#plot_out_lehman["post_lehman"] = np.where(plot_out_lehman.index > "2008-09-15",1,0)
#smf.ols(formula = "level ~ level_lag + slope_lag + curve_lag", data = plot_out_lehman).fit().summary()
#smf.ols(formula = "level ~ level_lag*post_lehman + slope_lag*post_lehman + curve_lag*post_lehman", 
#        data = plot_out_lehman).fit().summary()
#
## 0. calculating slope and curvature weighting functions/vectors:
#w_slope = np.array([-1 + (x - 30)/75 for x in [30, 60, 90, 120, 150,180]])
#w_curve = np.array([-2/5 + (x-30)/75 for x in [30,60,90]] + [8/5 - (x-30)/75 for x in [120,150,180]])
#
## 1. For each (secid, date) calculating the value weighted
#def level_slope_curve_weight(sub_df):
#    sub_df = sub_df.sort_values(["m"])
#    x = np.array(sub_df[VARIABLE])
#    level = np.mean(x)
#    slope = np.inner(x, w_slope)
#    curve = np.inner(x, w_curve)
#    return (level, slope, curve)
#
#lsc_ind_date = int_d_var.groupby(["secid", "date"]).apply(level_slope_curve_weight).apply(pd.Series)
#lsc_ind_date.columns = ["level", "slope", "curve"]
#lsc_ind_date = lsc_ind_date.reset_index()
#
## 2. For each (secid, month) averaging slope-weighted and
## curve-weighted measures
#lsc_ind_date["date_mon"] = lsc_ind_date["date"] + pd.offsets.MonthEnd(0)
#lsc_ind_month = lsc_ind_date.groupby(["secid","date_mon"])["level", "slope", "curve"].mean()
#lsc_ind_month = lsc_ind_month.reset_index()
#
## 3. for each month do a truncated mean of all companies
#lsc_factors = lsc_ind_month.groupby("date_mon")["level", "slope", "curve"].apply(mean_with_truncation).apply(pd.Series)
#lsc_factors.columns = ["level", "slope", "curve"]
#lsc_factors.plot(figsize = (10,7))
#
#lsc_factors.diff().corr()
#
#smf.ols(formula = "curve ~ level", data = lsc_factors).fit().resid.plot()
#
#
#
#
#
#
#
#
