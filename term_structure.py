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

    
# Filter observations (secid, date) that have the full term 
# structure present
def filter_full_term_structure(int_d):
    VARIABLE = "D_clamp"
    int_d_var = int_d[["secid","date", "m", VARIABLE]]
    int_d_var = int_d_var[~int_d_var[VARIABLE].isnull()]
    
    # Pivoting table to only leave observations with full term structure present:
    pivot_mat = pd.pivot_table(int_d_var, index = ["secid", "date"], columns = "m", values = VARIABLE)
    pivot_mat = pivot_mat.dropna().reset_index()
    
    # Leaving only (secid, date) with the full term structure, got
    # these on the previous step
    int_d_var = pd.merge(pivot_mat[["secid", "date"]], int_d_var, 
                         on = ["secid", "date"], how = "left")
    
    return int_d_var

def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.99)) & (x >= np.quantile(x, 0.01))])

# Separately pivoting the data
def pivot_data_min_share(df, min_share_obs):
    total_number_of_months = len(np.unique(df["date_mon"]))
    secid_share_months = df[df.m == 30].groupby("secid")["date_mon"].count().sort_values(ascending = False)/total_number_of_months
    secid_list = list(secid_share_months.index[secid_share_months >= min_share_obs])
    df_pivot = pd.pivot_table(
        df[df.secid.isin(secid_list)], 
        index = "date_mon", columns = ["m", "secid"], values = VARIABLE)
    return df_pivot

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

def construct_lsc_factors(df):
    df_der = df.groupby(["secid", "date"]).apply(lsc_from_term_structure).apply(pd.Series)
    df_der.columns = ["level", "slope", "curve"]
    df_der = df_der.reset_index()
    
    df_der["date_mon"] = df_der["date"] + pd.offsets.MonthEnd(0)
    df_month_der = df_der.groupby(["secid","date_mon"])["level", "slope", "curve"].mean()
    df_month_der = df_month_der.reset_index()
    
    df_month_der = df_month_der.groupby("date_mon")["level", "slope", "curve"].apply(mean_with_truncation).apply(pd.Series)
    for i in range(3):
        df_month_der.iloc[:,i] = df_month_der.iloc[:,i]/np.std(df_month_der.iloc[:,i])
        
    return df_month_der
        
# This function in contrast constructs LSC factors based on average monthly 
# term structure for each (secid, month) as opposed to daily term structures.
def construct_lsc_factors_monthly(df):
    df_der = df.groupby(["secid", "date_mon"]).apply(lsc_from_term_structure).apply(pd.Series)
    df_der.columns = ["level", "slope", "curve"]
    df_der = df_der.reset_index()
    
    # Calculating average across secids for each month
    df_month_der = df_der.groupby("date_mon")["level", "slope", "curve"].apply(mean_with_truncation).apply(pd.Series)
    
    # Standardizing the factors:
    for i in range(3):
        df_month_der.iloc[:,i] = df_month_der.iloc[:,i]/np.std(df_month_der.iloc[:,i])
        
    return df_month_der
    
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


def construct_pcs(df_pivot, num_pcs = 3, method = "unbalanced"):
    '''
    Function that calculates principal components in differen ways.
      1. PC based on unbalanced panel where observations are
         reweighted to make the weights sum up to 1 at each date
      2. PC based on largely balanced panel, where missing
         variables are filled with the time series mean of a
         particular observation.
         
    Takes as inputs:
        
        df_pivot - 
             
        method - balanced vs. unbalanced method

        min_share_obs - minimum share of observations needed to
             be included in PCA
    '''  
    
    
    if method == "unbalanced":
        # Standardizing the data:
        for i_col in range(df_pivot.shape[1]):
            x = np.array(df_pivot.iloc[:,i_col])
            df_pivot.iloc[:, i_col] = (x - np.nanmean(x))/np.nanstd(x) 
            
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
            
        for i_col in range(df_pivot.shape[1]):
            x = np.array(df_pivot.iloc[:,i_col])
            df_pivot.iloc[:, i_col] = (x - np.nanmean(x))/np.nanstd(x) 
            
        corr_df = np.corrcoef(np.array(df_pivot).T)
        eigenvalue_decompos = eigs(np.array(corr_df), corr_df.shape[0] + 1)
        all_eigs = eigenvalue_decompos[0].astype(float)
        exp_var = all_eigs/np.sum(all_eigs)
        w = eigenvalue_decompos[1].astype(float)[:, range(num_pcs)]
        w = w/np.tile(np.sum(w, axis = 0).reshape((-1,num_pcs)), (w.shape[0], 1))
        pc = (np.array(df_pivot) @ w)
        
        pc_df = pd.DataFrame(pc, index = df_pivot.index, columns = ["PC" + str(i+1) for i in range(num_pcs)])
        w_df = pd.DataFrame(w, index = df_pivot.columns)
        w_df.columns = ["W" + str(x+1) for x in range(num_pcs)]

        return w_df, pc_df, exp_var[range(num_pcs)]
    
    else:
        raise ValueError("method can be 'unbalanced' or 'balanced_fill'")
                

################################################################
# Writing a function that takes in a data frame with observations
# normalizes them to have mean zero and unit standard deviation
# and runs a regression on factors supplied by the user. Before
# running the regression the function orthgonalizes the factors
def estimate_regs_on_factor(df, factors, daily = False):
    num_factors = factors.shape[1]
    
    # Normalizing the variables in the data frame:
    for i_col in range(df.shape[1]):
        x = np.array(df.iloc[:,i_col])
        df.iloc[:, i_col] = (x - np.nanmean(x))/np.nanstd(x) 
        
    # Estimating the regression of each series on the first principal
    # component. From that we can calculate residuals and calculate
    # the overall R^2 which will say what is the share of explained
    # variance by the first PC.
    merge_column = "date_mon"
    
    if daily:
        merge_column = "date"
    
    reg_df = pd.merge(df, factors, left_index = True, right_index = True) 
    
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
        
    # Calculating marginal improvement on R squared:
    R2_agg_list_to_return = []
    for i in range(num_factors):
        if i == 0:
            R2_agg_list_to_return.append(R2_agg_list[i])
        else:   
            R2_agg_list_to_return.append(R2_agg_list[i] - R2_agg_list[i-1])
            
    # Returing statistics on regressions in a dataframe:    
    res_df_to_return = pd.DataFrame({
            "m": [x[0] for x in index_list],
            "secid": [x[1] for x in index_list],
            "pc_num": [x[2] for x in index_list],
            "R2": [x.rsquared for x in results_list],
            "N": [x.nobs for x in results_list]})

    return R2_agg_list_to_return, res_df_to_return


def pc_weights_sum_stats(w, norm_factor_list = None):
    '''
    The input should be a dataframe with multiindex where the first level
    is maturity (e.g. 30 or 120) and second level is secid.
    
    Using normalization factor to improve readability (e.g. for weight
    the are sometimes on order of 0.0001)
    '''
    num_pcs = w.shape[1]
    w_stats_list = []
    
    
    # Looping through different principal component
    for i in range(num_pcs):
        col_list = list(w.index.levels[0])
        if norm_factor_list is None:
            norm_factor = 1
        else:
            norm_factor = norm_factor_list[i]
            
        w_load = pd.DataFrame(columns = col_list)
        # Looping through different maturities:
        for i_m, m in enumerate(w_df.index.levels[0]):
            w_load.iloc[:, i_m] = (w.iloc[:,i].loc[m]*norm_factor).describe().loc[["count", "mean", "std", "25%", "50%", "75%"]]
        
        w_stats_list.append(w_load)
        
    return w_stats_list

def generate_pc_weight_sum_stats_table(sum_stat_list, name_list, path):
    n_pcs = len(sum_stat_list)
    n_loads = sum_stat_list[0].shape[1]
    
    f = open(path, "w")
    f.write("\\begin{tabular}{l" + "r"*n_loads + "}\n")
    
    for i_pc in range(n_pcs):
        l = sum_stat_list[i_pc].round(3)
        l.columns = ["%d days" % x for x in l.columns]
        f.write("\\multicolumn{_n_loads_}{c}{_table_name_} \\\\ \n".replace("_n_loads_", str(n_loads + 1)).replace("_i_pc_", str(i_pc + 1)).replace("_table_name_", name_list[i_pc]))            
        f.write("\hline \\\\ \n")
        f.write(l.to_latex().replace("\\begin{tabular}{l" + "r"*n_loads + "}\n\\toprule\n{}", "").replace("\end{tabular}\n", ""))
    
    f.write("\\end{tabular}\n")
    f.close()

    
def generate_pc_sp_weight_table(table_list, table_name_list, exp_var_list, path):
    n_tables = len(table_list)
    n_loads = table_list[0].shape[1]
    
    f = open(path, "w")
    f.write("\\begin{tabular}{l" + "r"*(n_loads + 1) + "}\n")
    
    for i_table in range(n_tables):
        cols = list(table_list[i_table].columns)
        # Adding a column with explained variance:
        table_list[i_table]["exp_var"] = exp_var_list[i_table]
        # Rearranging columns
        table_list[i_table] = table_list[i_table][["exp_var"] + cols]
        l = table_list[i_table].round(3)
        l.columns = ["Expl. Var."] + ["%d days" % x for x in cols]
        f.write("\\multicolumn{_n_loads_}{c}{_table_name_} \\\\ \n".replace("_n_loads_", str(n_loads + 2)).replace("_table_name_", table_name_list[i_table]))
        f.write("\hline \\\\ \n")
        f.write(l.to_latex().replace("\\begin{tabular}{l" + "r"*(n_loads+1) + "}\n\\toprule\n{}", "").replace("\end{tabular}\n", ""))
    
    f.write("\\end{tabular}\n")
    f.close()

def main(argv = None):
    
    MIN_OBS_PER_MONTH = 5
    VARIABLE = "D_clamp"
    
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
    
    
    # Filtering observations to have the full term structure
    int_d_var = filter_full_term_structure(int_d)
    int_d_var["date"] = pd.to_datetime(int_d_var["date"])
    
    ############################################################################
    # Averaging within a month for each company and calculating number 
    # of days with observations. Filter by the number 
    # of observations in a month, e.g. if the secid has more than 5 
    # observations in a given month we include it in the
    # sample and use it for calculations later on
    ############################################################################
    int_d_var["date_mon"] = int_d_var["date"] + pd.offsets.MonthEnd(0)
    
    # Counting number of full observations for each (secid, month).
    mean_cnt_df = int_d_var.groupby(["secid", "date_mon"])["date"].count().reset_index()
    mean_cnt_df = mean_cnt_df.rename({"date": "cnt"}, axis = 1)
    
    # Filtering by the minimum number of observations:
    mean_cnt_df = mean_cnt_df[mean_cnt_df.cnt >= MIN_OBS_PER_MONTH]
    num_secids_per_month = mean_cnt_df.reset_index().groupby("date_mon")["secid"].count()
    
    # Merging this back to dataframe with individual disaster series:
    int_d_var["date"] = pd.to_datetime(int_d_var["date"])
    int_d_var["date_mon"] = int_d_var["date"] + pd.offsets.MonthEnd(0)
    int_d_var = pd.merge(mean_cnt_df, int_d_var, on = ["secid", "date_mon"], how = "left")
                            
    # Averaging within each (secid, month, maturity):
    int_d_mon_mat = int_d_var.groupby(["secid", "date_mon", "m"])[VARIABLE].mean().reset_index()

    # Saving the average monthly term structure:
    int_d_mon_mat.to_csv("estimated_data/interpolated_D/average_term_structure.csv", index = False)
    
    ############################################################################
    # Constructing unbalanced PCs
    ############################################################################
    df_pivot_unbalance = pivot_data_min_share(int_d_mon_mat, 0.5)
        
    w_unbalance, pc_unbalance = construct_pcs(
            df_pivot_unbalance, num_pcs = 3, method = "unbalanced")
    
    # Calculating time series regressions of secids on unbalanced PCs:
    R2_unbalance, reg_unbalance = estimate_regs_on_factor(df_pivot_unbalance, pc_unbalance)
    
    # Statistics on weights for unbalanced PCs:
    w_sum_stats_unbalance = pc_weights_sum_stats(w_unbalance, norm_factor_list = [1000,1000,1000])
    name_list = ["Summary Statistics for Loadings on PC_i_pc_, explains _exp_var_".replace("_i_pc_", str(i+1)).replace("_exp_var_", str(round(R2_unbalance[i],3))) for i in range(len(w_sum_stats_unbalance))]
    path = "estimated_data/term_structure/w_unbalance_sum_stats.tex"
    generate_pc_weight_sum_stats_table(w_sum_stats_unbalance, name_list, path)
    
    # Generating tables with summary statistics on R squared:
    tables_to_sum = pd.pivot_table(reg_unbalance[reg_unbalance["N"] >= 10*12].drop("N", axis = 1), columns = "pc_num", index = ["m", "secid"])
    r2_unbalance_sum_stats = pc_weights_sum_stats(tables_to_sum)
    name_list = ["Summary Statistics for R$^2$ on PC_i_pc_".replace("_i_pc_", str(i+1)) for i in range(len(r2_unbalance_sum_stats))]
    path = "estimated_data/term_structure/r2_unbalance_sum_stats.tex"
    generate_pc_weight_sum_stats_table(r2_unbalance_sum_stats, name_list, path)
    
    # Saving tables with regressions:
    reg_unbalance.to_csv("estimated_data/term_structure/reg_results_pc_unbalance.csv", index = False)
    pc_unbalance.to_csv("estimated_data/term_structure/pc_unbalanced.csv")
    
    ############################################################################
    # Constructing balance PCs
    ############################################################################
    df_pivot_balance = pivot_data_min_share(
            int_d_mon_mat[int_d_mon_mat.date_mon >= "2003-01-01"], 0.8)
    
    w_balance, pc_balance, exp_var_balanced = construct_pcs(
            df_pivot_balance, num_pcs = 3, method = "balanced_fill")
    
    # Calculating regressions of secids on unbalanced PCs:
    R2_balance, reg_balance = estimate_regs_on_factor(df_pivot_balance, pc_balance)
    
    # Statistics on weights for unbalanced PCs:
    w_sum_stats_balance = pc_weights_sum_stats(w_balance, norm_factor_list = [1000,1000,1000])
    name_list = ["Summary Statistics for Loadings on PC_i_pc_, explains _exp_var_".replace("_i_pc_", str(i+1)).replace("_exp_var_", str(round(R2_balance[i],3))) for i in range(len(w_sum_stats_unbalance))]
    path = "estimated_data/term_structure/w_balance_sum_stats.tex"
    generate_pc_weight_sum_stats_table(w_sum_stats_balance, name_list, path)
    
    # Generating tables with summary statistics on R squared:
    tables_to_sum = pd.pivot_table(reg_balance[reg_balance["N"] >= 10*12].drop("N", axis = 1), columns = "pc_num", index = ["m", "secid"])
    r2_balance_sum_stats = pc_weights_sum_stats(tables_to_sum)
    name_list = ["Summary Statistics for R$^2$ on PC_i_pc_".replace("_i_pc_", str(i+1)) for i in range(len(r2_balance_sum_stats))]
    path = "estimated_data/term_structure/r2_unbalance_sum_stats.tex"
    generate_pc_weight_sum_stats_table(r2_balance_sum_stats, name_list, path)
    
    reg_balance.to_csv("estimated_data/term_structure/reg_results_pc_balance_post_02.csv", index = False)
    pc_balance.to_csv("estimated_data/term_structure/pc_balanced.csv")
    
    ############################################################################
    # Constructing LSC factors using numerical derivatives:
    ############################################################################
    # Using the same pivoted data as for unbalanced PCs. Working with average
    # quantities within each month. This gives an average term structure for
    # each (secid, month). Calculating the level, slope and curvature factors
    # from this average quantities.
    
    # Melting the monthly df_pivot back to calculate LSC factors
    df_to_calc_lsc = pd.melt(df_pivot_unbalance.reset_index(), id_vars = "date_mon").dropna()
    df_to_calc_lsc = df_to_calc_lsc.rename({"value": VARIABLE}, axis = 1)
    
    lsc_factors = construct_lsc_factors_monthly(df_to_calc_lsc)
        
    # Calculating regressions of secids on LSC factors:
    R2_lsc, reg_lsc = estimate_regs_on_factor(df_pivot_unbalance, lsc_factors)
    
    # Generating tables with summary statistics on R squared:
    tables_to_sum = pd.pivot_table(reg_lsc[reg_lsc["N"] >= 10*12].drop("N", axis = 1), columns = "pc_num", index = ["m", "secid"])
    r2_lsc_sum_stats = pc_weights_sum_stats(tables_to_sum)
    name_list = ["Summary Statistics for R$^2$ on _name_ Factor".replace("_name_", x) for x in ["Level", "Slope", "Curvature"]]
    path = "estimated_data/term_structure/r2_lsc_sum_stats.tex"
    generate_pc_weight_sum_stats_table(r2_lsc_sum_stats, name_list, path)
    
    reg_lsc.to_csv("estimated_data/term_structure/reg_results_pc_lsc.csv", index = False)
    lsc_factors.to_csv("estimated_data/term_structure/lsc_factors.csv")
    
    
    ############################################################################
    # Can we get a principal component on a daily level:
    ############################################################################
    total_number_of_days = len(np.unique(int_d_var["date"]))
    secid_share_days = int_d_var[int_d_var.m == 30].groupby("secid")["date"].count().sort_values(ascending = False)/total_number_of_days
    
    secid_list = list(secid_share_days.index[secid_share_days >= 0.3])
    
    df_pivot_daily = pd.pivot_table(
        int_d_var[int_d_var.secid.isin(secid_list)], 
        index = "date", columns = ["m", "secid"], values = VARIABLE)
    
    w_unbalance_daily, pc_unbalance_daily = construct_pcs(
            df_pivot_daily, num_pcs = 3, method = "unbalanced")
    
    R2_unbalance_daily, reg_unbalance_daily = estimate_regs_on_factor(
            df_pivot_daily, pc_unbalance_daily, daily = True)
    
    # Summary of loadings on PCs
    w_sum_stats_unbalance_daily = pc_weights_sum_stats(w_unbalance_daily, norm_factor_list = [1000, 1000, 1000])
    name_list = ["Summary Statistics for Loadings on PC_i_pc_, explains _exp_var_".replace("_i_pc_", str(i+1)).replace("_exp_var_", str(round(R2_unbalance_daily[i],3))) for i in range(len(w_sum_stats_unbalance))]
    path = "estimated_data/term_structure/w_unbalance_daily_sum_stats.tex"
    generate_pc_weight_sum_stats_table(w_sum_stats_unbalance_daily, name_list, path)
    
    # Generating tables with summary statistics on R squared:
    tables_to_sum = pd.pivot_table(reg_unbalance_daily.drop("N", axis = 1), columns = "pc_num", index = ["m", "secid"])
    r2_unbalance_daily_sum_stats = pc_weights_sum_stats(tables_to_sum)
    name_list = ["Summary Statistics for R$^2$ on PC_i_pc_".replace("_i_pc_", str(i+1)) for i in range(len(r2_unbalance_daily_sum_stats))]
    path = "estimated_data/term_structure/r2_unbalance_daily_sum_stats.tex"
    generate_pc_weight_sum_stats_table(r2_unbalance_daily_sum_stats, name_list, path)
    
    reg_unbalance_daily.to_csv("estimated_data/term_structure/reg_results_unbalance_daily.csv", index = False)
    pc_unbalance_daily.to_csv("estimated_data/term_structure/pc_unbalanced_daily.csv")
    
    ####################################################################
    # Analyzing term structure of risk derived from S&P options
    ####################################################################
    int_d_sp_columns = ["secid", "date"] + [VARIABLE]
    int_d_sp = pd.DataFrame(columns = int_d_sp_columns)
    
    #for days in [30, 40, 60, 90, 100, 120, 150, 180]:
    for days in [30, 60, 90, 120, 150, 180]:
        print(days)
        # Loading data pre-1996 SPX options from CME
        int_d_sp_tmp = pd.read_csv("estimated_data/interpolated_D/int_D_spx_old_CME_days_" + str(days) + ".csv")
        int_d_sp_tmp = int_d_sp_tmp[int_d_sp_columns]
        int_d_sp_tmp["m"] = days
        int_d_sp = int_d_sp.append(int_d_sp_tmp)
        
        # Loading data post-1996 SPX options from OptionMetrics
        int_d_sp_tmp = pd.read_csv("estimated_data/interpolated_D/int_D_spx_days_" + str(days) + ".csv")
        int_d_sp_tmp = int_d_sp_tmp[int_d_sp_columns]
        int_d_sp_tmp["m"] = days
        int_d_sp = int_d_sp.append(int_d_sp_tmp)
    
    int_d_sp["date"] = pd.to_datetime(int_d_sp["date"])
    int_d_sp_var = filter_full_term_structure(int_d_sp)
    
    df_pivot_sp_daily = pd.pivot_table( 
        int_d_sp_var, index = "date", columns = ["m", "secid"], values = VARIABLE)
    
    w_sp_daily, pc_sp_daily, exp_var_sp_daily = construct_pcs(
            df_pivot_sp_daily, num_pcs = 3, method = "balanced_fill")
            
    # Estimating regressions of secids on PCs from SPX options:
    R2_daily_sp, reg_daily_sp = estimate_regs_on_factor(
            df_pivot_daily, pc_sp_daily, daily = True)
    
    # Generating tables with summary statistics on R squared:
    tables_to_sum = pd.pivot_table(reg_daily_sp.drop("N", axis = 1), columns = "pc_num", index = ["m", "secid"])
    r2_daily_sp_sum_stats = pc_weights_sum_stats(tables_to_sum)
    name_list = ["Summary Statistics for R$^2$ on PC_i_pc_".replace("_i_pc_", str(i+1)) for i in range(len(r2_daily_sp_sum_stats))]
    path = "estimated_data/term_structure/r2_daily_sp_sum_stats.tex"
    generate_pc_weight_sum_stats_table(r2_daily_sp_sum_stats, name_list, path)
    
    reg_daily_sp.to_csv("estimated_data/term_structure/reg_results_sp_daily.csv", index = False)
    pc_sp_daily.to_csv("estimated_data/term_structure/pc_sp_daily.csv")
    
    ####################################################################
    # Averaging SPX disaster measures to monthly level and doing the same
    # analysis:
    ####################################################################
    int_d_sp_var["date_mon"] = int_d_sp_var["date"] + pd.offsets.MonthEnd(0)
    int_d_sp_mon = int_d_sp_var.groupby(["secid","date_mon", "m"])[VARIABLE].mean().reset_index()
    
    df_pivot_sp_mon = pd.pivot_table(
        int_d_sp_mon, index = "date_mon", columns = ["m", "secid"], values = VARIABLE)
    
    w_sp_mon, pc_sp_mon, exp_var_sp_mon = construct_pcs(
            df_pivot_sp_mon, num_pcs = 3, method = "balanced_fill")
    
    R2_mon_sp, reg_mon_sp = estimate_regs_on_factor(
            df_pivot_unbalance, pc_sp_mon)
    
    # Generating tables with summary statistics on R squared:
    tables_to_sum = pd.pivot_table(reg_mon_sp[reg_mon_sp["N"] >= 12*10].drop("N", axis = 1), columns = "pc_num", index = ["m", "secid"])
    r2_mon_sp_sum_stats = pc_weights_sum_stats(tables_to_sum)
    name_list = ["Summary Statistics for R$^2$ on PC_i_pc_".replace("_i_pc_", str(i+1)) for i in range(len(r2_mon_sp_sum_stats))]
    path = "estimated_data/term_structure/r2_mon_sp_sum_stats.tex"
    generate_pc_weight_sum_stats_table(r2_mon_sp_sum_stats, name_list, path)
    
    reg_mon_sp.to_csv("estimated_data/term_structure/reg_results_sp_mon.csv", index = False)
    pc_sp_mon.to_csv("estimated_data/term_structure/pc_sp_mon.csv")
    
    # Making a table with PC loadings for daily and monhtly SPX disaster series:
    table_list = [x.reset_index().drop("secid", axis = 1).set_index("m") for x in [w_sp_mon, w_sp_daily]]
    table_list = [x.rename(columns = {0: "PC1", 1:"PC2", 2: "PC3"}).T for x in table_list]
    table_name_list = ["Monthly SPX Disaster Series", "Daily SPX Disaster Series"]
    exp_var_list = [exp_var_sp_mon, exp_var_sp_daily]
    path = "estimated_data/term_structure/w_pc_sp.tex" 
    generate_pc_sp_weight_table(table_list, table_name_list, exp_var_list, path)
    
    
    ################################################################
    # Constructing a short table with statistics on the distribution
    # of loadings
    ################################################################
    table_to_sum_list = [
            w_balance,
            w_unbalance,
            w_unbalance_daily]
    
    
    level_list = ["Individual"] + [""]*5 + ["SPX",""]
    type_list = ["Balanced","(post 02)"] + ["Unbalanced",""] + ["Unbalanced", "Daily"] + ["Monthly", "Daily"]
    stat_list = ["Mean","Median"]*3 + ["Actual"]*2
    
    path = "estimated_data/term_structure/w_all_sum_stats.tex"
    
    f = open(path, "w")
    f.write("\\begin{tabular}{lllcccccc}\n")
    #f.write("\\toprule \n")
    
    i_pc = 0
    
    for i_pc in range(3):
        sum_stats_agg_ind = [(x*1000).reset_index().groupby("m")["W"+str(i_pc+1)].describe().T.loc[["mean","50%"]] for x in table_to_sum_list]
        sum_stats_agg_ind = reduce(lambda df1, df2: df1.append(df2), sum_stats_agg_ind)
        sum_stats_agg = sum_stats_agg_ind.append(w_sp_mon.reset_index().set_index("m")["W"+str(i_pc+1)].T)
        sum_stats_agg = sum_stats_agg.append(w_sp_daily.reset_index().set_index("m")["W"+str(i_pc+1)].T)
        
        f.write("\multicolumn{9}{l}{\\textbf{Principal Component %d}} \\\\ \\\\[-1.8ex] \n" % (i_pc+1))
        f.write("\hline \\\\[-1.8ex] \n")
        f.write(" &  &  & 30 days & 60 days & 90 days & 120 days & 150 days & 180 days \\\\ \\\\[-1.8ex] \n")
        f.write("\hline \\\\[-1.8ex] \n")
        
        for i_row in range(sum_stats_agg.shape[0]):
            vars_to_write = [level_list[i_row], type_list[i_row], stat_list[i_row]]
            vars_to_write = vars_to_write + list(sum_stats_agg.iloc[i_row])
            
            if i_row in [1,3]:
                f.write("{} & {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
                f.write("\\cline{2-9} \\\\[-1.8ex] \n")
            elif i_row in [5,7]:
                f.write("{} & {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
                f.write("\hline \\\\[-1.8ex] \n")
            else:
                f.write("{} & {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\\n".format(*vars_to_write))
        
        f.write("\\\\[-0.8ex] \n")
    
    #f.write("\\bottomrule \n")
    f.write("\end{tabular} \n")     
    f.close()
    
    ################################################################
    # Constructing a short table with statistics on the distribution
    # of R squared without separation by maturity
    ################################################################
    table_list = [
            reg_balance,
            reg_unbalance,
            reg_lsc,
            reg_unbalance_daily,
            reg_mon_sp,
            reg_daily_sp]
    
    sum_stats_agg = [x.groupby("pc_num").R2.describe().T.loc[["mean", "50%"]] for x in table_list]
    sum_stats_agg = reduce(lambda df1, df2: df1.append(df2), sum_stats_agg)
    mult_index = [mult_index_level_1, mult_index_level_2, mult_index_level_3]
    mult_index = list(zip(*mult_index))
    mult_index = pd.MultiIndex.from_tuples(mult_index, names=["Level", "Type", "Stat"])
    sum_stats_agg.index = mult_index
    sum_stats_agg = sum_stats_agg*100
    
    level_list = ["Individual"] + [""]*7 + ["SPX"] + [""]*3
    type_list = ["Balanced","(post 02)"] + ["Unbalanced",""] + ["Unbalaced", "(Daily)"] + ["LSC",""] + ["Monthly",""] + ["Daily",""]
    stat_list = ["Mean", "Median"]*6
    
    # Converting to percent:
    
    path = "estimated_data/term_structure/r2_all_sum_stats.tex"
    f = open(path, "w")
    f.write("\\begin{tabular}{lllccc}\n")
    f.write("\\toprule \n")
    f.write(" &  &  & PC1 & PC2 & PC3 \\\\ \n")
    f.write(" &  &  & (Level) & (Slope) & (Curvature) \\\\ \\\\[-1.8ex] \n")
    f.write("\hline \\\\[-1.8ex] \n")
    
    for i_row in range(sum_stats_agg.shape[0]):
        vars_to_write = [level_list[i_row], type_list[i_row], stat_list[i_row]]
        vars_to_write = vars_to_write + list(sum_stats_agg.iloc[i_row])
        
        if i_row in [1,3,5,9]:
            f.write("{} & {} & {} & {:.1f} & {:.1f} & {:.1f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
            f.write("\\cline{2-6} \\\\[-1.8ex] \n")
        elif i_row == 7:
            f.write("{} & {} & {} & {:.1f} & {:.1f} & {:.1f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
            f.write("\hline \\\\[-1.8ex] \n")
        else:
            f.write("{} & {} & {} & {:.1f} & {:.1f} & {:.1f} \\\\\n".format(*vars_to_write))
    
    f.write("\\bottomrule \n")
    f.write("\end{tabular} \n")  
    f.close()
    


if __name__ == "__main__":
	sys.exit(main(sys.argv))





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
