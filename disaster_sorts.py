"""
This scripts performs company sort based on the value of an individual 
disaster measure. 
"""

import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from statsmodels.iolib.summary2 import summary_col # For summarizing regression results
import os
from matplotlib import pyplot as plt
from functools import reduce
from stargazer.stargazer import Stargazer

import crsp_comp

os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")



def load_and_filter_ind_disaster(days, min_obs_in_month, min_share_month):
    ########################################################################
    # Loading interpolated measures according to the specified number of days
    # of interpolation
    
    file_name = "estimated_data/interpolated_D/int_ind_disaster_days_" + str(days) + ".csv"
    D_df = pd.read_csv(file_name)
    
    # Dealing with dates:
    D_df["date"] = pd.to_datetime(D_df["date"])
    D_df["date_eom"] = D_df["date"] + pd.offsets.MonthEnd(0)
    D_df = D_df.drop("date", axis = 1)
    
    ########################################################################
    # Limiting to companies with at least 15 observations in a month in at least 80%
    # months in the sample from January 1996 to December 2017.
    def min_month_obs(x):
        return x["D_clamp"].count() > min_obs_in_month
    
    D_filter_1 = D_df.groupby(["secid", "date_eom"]).filter(min_month_obs)
    D_mon_mean = D_filter_1.groupby(["secid", "date_eom"]).mean().reset_index()
    
    num_months = len(np.unique(D_mon_mean["date_eom"]))
    def min_sample_obs(x):
        return x["D_clamp"].count() > num_months * min_share_month
    
    D_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)
    D_filter = D_filter.rename(columns = {"date_eom": "date"})
    
    return D_filter

def link_to_secid(df):
    """
    df should contain columns date and permno to get the match

    returns the same data frame with added column for OM secid
    """
    
    # Manually reading optionmetrics-crsp linking suite since there is
    # no dataset to download this from WRDS
    oclink = pd.read_csv("estimated_data/crsp_data/optionmetrics_crsp_link.csv")
    
    # Getting the best link for each month end
    oclink = oclink[oclink.score < 6]
    oclink["sdate"] = [str(int(x)) for x in oclink["sdate"]]
    oclink["sdate"] = pd.to_datetime(oclink["sdate"], format = "%Y%m%d")
    oclink["edate"] = [str(int(x)) for x in oclink["edate"]]
    oclink["edate"] = pd.to_datetime(oclink["edate"], format = "%Y%m%d")
    
    q1 = """
    select 
        d.*,
        s1.secid as secid_1, 
        s2.secid as secid_2,
        s3.secid as secid_3,
        s4.secid as secid_4,
        s5.secid as secid_5
    from df as d
    
    left join (select secid, permno, sdate, edate from oclink where score = 1) as s1
    on d.permno = s1.permno and d.date >= s1.sdate and d.date <= s1.edate
    
    left join (select secid, permno, sdate, edate from oclink where score = 2) as s2
    on d.permno = s2.permno and d.date >= s2.sdate and d.date <= s2.edate
    
    left join (select secid, permno, sdate, edate from oclink where score = 3) as s3
    on d.permno = s3.permno and d.date >= s3.sdate and d.date <= s3.edate
    
    left join (select secid, permno, sdate, edate from oclink where score = 4) as s4
    on d.permno = s4.permno and d.date >= s4.sdate and d.date <= s4.edate
    
    left join (select secid, permno, sdate, edate from oclink where score = 5) as s5
    on d.permno = s5.permno and d.date >= s5.sdate and d.date <= s5.edate
    """
    
    tmp = sqldf(q1, locals())
    
    # Filtering and providing the best match:
    q2 = """
    select 
        *, COALESCE(secid_1, secid_2, secid_3, secid_4, secid_5) as secid
    from tmp
    """
    
    df = sqldf(q2, locals())
    df = df.drop(columns = ["secid_1", "secid_2", "secid_3", "secid_4", "secid_5"])
    
    # Converting date columns to date format:
    df["date"] = pd.to_datetime(df["date"])
    
    return df

def portfolio_sorts_alt(db, crsp, df, sort_cols, ncuts, smoothing = 6):

    # Ensure CRSP data is at the end of each month for mergers later
    crsp.loc[:, 'date'] = crsp.loc[:, 'date'] + pd.offsets.MonthEnd(0)

    # == Lag characteristics for portfolio sorts so that returns in month m
    #    are based on characteristics in month m - 1 #
    crsp['form_date'] = crsp['date'] - pd.offsets.MonthEnd()

    # == Compute cuts == #
    psorts = df.groupby('date')[sort_cols].transform(\
                                        lambda q: pd.qcut(q, ncuts, 
                                        labels = np.arange(1, ncuts + 1)))

    
    # Merge back PERMNO-date information
    psorts = psorts.join(df[['date', 'secid']])

    # == Get value-weights  == #
    psorts = pd.merge(psorts, crsp[['date', 'secid', 'permno', 'permco_mktcap']],
                      left_on = ['date', 'secid'],
                      right_on = ['date', 'secid'],
                      how = 'inner')

    # Compute value-weights in each bin for each characteristic
    psorts = crsp_comp.compute_value_weights(psorts, sort_cols)

    # == Merge with returns and compute portfolio returns == #
    crsp_char = pd.merge(crsp, psorts, left_on = ['permno', 'form_date'],
                         right_on = ['permno', 'date'],
                         how = 'inner')

    crsp_char.rename(columns = {'date_x': 'date'}, inplace = True)
    crsp_char.drop(['date_y'], 1, inplace = True)
    
    
    # == Assemble portfolios == #
    portfolios = assemble_portfolios_alt(crsp_char, sort_cols, ncuts)
    
    return portfolios

def assemble_portfolios_alt(df, scols, ncuts):
    storage_dict = {}

    for char in scols:
        
        #Initialize dictionary entries
        storage_dict[char] = {}

        # Compute EW returns and counts, then join
        ew_ret = df.groupby(['date', char])['ret'].mean().reset_index().\
                    pivot(index = 'date', columns = char, values = 'ret')
        ew_count = df.groupby(['date']).\
                        apply(lambda x: len(x[[char,'ret']].dropna()))
        ew_count.name = 'ew_count'
        ew_ret = pd.DataFrame(ew_count).join(ew_ret)
        ew_ret.rename(columns = dict([(x, 'ew_' + str(x) ) for x in \
                                      np.arange(1, ncuts + 1)]),
                      inplace = True)

        # Compute VW returns and counts, then join
        df[char + '_vw_ret'] = df[char + '_vw'] * df['ret']
        vw_ret = df.groupby(['date', char])[char + '_vw_ret'].sum().\
                            reset_index().\
                            pivot(index = 'date', columns = char, 
                                  values = char + '_vw_ret') 
        vw_count = df.groupby(['date']).\
                        apply(lambda x: len(x[char + '_vw_ret'].dropna()))
        vw_count.name = 'vw_count'
        vw_ret = pd.DataFrame(vw_count).join(vw_ret)
        vw_ret.rename(columns = dict([(x, 'vw_' + str(x) ) for x in \
                                      np.arange(1, ncuts + 1)]),
                     inplace = True)
    
        # Merge equal-weighted and value-weighted return dataframes
        storage_dict[char]['ret'] = ew_ret.join(vw_ret)

        # Store portfolio constituents 
        storage_dict[char]['constituents'] = df[['form_date', 'permno', char]].dropna()

    return storage_dict

    
########################################################v
# Example of how sorting works
########################################################v
# Before starting, open the connection:
db = wrds.Connection()    

# First, loading CRSP monthly returns:
crsp_ret = crsp_comp.get_monthly_returns(db, "1996-01-01", "2017-12-31", balanced = True)
crsp_ret["date"] = pd.to_datetime(crsp_ret["date"]) + pd.offsets.MonthEnd(0)

# Second, using CRSP-OptionMetrics Linking suite to add secid to CRSP:
crsp_ret = link_to_secid(crsp_ret)

# Saving month by month CRSP-OptionMetrics link
crsp_ret[["permno", "date", "secid"]].to_csv("CRSP_OM_link_by_month.csv", index = False)


ports_all_days = {}
days_list = [30,60,90,120,150,180]
measure_list = ["D_clamp", "rn_prob_20", "rn_prob_40", "rn_prob_60", "rn_prob_80"]
for days in days_list:
    disaster_df = load_and_filter_ind_disaster(days, 5, 0)
    ports_all_days[days] = portfolio_sorts_alt(db, crsp_ret, disaster_df, measure_list, ncuts = 5, smoothing = 6)
    print("Completed %d days" % days)

# Combining return results in one table:
columns = ["days", "measure"]
columns = columns + ["ew_count"] + ["ew_"+str(i+1) for i in range(5)]
columns = columns + ["vw_count"] + ["vw_"+str(i+1) for i in range(5)]
ret_ind_sort_combined = pd.DataFrame(columns = columns)
for days in days_list:
    for measure in measure_list:
        to_append = ports_all_days[days][measure]["ret"]
        to_append["days"] = days
        to_append["measure"] = measure
        ret_ind_sort_combined = ret_ind_sort_combined.append(to_append)
        
# Summary statistics:
mean_comp_ew = ret_ind_sort_combined.groupby(["days","measure"])[["ew_1", "ew_5"]].mean()*12
mean_comp_vw = ret_ind_sort_combined.groupby(["days","measure"])[["vw_1", "vw_5"]].mean()*12

# Plotting the cumulative return for the most pronounced differences:

# 3. Loading FF portfolios to compare with disaster sort portfolio
ff_df = pd.read_csv("estimated_data/final_regression_dfs/ff_factors.csv")
ff_df["date"] = [str(x) + "01" for x in ff_df["date"]]
ff_df["date"] = pd.to_datetime(ff_df["date"], format = "%Y%m%d")
ff_df["date"] = ff_df["date"] + pd.offsets.MonthEnd(0)
for i in range(len(ff_df.columns) - 1):
    ff_df.iloc[:,i+1] = ff_df.iloc[:,i+1]/100
ff_df = ff_df.set_index("date")
ff_df = ff_df.rename({"Mkt-RF": "MKT"}, axis = 1)

# 5. Estimating regression of the return on each strategy on FF 5 factors:
res_list_ew_1 = []
res_list_ew_3 = []
res_list_ew_5 = []
res_list_vw_1 = []
res_list_vw_3 = []
res_list_vw_5 = []

for days in days_list:
    for measure in measure_list:
        reg_df = ports_all_days[days][measure]["ret"][["ew_5", "ew_1"]]
        reg_df["diff"] = reg_df["ew_5"] - reg_df["ew_1"]
        reg_df = pd.merge(reg_df, ff_df, left_index = True, right_index = True)
        res_list_ew_1.append(smf.ols(formula = "diff ~ MKT", data = reg_df*12).fit())
        res_list_ew_3.append(smf.ols(formula = "diff ~ MKT + SMB + HML", data = reg_df*12).fit())
        res_list_ew_5.append(smf.ols(formula = "diff ~ MKT + SMB + HML + CMA + RMW", data = reg_df*12).fit())
        
        reg_df = ports_all_days[days][measure]["ret"][["vw_5", "vw_1"]]
        reg_df["diff"] = reg_df["vw_5"] - reg_df["vw_1"]
        reg_df = pd.merge(reg_df, ff_df, left_index = True, right_index = True)
        res_list_vw_1.append(smf.ols(formula = "diff ~ MKT", data = reg_df*12).fit())
        res_list_vw_3.append(smf.ols(formula = "diff ~ MKT + SMB + HML", data = reg_df*12).fit())
        res_list_vw_5.append(smf.ols(formula = "diff ~ MKT + SMB + HML + CMA + RMW", data = reg_df*12).fit())





############################################################################
#
############################################################################

def merge_and_filter_ind_disaster(days, var, min_obs_in_month, min_share_month): 
    ########################################################################
    # Loading interpolated measures according to the specified number of days
    # of interpolation
    
    file_name = "estimated_data/interpolated_D/int_ind_disaster_days_" + str(days) + ".csv"
    D_df = pd.read_csv(file_name)
    
    # Dealing with dates:
    D_df["date"] = pd.to_datetime(D_df["date"])
    D_df["date_adj"] = D_df["date"] + pd.offsets.MonthEnd(0)
    D_df = D_df.drop("date", axis = 1)
    
    ########################################################################
    # Limiting to companies with at least 15 observations in a month in at least 80%
    # months in the sample from January 1996 to December 2017.
    def min_month_obs(x):
        return x[var].count() > min_obs_in_month
    
    D_filter_1 = D_df.groupby(["secid", "date_adj"]).filter(min_month_obs)
    D_mon_mean = D_filter_1.groupby(["secid", "date_adj"]).mean().reset_index()
    
    num_months = len(np.unique(D_mon_mean["date_adj"]))
    def min_sample_obs(x):
        return x[var].count() > num_months * min_share_month
    
    D_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)
    
    ########################################################################
    # Loading data on monthly return and linking data:
    ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_returns.csv")
    ret_df["MV"] = ret_df["prc"] * ret_df["shrout"]
    oclink = pd.read_csv("estimated_data/crsp_data/optionmetrics_crsp_link.csv")
    
    # Getting the best link for each month end of D-clamp:
    oclink = oclink[oclink.score < 6]
    oclink["sdate"] = [str(int(x)) for x in oclink["sdate"]]
    oclink["sdate"] = pd.to_datetime(oclink["sdate"], format = "%Y%m%d")
    oclink["edate"] = [str(int(x)) for x in oclink["edate"]]
    oclink["edate"] = pd.to_datetime(oclink["edate"], format = "%Y%m%d")
    
    q1 = """
    select 
        d.*,
        s1.permno as permno_1, 
        s2.permno as permno_2,
        s3.permno as permno_3,
        s4.permno as permno_4,
        s5.permno as permno_5
    from D_filter as d
    left join (
        select 
            secid, permno, sdate, edate
        from oclink
        where score = 1
    ) as s1
    on d.secid = s1.secid
    and d.date_adj >= s1.sdate
    and d.date_adj <= s1.edate
    left join (
        select 
            secid, permno, sdate, edate
        from oclink
        where score = 2
    ) as s2
    on d.secid = s2.secid
    and d.date_adj >= s2.sdate
    and d.date_adj <= s2.edate
    left join (
        select 
            secid, permno, sdate, edate
        from oclink
        where score = 3
    ) as s3
    on d.secid = s3.secid
    and d.date_adj >= s3.sdate
    and d.date_adj <= s3.edate
    left join (
        select 
            secid, permno, sdate, edate
        from oclink
        where score = 4
    ) as s4
    on d.secid = s4.secid
    and d.date_adj >= s4.sdate
    and d.date_adj <= s4.edate
    left join (
        select 
            secid, permno, sdate, edate
        from oclink
        where score = 5
    ) as s5
    on d.secid = s5.secid
    and d.date_adj >= s5.sdate
    and d.date_adj <= s5.edate
    """
    
    tmp = sqldf(q1, locals())
    
    # Filtering and providing the best match:
    q2 = """
    select 
        *,
        COALESCE(permno_1, permno_2, permno_3, permno_4, permno_5) as permno
    from tmp
    """
    
    disaster_ret_df = sqldf(q2, locals())
    disaster_ret_df = disaster_ret_df.drop(
            ["permno_1", "permno_2", "permno_3", "permno_4", "permno_5"], axis = 1)
    
    # Merging with returns next month:
    disaster_ret_df = disaster_ret_df.rename({"date_adj": "date"}, axis = 1)
    disaster_ret_df["date"] = pd.to_datetime(disaster_ret_df["date"])
    disaster_ret_df["month_lead"] = disaster_ret_df["date"] + pd.offsets.MonthEnd(1)
    disaster_ret_df = disaster_ret_df.drop("date", axis = 1)
    
    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)
    
    # Merging this month's disaster variable with next month's return on the stock
    disaster_ret_df = pd.merge(disaster_ret_df, ret_df[["date", "permno", "ret"]],
                                 left_on = ["permno", "month_lead"], 
                                 right_on = ["permno", "date"], how = "left")
    
    # Merging this month's disaster variable, next month's return on the stock
    # with this month's market value = |PRC|*SCHROUT
    disaster_ret_df = pd.merge(disaster_ret_df, ret_df[["date", "permno", "MV"]],
                               left_on = ["permno", "date"], 
                               right_on = ["permno", "date"], how = "left")

    return disaster_ret_df
    
# Function for calculating weighted average (to weight return by market value)
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

# Function to calculate sort based on a specified data frame and variable:    
def estimate_disaster_sort_strategy(disaster_ret_df, var_to_sort, value_weighted = True):
    
    # Calculate outliers that we are not going to invest in:
    quant_low_out = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.01).rename("quant_low_out")
    quant_high_out = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.99).rename("quant_high_out")
    
    quant_low = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.3).rename("quant_low")
    quant_high = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.7).rename("quant_high")
    
    # 2. Merging on original dataframe and assigning buckets:
    disaster_ret_quants_df = disaster_ret_df[["month_lead", var_to_sort, "ret", "MV"]]
    disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_low, on = "month_lead", how = "inner")
    disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_high, on = "month_lead", how = "inner")
    disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_low_out, on = "month_lead", how = "inner")
    disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_high_out, on = "month_lead", how = "inner")
    
    disaster_ret_quants_df["action"] = 0
    disaster_ret_quants_df.loc[(disaster_ret_quants_df[var_to_sort] < disaster_ret_quants_df["quant_low"]) & 
                               (disaster_ret_quants_df[var_to_sort] > disaster_ret_quants_df["quant_low_out"]),
               "action"] = -1
    disaster_ret_quants_df.loc[(disaster_ret_quants_df[var_to_sort] > disaster_ret_quants_df["quant_high"]) & 
                               (disaster_ret_quants_df[var_to_sort] < disaster_ret_quants_df["quant_high_out"]),
               "action"] = 1

    # Doing weighting:
    if value_weighted:
        ret_intermediate = disaster_ret_quants_df \
            .groupby(["month_lead", "action"]) \
            .apply(wavg, "ret", "MV") \
            .reset_index() \
            .rename({0: "ret"}, axis = 1)
        
        port_pivot = pd.pivot_table(
                ret_intermediate, values = "ret", index = ["month_lead"], 
                columns=["action"], aggfunc = np.sum)
        
        strategy_ret = port_pivot.iloc[:, 2] - port_pivot.iloc[:, 0]
        strategy_ret = strategy_ret.rename("strategy_ret")
        
    else:
        disaster_ret_quants_df["strategy_ret"] = disaster_ret_quants_df["ret"] * disaster_ret_quants_df["action"]
        strategy_ret = disaster_ret_quants_df.groupby("month_lead")["strategy_ret"].mean()
        
    
    return strategy_ret

################################################################################
# Merging and disaster series for individual companies with returns and 
# market value and calculating returns on a strategy based on disaster variable
# sorts
################################################################################
disaster_ret_30_df = merge_and_filter_ind_disaster(30, "D_clamp", 5, 0)
disaster_ret_60_df = merge_and_filter_ind_disaster(60, "D_clamp", 5, 0)
disaster_ret_120_df = merge_and_filter_ind_disaster(120, "D_clamp", 5, 0)

# Saving the merged returns-disaster risk data:
disaster_ret_30_df = disaster_ret_30_df.drop("date", axis = 1)
disaster_ret_60_df = disaster_ret_60_df.drop("date", axis = 1)
disaster_ret_120_df = disaster_ret_120_df.drop("date", axis = 1)

disaster_ret_30_df.to_csv("estimated_data/merged_disaster_ret_data/disaster_ret_30.csv", index = False)
disaster_ret_60_df.to_csv("estimated_data/merged_disaster_ret_data/disaster_ret_60.csv", index = False)
disaster_ret_120_df.to_csv("estimated_data/merged_disaster_ret_data/disaster_ret_120.csv", index = False)

# Calculating the returns on the trading strategy with value-weighted portfolios:
strategy_ret_30_D_clamp = estimate_disaster_sort_strategy(disaster_ret_30_df, "D_clamp")
strategy_ret_60_D_clamp = estimate_disaster_sort_strategy(disaster_ret_60_df, "D_clamp")
strategy_ret_120_D_clamp = estimate_disaster_sort_strategy(disaster_ret_120_df, "D_clamp")

strategy_ret_30_prob_20 = estimate_disaster_sort_strategy(disaster_ret_30_df, "rn_prob_20")
strategy_ret_60_prob_20 = estimate_disaster_sort_strategy(disaster_ret_60_df, "rn_prob_20")
strategy_ret_120_prob_20 = estimate_disaster_sort_strategy(disaster_ret_120_df, "rn_prob_20")

strategy_ret_30_prob_40 = estimate_disaster_sort_strategy(disaster_ret_30_df, "rn_prob_40")
strategy_ret_60_prob_40 = estimate_disaster_sort_strategy(disaster_ret_60_df, "rn_prob_40")
strategy_ret_120_prob_40 = estimate_disaster_sort_strategy(disaster_ret_120_df, "rn_prob_40")



strategy_ret_list = [strategy_ret_30_D_clamp, strategy_ret_60_D_clamp, strategy_ret_120_D_clamp,
         strategy_ret_30_prob_20, strategy_ret_60_prob_20, strategy_ret_120_prob_20,
         strategy_ret_30_prob_40, strategy_ret_60_prob_40, strategy_ret_120_prob_40]

strategy_name_list = ["D_30", "D_60", "D_120", 
                      "p_20_30", "p_20_60", "p_20_120",
                      "p_40_30", "p_40_60", "p_40_120"]

# Concatenating returns of each strategy and saving them in a single file:
strategy_ret_to_save = reduce(
        lambda a,b: pd.merge(a,b, left_index = True, right_index = True),
        strategy_ret_list)
strategy_ret_to_save.columns = strategy_name_list
strategy_ret_to_save.to_csv("estimated_data/final_regression_dfs/disaster_sort_ret.csv")

################################################################################
# Loading data on trading return of different sorts to do analysis
################################################################################
strategy_ret_df = pd.read_csv("estimated_data/final_regression_dfs/disaster_sort_ret.csv")
strategy_ret_df = strategy_ret_df.set_index("month_lead")

# 1. Plotting standardized cumulative log return of each sort type
log_ret = strategy_ret_df.copy()
for i in range(log_ret.shape[1]):
    log_ret.iloc[:,i] = np.log(1+log_ret.iloc[:,i])
    log_ret.iloc[:,i] = log_ret.iloc[:,i]/np.std(log_ret.iloc[:, i])
    
# Comparing different variables at 30 days interpolation:
log_ret.iloc[:,[0,3,6]].cumsum().plot(figsize = (7, 5))
plt.tight_layout()
plt.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/images/disaster_sort_vw_comp_1.pdf")    
    
# Comparing sort D for different interpolations:
log_ret.iloc[:,[0,1,2]].cumsum().plot(figsize = (7, 5))
plt.tight_layout()
plt.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/images/disaster_sort_vw_comp_2.pdf")    

# Comparing sort on prob-20 for different interpolations:
log_ret.iloc[:,[3,4,5]].cumsum().plot(figsize = (7, 5))
plt.tight_layout()
plt.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/images/disaster_sort_vw_comp_3.pdf")    

# Comparing sort on prob-40 for different interpolations:
log_ret.iloc[:,[6,7,8]].cumsum().plot(figsize = (7, 5))
plt.tight_layout()
plt.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/images/disaster_sort_vw_comp_4.pdf")    






# 2. Calculating correlations between returns:
strategy_ret_df.iloc[:,0:9].corr()

(strategy_ret_df.loc[:, "D_30"] + 1).cumprod().plot()
(ff_df[ff_df.index >= "1996-01-01"].loc[:, "MKT"] + 1).cumprod().plot()




# 3. Loading FF portfolios to compare with disaster sort portfolio
ff_df = pd.read_csv("estimated_data/final_regression_dfs/ff_factors.csv")
ff_df["date"] = [str(x) + "01" for x in ff_df["date"]]
ff_df["date"] = pd.to_datetime(ff_df["date"], format = "%Y%m%d")
ff_df["date"] = ff_df["date"] + pd.offsets.MonthEnd(0)
for i in range(len(ff_df.columns) - 1):
    ff_df.iloc[:,i+1] = ff_df.iloc[:,i+1]/100
ff_df = ff_df.set_index("date")
ff_df = ff_df.rename({"Mkt-RF": "MKT"}, axis = 1)

# Standardizing log returns to compare cumulative returns of FF portfolios and
# disaster sort portfolio
ff_to_comp = pd.merge(strategy_ret_df.iloc[:,[0,3]], ff_df, left_index = True, right_index = True)
ff_to_comp = ff_to_comp.drop("RF", axis = 1)
for i in range(ff_to_comp.shape[1]):
    ff_to_comp.iloc[:,i] = np.log(1 + ff_to_comp.iloc[:,i])
    ff_to_comp.iloc[:,i] = ff_to_comp.iloc[:,i]/np.std(ff_to_comp.iloc[:, i])

# Plotting and outputting the cumulative return comparison
ff_to_comp.cumsum().plot(figsize = (8, 6))
plt.tight_layout()
plt.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/images/disaster_sort_vw_compare_with_ff.pdf")

ff_to_comp.loc[:, ["D_30", "RMW"]].cumsum().plot(figsize = (8, 6))
plt.tight_layout()
plt.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/images/disaster_sort_vw_compare_with_rmw.pdf")

# 4. Correlation with FF portfolios
f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/disaster_sort_vw_corr_with_ff.tex", "w")
f.write(ff_to_comp.corr().round(3).to_latex())
f.close()


# 5. Estimating regression of the return on each strategy on FF 5 factors:
reg_df = strategy_ret_df.copy()
reg_df = pd.merge(reg_df, ff_df, left_index = True, right_index = True)

strategy_name_list = list(strategy_ret_df.columns)
results_list = []
for name in strategy_name_list:
    # to have the same name for all variables
    reg_df_tmp = reg_df.rename({name: "ret"}, axis = 1) 
    results_list.append(
            smf.ols(formula = "ret ~ MKT + SMB + HML + CMA + RMW", 
                    data = reg_df_tmp*12).fit())
    
# Outputting short regression results:
stargazer = Stargazer([results_list[0], results_list[3], results_list[6]])
stargazer.custom_columns(['D 30', 'prob 20', 'prob 40'], [1, 1,1])
stargazer.covariate_order(['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])
stargazer.show_degrees_of_freedom(False)
f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/disaster_sort_reg_on_ff.tex", "w")
f.write(stargazer.render_latex())
f.close()

# Doing extended regression table where I do regressions of strategy return on 
# (1) just the market, (2) FF 3 factors and (3) FF 5 factors.
results_list = []
for name in ["D_30", "p_20_30"]:
    # to have the same name for all variables
    reg_df_tmp = reg_df.rename({name: "ret"}, axis = 1) 
    results_list.append(
            smf.ols(formula = "ret ~ MKT", 
                    data = reg_df_tmp*12).fit())
    results_list.append(
            smf.ols(formula = "ret ~ MKT + SMB + HML", 
                    data = reg_df_tmp*12).fit())
    results_list.append(
            smf.ols(formula = "ret ~ MKT + SMB + HML + CMA + RMW", 
                    data = reg_df_tmp*12).fit())
    
# Outputting short regression results:
stargazer = Stargazer(results_list)
stargazer.custom_columns(["D_30"]*3 + ["prob 20"]*3, [1]*6)
stargazer.covariate_order(['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])
stargazer.show_degrees_of_freedom(False)
f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/disaster_sort_reg_on_ff_2.tex", "w")
f.write(stargazer.render_latex())
f.close()

####################################################################
# Generating merged file with disaster measure, return, disaster
# sorted portfolio assignment for each permno-month
####################################################################
disaster_ret_30_df = merge_and_filter_ind_disaster(30, "D_clamp", 15, 0)
disaster_ret_60_df = merge_and_filter_ind_disaster(60, "D_clamp", 15, 0)
disaster_ret_120_df = merge_and_filter_ind_disaster(120, "D_clamp", 15, 0)

# Analyzing Book-to-Market and Operating Profitability
bm_df = pd.read_csv("estimated_data/ind_disaster_sorts/port_sort_bm.csv")
bm_df.groupby(["variable", "days"])[["ew_" + str(x+1) for x in range(5)]].mean().T.plot()
bm_df.groupby(["variable", "days"])[["vw_" + str(x+1) for x in range(5)]].mean().T.plot()

op_df = pd.read_csv("estimated_data/ind_disaster_sorts/port_sort_op.csv")
op_df.groupby(["variable", "days"])[["ew_" + str(x+1) for x in range(5)]].mean().T.plot()
op_df.groupby(["variable", "days"])[["vw_" + str(x+1) for x in range(5)]].mean().T.plot()

# Analyzing returns:
df = pd.read_csv("estimated_data/ind_disaster_sorts/port_sort_ret.csv")
df.groupby(["variable", "days"])[["ew_" + str(x+1) for x in range(5)]].mean()
df.groupby(["variable", "days"])[["vw_" + str(x+1) for x in range(5)]].mean()




####################################################################
# Comparing equal vs. value weighted sorts
####################################################################

# First, plotting the figures side-by-side
disaster_ret_df = pd.read_csv("estimated_data/merged_disaster_ret_data/disaster_ret_30.csv")

port_return_ew = estimate_disaster_sort_strategy(disaster_ret_df, "rn_prob_20mon", value_weighted = False)
port_return_vw = estimate_disaster_sort_strategy(disaster_ret_df, "rn_prob_20mon", value_weighted = True)

port_return_ew_vw_comp = pd.merge(
        port_return_ew.rename("EW"), port_return_vw.rename("VW"), 
        left_index = True, right_index = True)
(port_return_ew_vw_comp+1).cumprod().plot()

# Next, looking at companies in more details
var_to_sort = "rn_prob_20mon"

quant_low_out = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.01).rename("quant_low_out")
quant_high_out = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.99).rename("quant_high_out")

quant_low = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.3).rename("quant_low")
quant_high = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.7).rename("quant_high")

# Plotting quantiles:
quants_compare = pd.merge(
        quant_low, quant_high, left_index = True, right_index = True)
quants_compare.plot()


# 2. Merging on original dataframe and assigning buckets:
disaster_ret_quants_df = disaster_ret_df[["month_lead", var_to_sort, "ret", "MV"]]
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_low, on = "month_lead", how = "inner")
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_high, on = "month_lead", how = "inner")
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_low_out, on = "month_lead", how = "inner")
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_high_out, on = "month_lead", how = "inner")

disaster_ret_quants_df["action"] = 0
disaster_ret_quants_df.loc[(disaster_ret_quants_df[var_to_sort] < disaster_ret_quants_df["quant_low"]) & 
                           (disaster_ret_quants_df[var_to_sort] > disaster_ret_quants_df["quant_low_out"]),
           "action"] = -1
disaster_ret_quants_df.loc[(disaster_ret_quants_df[var_to_sort] > disaster_ret_quants_df["quant_high"]) & 
                           (disaster_ret_quants_df[var_to_sort] < disaster_ret_quants_df["quant_high_out"]),
           "action"] = 1

# Plotting the distribution of market value for long and short companies
# at different points in time:
month_to_lot = "2010-01-31"
df_to_plot = disaster_ret_quants_df[
        disaster_ret_quants_df["month_lead"] == month_to_lot]

plt.hist(df_to_plot[df_to_plot["action"] == 1]["MV"])







# Doing weighting:
if value_weighted:
    ret_intermediate = disaster_ret_quants_df \
        .groupby(["month_lead", "action"]) \
        .apply(wavg, "ret", "MV") \
        .reset_index() \
        .rename({0: "ret"}, axis = 1)
    
    port_pivot = pd.pivot_table(
            ret_intermediate, values = "ret", index = ["month_lead"], 
            columns=["action"], aggfunc = np.sum)
    
    strategy_ret = port_pivot.iloc[:, 2] - port_pivot.iloc[:, 0]
    strategy_ret = strategy_ret.rename("strategy_ret")
    
else:
    disaster_ret_quants_df["strategy_ret"] = disaster_ret_quants_df["ret"] * disaster_ret_quants_df["action"]
    strategy_ret = disaster_ret_quants_df.groupby("month_lead")["strategy_ret"].mean()




