"""
Script for merging disaster series with monthly returns data
"""


########################################################################
# Loading interpolated measures according to the specified number of days
# of interpolation
days = 30
min_obs_in_month = 15
min_share_month = 0.8
var = "D_clamp"

file_name = "estimated_data/interpolated_D/int_D_clamp_days_" + str(days) + ".csv"
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
oclink = pd.read_csv("estimated_data/crsp_data/optionmetrics_crsp_link.csv")

# Getting the best link for each month end of D-clamp:
oclink = oclink[oclink.score < 6]
oclink["sdate"] = [str(int(x)) for x in oclink["sdate"]]
oclink["sdate"] = pd.to_datetime(oclink["sdate"], format = "%Y%m%d")
oclink["edate"] = [str(int(x)) for x in oclink["edate"]]
oclink["edate"] = pd.to_datetime(oclink["edate"], format = "%Y%m%d")

pysqldf = lambda q: sqldf(q, globals())

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

tmp = pysqldf(q1)

# Filtering and providing the best match:
q2 = """
select 
    *,
    COALESCE(permno_1, permno_2, permno_3, permno_4, permno_5) as permno
from tmp
"""

disaster_ret_df = pysqldf(q2)
disaster_ret_df = disaster_ret_df.drop(
        ["permno_1", "permno_2", "permno_3", "permno_4", "permno_5"], axis = 1)

# Merging with returns next month:
disaster_ret_df = disaster_ret_df.rename({"date_adj": "date"}, axis = 1)
disaster_ret_df["date"] = pd.to_datetime(disaster_ret_df["date"])
disaster_ret_df["month_lead"] = disaster_ret_df["date"] + pd.offsets.MonthEnd(1)
disaster_ret_df = disaster_ret_df.drop("date", axis = 1)

ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)

disaster_ret_df = pd.merge(disaster_ret_df, ret_df[["date", "permno", "ret"]],
                             left_on = ["permno", "month_lead"], 
                             right_on = ["permno", "date"], how = "left")


ret = estimate_disaster_sort_strategy(disaster_ret_df, "D_clamp")

gross_ret = ret + 1
log_ret = np.log(gross_ret)
log_ret = log_ret/(np.std(log_ret))
log_ret.cumsum().plot()






################################################################

ret = estimate_disaster_sort_strategy(disaster_ret_30_df, "D_clamp")
gross_ret = ret + 1
log_ret = np.log(gross_ret)
log_ret = log_ret/(np.std(log_ret))
log_ret.cumsum().plot()

ret = estimate_disaster_sort_strategy(disaster_ret_30_filter_df, "D_clamp")
gross_ret = ret + 1
log_ret = np.log(gross_ret)
log_ret = log_ret/(np.std(log_ret))
log_ret.cumsum().plot()


tmp_df = merge_and_filter_ind_disaster(30, "D_clamp", 15, 0.8)
ret = estimate_disaster_sort_strategy(tmp_df, "D_clamp")

gross_ret = ret + 1
log_ret = np.log(gross_ret)
log_ret = log_ret/(np.std(log_ret))
log_ret.cumsum().plot()




