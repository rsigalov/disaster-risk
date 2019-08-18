"""
Comparing disaster sorted portfolios with 5x5 BM and OP FF sorted portfolios
"""


from functools import reduce

# Loading info on disaster portfolios:
sort_ind = pd.read_csv("estimated_data/disaster_sorts/port_sort_ret.csv")
sort_ind.rename(columns = {sort_ind.columns[0]:"date"}, inplace = True)
sort_ind["date"] = pd.to_datetime(sort_ind["date"])
sort_ind["ew_diff_ind"] = sort_ind["ew_5"] - sort_ind["ew_1"]
sort_ind["vw_diff_ind"] = sort_ind["vw_5"] - sort_ind["vw_1"]

sort_agg = pd.read_csv("estimated_data/disaster_sorts/port_sort_agg_ret.csv")
sort_agg.rename(columns = {sort_agg.columns[0]:"date"}, inplace = True)
sort_agg["date"] = pd.to_datetime(sort_agg["date"])
sort_agg["ew_diff_agg"] = sort_agg["ew_1"] - sort_agg["ew_5"]
sort_agg["vw_diff_agg"] = sort_agg["vw_1"] - sort_agg["vw_5"]

# Reading 25 portfolios sorted on BM and OP:
bm_op_vw = pd.read_csv("bm_op_vw.csv")
bm_op_vw["date"] = [str(x) + "01" for x in bm_op_vw["date"]]
bm_op_vw["date"] = pd.to_datetime(bm_op_vw["date"], format = "%Y%m%d") + pd.offsets.MonthEnd(0)
bm_op_ew = pd.read_csv("bm_op_ew.csv")
bm_op_ew["date"] = [str(x) + "01" for x in bm_op_ew["date"]]
bm_op_ew["date"] = pd.to_datetime(bm_op_ew["date"], format = "%Y%m%d") + pd.offsets.MonthEnd(0)

# Merging datasets:
sort_ind_sub = sort_ind[(sort_ind.variable == "D_clamp") & (sort_ind.days == -99)]
sort_agg_sub = sort_agg[sort_agg.variable == "level_factor"]
comp_ew = reduce(lambda df1, df2: pd.merge(df1,df2,on="date"),
                 [sort_ind_sub[["date", "ew_diff_ind", "vw_diff_ind"]],
                  sort_agg_sub[["date", "ew_diff_agg", "vw_diff_agg"]],
                  bm_op_ew])
comp_vw = reduce(lambda df1, df2: pd.merge(df1,df2,on="date"),
                 [sort_ind_sub[["date", "ew_diff_ind", "vw_diff_ind"]],
                  sort_agg_sub[["date", "ew_diff_agg", "vw_diff_agg"]],
                  bm_op_vw])

res = [comp_ew.corr().loc["ew_diff_ind", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values(),
       comp_ew.corr().loc["vw_diff_ind", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values(),
       comp_ew.corr().loc["ew_diff_agg", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values(),
       comp_ew.corr().loc["vw_diff_agg", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values(),
       comp_vw.corr().loc["ew_diff_ind", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values(),
       comp_vw.corr().loc["vw_diff_ind", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values(),
       comp_vw.corr().loc["ew_diff_agg", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values(),
       comp_vw.corr().loc["vw_diff_agg", ["HiBM HiOP", "LoBM HiOP", "HiBM LoOP", "LoBM LoOP"]].sort_values()]

reduce(lambda df1, df2: pd.merge(df1,df2,left_index=True,right_index=True), res).T


comp_ew.corr().loc["vw_diff_ind"].sort_values()
