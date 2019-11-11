"""

"""

import seaborn as sns
from functools import reduce


####################################################################
#
####################################################################
disaster_factors = pd.read_csv("estimated_data/disaster_risk_measures/disaster_risk_measures.csv")
disaster_factors["date"] = pd.to_datetime(disaster_factors["date"])
level = disaster_factors[(disaster_factors.level == "Ind") &
                         (disaster_factors.maturity == "level") &
                         (disaster_factors.variable == "D_clamp")]

level = level[["date", "value"]].set_index("date")

ax = level.plot()
ax.set_xlabel("")
ax.get_legend().remove()
plt.tight_layout()
plt.savefig("SS_figures/level_factor.pdf")

####################################################################
# Figure where compare with other measures
####################################################################
# 1. VIX:
vix_archive = pd.read_csv("data/VIX/vixarchive.csv")
vix_archive = vix_archive[["Date", "VIX Close"]].rename(columns = {"Date":"date", "VIX Close":"VIX"})
vix_archive["date"]= pd.to_datetime(vix_archive["date"])

vix_current = pd.read_csv("data/VIX/vixcurrent.csv")
vix_current = vix_current[["Date", "VIX Close"]].rename(columns = {"Date":"date", "VIX Close":"VIX"})
vix_current["date"]= pd.to_datetime(vix_current["date"])

vix = vix_archive.append(vix_current)
vix.to_csv("data/VIX_full.csv", index = False)
vix["date_mon"] = vix["date"] + pd.offsets.MonthEnd(0)
vix_mon_mean = vix.groupby("date_mon").VIX.mean()

# 2. Shiller's P/D
shiller_pd = pd.read_csv("data/shiller_pd.csv")
shiller_pd["Date"] = pd.DatetimeIndex(start = "1871-01-01", freq = "M", periods = shiller_pd.shape[0])
shiller_pd["D/P"] = shiller_pd["D"]/shiller_pd["P"]
shiller_pd = shiller_pd[["Date", "D/P"]].set_index("Date")
plt.plot(shiller_pd)

# 3. Disaster measure derived from S&P options:
comb_disaster_df = pd.read_csv("estimated_data/disaster-risk-series/combined_disaster_df.csv")
sp_D = comb_disaster_df[(comb_disaster_df.level == "sp_500_OM") & (comb_disaster_df["var"] == "D_clamp")]
sp_D = sp_D.groupby("date").value.mean().rename("sp_500_D")

# 4. Replicating Cochrane and Piazzesi forward factor:
zcb_df = pd.read_csv("data/fama_bliss_zero_yields.csv")
zcb_df.dropna(inplace = True)
zcb_df.replace({"KYTREASNOX":{2000047:"p1", 2000048:"p2", 2000049:"p3", 2000050:"p4", 2000051:"p5"}}, inplace = True)
zcb_df.columns = ["m", "date", "price"]
zcb_df["date"] = pd.to_datetime(zcb_df["date"])
zcb_df = pd.pivot_table(zcb_df, index = "date", columns = "m", values = "price")
zcb_df = zcb_df/100

zcb_df["y1"] = 0 - np.log(zcb_df["p1"])    
zcb_df["f2"] = np.log(zcb_df["p1"]) - np.log(zcb_df["p2"])
zcb_df["f3"] = np.log(zcb_df["p2"]) - np.log(zcb_df["p3"])
zcb_df["f4"] = np.log(zcb_df["p3"]) - np.log(zcb_df["p4"])
zcb_df["f5"] = np.log(zcb_df["p4"]) - np.log(zcb_df["p5"])

zcb_df["p1_lead"] = zcb_df["p1"].shift(-12)
zcb_df["p2_lead"] = zcb_df["p2"].shift(-12)
zcb_df["p3_lead"] = zcb_df["p3"].shift(-12)
zcb_df["p4_lead"] = zcb_df["p4"].shift(-12)

zcb_df["r2"] = np.log(zcb_df["p1_lead"]) - np.log(zcb_df["p2"])
zcb_df["r3"] = np.log(zcb_df["p2_lead"]) - np.log(zcb_df["p3"])
zcb_df["r4"] = np.log(zcb_df["p3_lead"]) - np.log(zcb_df["p4"])
zcb_df["r5"] = np.log(zcb_df["p4_lead"]) - np.log(zcb_df["p5"])

zcb_df["rx2"] = zcb_df["r2"] - zcb_df["y1"]
zcb_df["rx3"] = zcb_df["r3"] - zcb_df["y1"]
zcb_df["rx4"] = zcb_df["r4"] - zcb_df["y1"]
zcb_df["rx5"] = zcb_df["r5"] - zcb_df["y1"]
zcb_df["rx_bar"] = 0.25*(zcb_df["rx2"] + zcb_df["rx3"] + zcb_df["rx4"] + zcb_df["rx5"])

smf.ols(formula = "rx_bar ~ y1 + f2 + f3 + f4 + f5", data = zcb_df[(zcb_df.index>="1964-01-01")&(zcb_df.index<="2003-12-31")]).fit().summary()

# Using Cochrane Piazzesi estimates to construct the factor
zcb_df["CP_factor_original"] = -(-3.24 - 2.14*zcb_df["y1"] + 0.81*zcb_df["f2"] + 3*zcb_df["f3"] + 0.8*zcb_df["f4"]-3.08*zcb_df["f5"])

# Reestimating CP regression using data through 2019:
#CP_new_params = smf.ols(formula = "rx_bar ~ y1 + f2 + f3 + f4 + f5", data = zcb_df[zcb_df.index>="1964-01-01"]).fit().params
#zcb_df["CP_factor_extended"] = -1*(CP_new_params[0] + np.sum(zcb_df[["y1", "f2", "f3", "f4", "f5"]]*np.tile(np.array(CP_new_params[1:6]), (zcb_df.shape[0],1)), axis = 1))

# subtracting the mean from CP factors:
zcb_df["CP_factor_original"] = zcb_df["CP_factor_original"] - np.mean(zcb_df["CP_factor_original"])
#zcb_df["CP_factor_extended"] = zcb_df["CP_factor_extended"] - np.mean(zcb_df["CP_factor_extended"])

# Calculating term premium as 10yr-2yr rate:
rate2 = pd.read_csv("data/DGS2.csv",na_values='.')
rate10 = pd.read_csv("data/DGS10.csv",na_values='.')

rate2["DATE"] = pd.to_datetime(rate2["DATE"])
rate2["date_mon"] = rate2["DATE"] + pd.offsets.MonthEnd(0)
rate2 = rate2.groupby("date_mon")["DGS2"].last()/100

rate10["DATE"] = pd.to_datetime(rate10["DATE"])
rate10["date_mon"] = rate10["DATE"] + pd.offsets.MonthEnd(0)
rate10 = rate10.groupby("date_mon")["DGS10"].last()/100
tp = pd.merge(rate2, rate10, left_index = True, right_index = True)
tp["TP"] = tp["DGS10"] - tp["DGS2"]

# Combining together:
all_df = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), 
                [level, sp_D, vix_mon_mean, shiller_pd, 
                 zcb_df["CP_factor_original"], tp["TP"] ])
all_df.columns = ["Level D", "S&P 500 D", "VIX", "Shiller's D/P", "CP", "Term Premium"]

for icol in range(len(all_df.columns)):
    all_df.iloc[:, icol] = all_df.iloc[:, icol]/np.std(all_df.iloc[:, icol])
    
ax = all_df[["Level D", "S&P 500 D", "VIX", "Shiller's D/P"]].plot(figsize = (6,4), alpha = 0.8)
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig("SS_figures/compare_D_to_fin_market_indicators_1.pdf")

ax = all_df[["Level D", "CP", "Term Premium"]].plot(figsize = (6,4), alpha = 0.8)
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig("SS_figures/compare_D_to_fin_market_indicators_2.pdf")

####################################################################
# Table with correlation in first differences between measures
####################################################################
corr_diff = all_df.diff().corr().round(3)
path = "SS_tables/corr_D_fin_market_indicators.tex"
f = open(path, "w")
f.write(corr_diff.to_latex(column_format = "lccccccc"))
f.close()

####################################################################
# Figure with zooming in on Dot-com bubble and daily data
####################################################################
int_d = pd.DataFrame(columns = ["secid", "date", "D_clamp"])
    
for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    int_d_to_append = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
    int_d_to_append = int_d_to_append[["secid", "date", "D_clamp", "rn_prob_20", "rn_prob_80"]]
    int_d_to_append["days"] = days
    int_d = int_d.append(int_d_to_append)

int_d["date"] = pd.to_datetime(int_d["date"])
int_d["date_mon"] = int_d["date"] + pd.offsets.MonthEnd(0)
int_d["date_week"] = int_d['date'] - int_d['date'].dt.weekday.astype('timedelta64[D]')

level_daily = int_d.dropna().groupby(["date"])["D_clamp"].apply(mean_with_truncation).rename("D_clamp")
level_factors_daily = pd.merge(level_daily, int_spx.groupby("date")["D_clamp"].mean(),
                               left_index = True, right_index = True)
level_factors_daily.columns = ["Individual", "SPX"]

# Daily disaster series:
dot_com_daily = level_factors_daily[
        (level_factors_daily.index >= "1998-01-01") &
        (level_factors_daily.index <= "2003-12-31")]
dot_com_daily.plot(alpha = 0.85)
ax.set_xlabel("")
plt.tight_layout()
plt.savefig("SS_figures/zoom_dot_com_daily.pdf")

great_recession_daily = level_factors_daily[
        (level_factors_daily.index >= "2007-01-01") &
        (level_factors_daily.index <= "2009-12-31")]
great_recession_daily.plot(alpha = 0.85)
ax.set_xlabel("")
plt.tight_layout()
plt.savefig("SS_figures/zoom_great_recession_daily.pdf")


####################################################################
# Table with statistics on portfolio returns
variable = "D_clamp"
maturity = "level"
level = "Ind"

# Getting info on raw returns and disaster measure
port_ret = pd.read_csv("estimated_data/disaster_sorts/port_sort_ret.csv").rename(columns = {"Unnamed: 0":"date"})
port_ret = port_ret[(port_ret.variable == variable) & (port_ret.maturity == maturity) & (port_ret.level == level)]
port_ret["date"] = pd.to_datetime(port_ret["date"])

# Adding NBER recession indicator:
port_ret["rec"] = np.where(
        (port_ret["date"] >= "2001-04-01") & (port_ret["date"] < "2001-12-01") |
        (port_ret["date"] >= "2008-01-01") & (port_ret["date"] < "2009-07-01"), 1, 0)

dis_measures = pd.read_csv("estimated_data/disaster_risk_measures/disaster_risk_measures.csv")
level = dis_measures[(dis_measures.level == level) & (dis_measures.maturity == maturity) & (dis_measures.variable == variable)]
level = level.set_index("date")["value"].rename("level")
level_diff = level.diff()

# Renaming portfolios to make 1st to be least exposed and 5th most exposed
port_ret.rename(columns = {"ew_1":"ew_5", "ew_2":"ew_4","ew_3":"ew_3","ew_4":"ew_2","ew_5":"ew_1"}, inplace = True)
port_ret.rename(columns = {"vw_1":"vw_5", "vw_2":"vw_4","vw_3":"vw_3","vw_4":"vw_2","vw_5":"vw_1"}, inplace = True)

# Subtracting the risk free rate from all portfolios:
FF = crsp_comp.load_FF()
RF = FF[["RF"]]
port_ret = pd.merge(port_ret, RF, left_on = "date", right_index = True, how = "left")
for port_name in ["ew_" + str(x +1) for x in range(5)] + ["vw_" + str(x +1) for x in range(5)]:
    port_ret.loc[:,port_name] = port_ret.loc[:,port_name] - port_ret.loc[:,"RF"]
port_ret.drop(columns = "RF", inplace = True)

port_ret["ew_diff"] = port_ret["ew_5"] - port_ret["ew_1"]
port_ret["vw_diff"] = port_ret["vw_5"] - port_ret["vw_1"]

# Constructing statistics:
colname_list = ["ew_" + str(x+1) for x in range(5)] + ["ew_diff"]
colname_list += ["vw_" + str(x+1) for x in range(5)] + ["vw_diff"]
sum_stat_dict = {}
for colname in colname_list:
    add_dict = {}
    port_ret_col = port_ret.set_index("date")[colname]

    add_dict["mean"] = np.mean(port_ret_col)*12
    add_dict["std"] = np.std(port_ret_col)*np.sqrt(12)    
    add_dict["sharpe"] = add_dict["mean"]/add_dict["std"]
    add_dict["N"] = port_ret_col.shape[0]
    
    port_ret_col = pd.merge(port_ret_col, level_diff, left_index = True, right_index = True)
    reg_res = smf.ols(formula = colname + " ~ level", data = port_ret_col).fit()
    add_dict["beta_level"] = reg_res.params[1]
    
    sum_stat_dict[colname] = add_dict
    
# Constructing the summary statistics table:
ew_names = ["ew_" + str(x+1) for x in range(5)] + ["ew_diff"]
vw_names = ["vw_" + str(x+1) for x in range(5)] + ["vw_diff"]

path = "SS_tables/port_sum_stats.tex"
f = open(path, "w")
f.write("\\begin{tabular}{lcccccc}\n")
f.write("\\toprule \n")
f.write(" Portfolio & 1 & 2 & 3 & 4 & 5 & 5-1 \\\\ \n")
f.write("\hline \\\\[-1.8ex] \n")
f.write("\multicolumn{7}{l}{\\textbf{Equal Weighted Portfolios}} \\\\ \n")
f.write("\hline \\\\[-1.8ex] \n")

rows_to_write = [
        ["Mean($r - r_f$)"] + [sum_stat_dict[x]["mean"] for x in ew_names],
        ["Std($r - r_f$)"] + [sum_stat_dict[x]["std"] for x in ew_names],
        ["Sharpe"] + [sum_stat_dict[x]["sharpe"] for x in ew_names],
        ["$\\beta_\\mathbb{D}$"] + [sum_stat_dict[x]["beta_level"] for x in ew_names]]

for row_to_write in rows_to_write:
    f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*row_to_write))

f.write("\hline \\\\[-1.8ex] \n")
f.write("\multicolumn{7}{l}{\\textbf{Value Weighted Portfolios}} \\\\ \n")
f.write("\hline \\\\[-1.8ex] \n")

rows_to_write = [
        ["Mean($r - r_f$)"] + [sum_stat_dict[x]["mean"] for x in vw_names],
        ["Std($r - r_f$)"] + [sum_stat_dict[x]["std"] for x in vw_names],
        ["Sharpe"] + [sum_stat_dict[x]["sharpe"] for x in vw_names],
        ["$\\beta_\\mathbb{D}$"] + [sum_stat_dict[x]["beta_level"] for x in vw_names]]

for row_to_write in rows_to_write:
    f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*row_to_write))
    
f.write("\\bottomrule \n")
f.write("\end{tabular}")  
f.close()


####################################################################
# Table with stats on Cremers port returns:

# Getting info on raw returns and disaster measure
port_ret = pd.read_csv("estimated_data/disaster_sorts/port_sort_cremers_jump_ret.csv").rename(columns = {"Unnamed: 0":"date"})
port_ret["date"] = pd.to_datetime(port_ret["date"])

cremers_df = pd.read_csv("data/cremers_factors.csv")
cremers_df["date"] = pd.to_datetime(cremers_df["date"])
cremers_df["date_mon"] = cremers_df["date"] + pd.offsets.MonthEnd(0)
cremers_df["JUMP"] = cremers_df["JUMP"] + 1
cremers_df = pd.DataFrame(cremers_df.groupby("date_mon")["JUMP"].prod())
cremers_df["JUMP"] = cremers_df["JUMP"] - 1

# Subtracting the risk free rate from all portfolios:
FF = crsp_comp.load_FF()
RF = FF[["RF"]]
port_ret = pd.merge(port_ret, RF, left_on = "date", right_index = True, how = "left")
for port_name in ["ew_" + str(x +1) for x in range(5)] + ["vw_" + str(x +1) for x in range(5)]:
    port_ret.loc[:,port_name] = port_ret.loc[:,port_name] - port_ret.loc[:,"RF"]
port_ret.drop(columns = "RF", inplace = True)

port_ret["ew_diff"] = port_ret["ew_5"] - port_ret["ew_1"]
port_ret["vw_diff"] = port_ret["vw_5"] - port_ret["vw_1"]

# Constructing statistics:
colname_list = ["ew_" + str(x+1) for x in range(5)] + ["ew_diff"]
colname_list += ["vw_" + str(x+1) for x in range(5)] + ["vw_diff"]
sum_stat_dict = {}
for colname in colname_list:
    add_dict = {}
    port_ret_col = port_ret.set_index("date")[colname]

    add_dict["mean"] = np.mean(port_ret_col)*12
    add_dict["std"] = np.std(port_ret_col)*np.sqrt(12)    
    add_dict["sharpe"] = add_dict["mean"]/add_dict["std"]
    add_dict["N"] = port_ret_col.shape[0]
    
    port_ret_col = pd.merge(port_ret_col, cremers_df, left_index = True, right_index = True)
    reg_res = smf.ols(formula = colname + " ~ JUMP", data = port_ret_col).fit()
    add_dict["beta_level"] = reg_res.params[1]
    
    sum_stat_dict[colname] = add_dict
    
# Constructing the summary statistics table:
ew_names = ["ew_" + str(x+1) for x in range(5)] + ["ew_diff"]
vw_names = ["vw_" + str(x+1) for x in range(5)] + ["vw_diff"]

path = "SS_tables/port_sum_stats_cremers.tex"
f = open(path, "w")
f.write("\\begin{tabular}{lcccccc}\n")
f.write("\\toprule \n")
f.write(" Portfolio & 1 & 2 & 3 & 4 & 5 & 5-1 \\\\ \n")
f.write("\hline \\\\[-1.8ex] \n")
f.write("\multicolumn{7}{l}{\\textbf{Equal Weighted Portfolios}} \\\\ \n")
f.write("\hline \\\\[-1.8ex] \n")

rows_to_write = [
        ["Mean($r - r_f$)"] + [sum_stat_dict[x]["mean"] for x in ew_names],
        ["Std($r - r_f$)"] + [sum_stat_dict[x]["std"] for x in ew_names],
        ["Sharpe"] + [sum_stat_dict[x]["sharpe"] for x in ew_names],
        ["$\\beta_\\mathbb{D}$"] + [sum_stat_dict[x]["beta_level"] for x in ew_names]]

for row_to_write in rows_to_write:
    f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*row_to_write))

f.write("\hline \\\\[-1.8ex] \n")
f.write("\multicolumn{7}{l}{\\textbf{Value Weighted Portfolios}} \\\\ \n")
f.write("\hline \\\\[-1.8ex] \n")

rows_to_write = [
        ["Mean($r - r_f$)"] + [sum_stat_dict[x]["mean"] for x in vw_names],
        ["Std($r - r_f$)"] + [sum_stat_dict[x]["std"] for x in vw_names],
        ["Sharpe"] + [sum_stat_dict[x]["sharpe"] for x in vw_names],
        ["$\\beta_\\mathbb{D}$"] + [sum_stat_dict[x]["beta_level"] for x in vw_names]]

for row_to_write in rows_to_write:
    f.write("{} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*row_to_write))
    
f.write("\\bottomrule \n")
f.write("\end{tabular}")  
f.close()



####################################################################
# Table with FF regressions of portfolios

reg_res_df = pd.read_csv("estimated_data/disaster_sorts/reg_results.csv")
reg_res_df = reg_res_df[(reg_res_df.variable == variable) & (reg_res_df.maturity == "level") & (reg_res_df.level == "Ind")]
reg_res_df = reg_res_df[reg_res_df.port.isin(["ew_1","ew_5","ew_diff", "vw_1","vw_5","vw_diff"])]
reg_res_df = reg_res_df[reg_res_df.FF.isin([1,3,5])]
reg_res_df = reg_res_df.sort_values(["FF", "port"])

path = "SS_tables/reg_ff.tex"
f = open(path, "w")
f.write("\small")
f.write("\\begin{tabular}{lccccccccc}\n")
f.write("\\toprule \n")
f.write("  & \\multicolumn{3}{c}{CAPM} & \\multicolumn{3}{c}{FF3} & \\multicolumn{3}{c}{FF5} \\\\ \n")
f.write("\cline{2-10} \n")
f.write(" Portfolio & 1 & 5 & 5-1 & 1 & 5 & 5-1 & 1 & 5 & 5-1 \\\\ \n")

i = 0
for var_list in [ew_names, vw_names]:
    if i == 0:
        f.write("\hline \\\\[-1.8ex] \n")
        f.write("\multicolumn{7}{l}{\\textbf{Equal Weighted Portfolios}} \\\\ \n")
        f.write("\hline \\\\[-1.8ex] \n")
        i+=1
    else:
        f.write("\hline \\\\[-1.8ex] \n")
        f.write("\multicolumn{7}{l}{\\textbf{Value Weighted Portfolios}} \\\\ \n")
        f.write("\hline \\\\[-1.8ex] \n")
        

    f.write("$\\alpha$ & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["alpha"])))
    f.write(" & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["alpha_se"])))
    
    f.write("$MKT$ & {:.3f}   & {:.3f}   & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_MKT"])))
    f.write("               & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_MKT_se"])))
    
    f.write("$SMB$ &   &  &  & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f}  \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_SMB"].dropna())))
    f.write(" &   &   &  &  ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] ".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_SMB_se"].dropna())))
    
    f.write("$HML$ &   &  &  & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \n ".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_HML"].dropna())))
    f.write("      &    &   &  &  ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_HML_se"].dropna())))
    
    f.write("$CMA$ &   &  &  & & & & {:.3f}  & {:.3f} & {:.3f}  \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_CMA"].dropna())))
    f.write("      &   &  &  & & & & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex]\n ".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_CMA_se"].dropna())))
    
    f.write("$RMW$ &   &  &  & & & & {:.3f}  & {:.3f} & {:.3f}  \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_RMW"].dropna())))
    f.write("      &   &  &  & & & & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_RMW_se"].dropna())))
    
    f.write("$R^2$ & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["R2"])))

f.write("\\bottomrule \n")
f.write("\end{tabular}")  
f.close()

####################################################################
# Regressions for Cremers factors

reg_res_df = pd.read_csv("estimated_data/disaster_sorts/reg_results_cremers.csv")
reg_res_df = reg_res_df[reg_res_df.port.isin(["ew_1","ew_5","ew_diff", "vw_1","vw_5","vw_diff"])]
reg_res_df = reg_res_df[reg_res_df.FF.isin([1,3,5])]
reg_res_df = reg_res_df.sort_values(["FF", "port"])

path = "SS_tables/reg_ff_cremers.tex"
f = open(path, "w")
f.write("\small")
f.write("\\begin{tabular}{lccccccccc}\n")
f.write("\\toprule \n")
f.write("  & \\multicolumn{3}{c}{CAPM} & \\multicolumn{3}{c}{FF3} & \\multicolumn{3}{c}{FF5} \\\\ \n")
f.write("\cline{2-10} \n")
f.write(" Portfolio & 1 & 5 & 5-1 & 1 & 5 & 5-1 & 1 & 5 & 5-1 \\\\ \n")

i = 0
for var_list in [ew_names, vw_names]:
    if i == 0:
        f.write("\hline \\\\[-1.8ex] \n")
        f.write("\multicolumn{7}{l}{\\textbf{Equal Weighted Portfolios}} \\\\ \n")
        f.write("\hline \\\\[-1.8ex] \n")
        i+=1
    else:
        f.write("\hline \\\\[-1.8ex] \n")
        f.write("\multicolumn{7}{l}{\\textbf{Value Weighted Portfolios}} \\\\ \n")
        f.write("\hline \\\\[-1.8ex] \n")
        
    f.write("$\\alpha$ & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["alpha"])))
    f.write(" & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["alpha_se"])))
    
    f.write("$MKT$ & {:.3f}   & {:.3f}   & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_MKT"])))
    f.write("               & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_MKT_se"])))
    
    f.write("$SMB$ &   &  &  & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f}  \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_SMB"].dropna())))
    f.write(" &   &   &  &  ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] ".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_SMB_se"].dropna())))
    
    f.write("$HML$ &   &  &  & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \n ".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_HML"].dropna())))
    f.write("      &    &   &  &  ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_HML_se"].dropna())))
    
    f.write("$CMA$ &   &  &  & & & & {:.3f}  & {:.3f} & {:.3f}  \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_CMA"].dropna())))
    f.write("      &   &  &  & & & & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex]\n ".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_CMA_se"].dropna())))
    
    f.write("$RMW$ &   &  &  & & & & {:.3f}  & {:.3f} & {:.3f}  \\\\ \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_RMW"].dropna())))
    f.write("      &   &  &  & & & & ({:.3f}) & ({:.3f}) & ({:.3f}) \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["beta_RMW_se"].dropna())))
    
    f.write("$R^2$ & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex] \n".format(*list(reg_res_df[reg_res_df.port.isin(var_list)]["R2"])))

f.write("\\bottomrule \n")
f.write("\end{tabular}")  
f.close()

################################################################
# Correlation of disaster measures with Cremers Factor




