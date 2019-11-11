#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:16:08 2019

@author: rsigalov
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")


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
    return np.mean(x[(x <= np.nanquantile(x, 0.975)) & (x >= np.nanquantile(x, 0.025))])

# First calculating by-maturity disaster measure on individual and aggregate
# levels:
MIN_OBS_PER_MONTH = 5
VARIABLE = "D_clamp"

################################################################
# Loading data on different maturity variable
int_d_columns = ["secid", "date"] + [VARIABLE]
int_d = pd.DataFrame(columns = int_d_columns)

for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    int_d_tmp = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
    int_d_tmp = int_d_tmp[int_d_columns]
    int_d_tmp["m"] = days
    int_d_tmp = int_d_tmp[(int_d_tmp.date >= "2006-01-01") & (int_d_tmp.date <= "2013-01-01")]
    int_d = int_d.append(int_d_tmp)

int_d["date"] = pd.to_datetime(int_d["date"])
int_d_var = filter_full_term_structure(int_d)
int_d_var["date"] = pd.to_datetime(int_d_var["date"])

# Averaging values for each date. Both by maturity averages and overall
# averages
ind_mean = int_d.groupby("date")[VARIABLE].apply(mean_with_truncation).rename("D").reset_index()
ind_mean_mat = int_d.groupby(["date", "m"])[VARIABLE].apply(mean_with_truncation).rename("D").reset_index()
#ind_mean = int_d.groupby("date")[VARIABLE].mean().rename("D").reset_index()
#ind_mean_mat = int_d.groupby(["date", "m"])[VARIABLE].mean().rename("D").reset_index()
ind_mean["date"] = pd.to_datetime(ind_mean["date"])
ind_mean_mat["date"] = pd.to_datetime(ind_mean_mat["date"])


#int_d_var[int_d_var.date == "2007-04-16"].D_clamp.mean()
#int_d_var[int_d_var.date == "2007-04-16"].groupby("m")["D_clamp"].mean()
#
#
#mean_with_truncation(int_d_var[int_d_var.date == "2007-04-16"]["D_clamp"])
#
#int_d_var[int_d_var.date == "2007-04-16"].groupby("m")["D_clamp"].apply(mean_with_truncation)
#int_d_var[(int_d_var.date == "2007-04-16") & (int_d_var.m == 150)]["D_clamp"].plot.hist()
#
#


# Loading disaster measure derived from SPX options:
spx = pd.read_csv("estimated_data/disaster-risk-series/spx_daily_disaster.csv")
spx["date"] = pd.to_datetime(spx["date"])
spx = spx[["date", "days", "D_clamp"]]
spx.sort_values(["date","days"], inplace = True)
spx_av = spx.groupby("date")["D_clamp"].mean().reset_index()
spx_av.sort_values("date", inplace = True)

# Pick a date and construct a plot that shows the current term structure and average
# disaster measure, previous days quantities, week ago quantities and month ago
# quantities

# Loading data on crisis event descriptions:
timeline = pd.read_csv("crisis_timeline/crisis_timeline_processed.csv")
timeline["date"] = pd.to_datetime(timeline["date"])
date_list = timeline["date"]

date_figures = []
notes_figures = []

# For all dates in the time line, find the next available date in the
# disaster series
timeline["date_next"] = None
for i_row in range(timeline.shape[0]):
    timeline.iloc[i_row, 2] = D[D.date >= timeline["date"].iloc[i_row]].date.iloc[0]
timeline["date_next"] = pd.to_datetime(timeline["date_next"])

# If for a particular event the next date with available disaster information 
# does not equal the even date, we need to cumulate all the news:
desc_cumulative = []
date_cumulative = []
for date in np.unique(timeline["date_next"]):
    date_cumulative.append(date)
    timeline_sub = timeline[timeline.date_next == date]
    desc_list_cur = list(timeline_sub["desc"])
    date_list_cur = list(timeline_sub["date"])
    if len(desc_list_cur) == 1:
        to_append = date_list_cur[0].strftime("%Y-%m-%d") + ": " + desc_list_cur[0]
        desc_cumulative.append(to_append)
    else:
        to_append = ""
        for j in range(len(desc_list_cur)):
            to_append += date_list_cur[0].strftime("%Y-%m-%d") + ": " + desc_list_cur[j] + "\n\n "
        desc_cumulative.append(to_append)

desc_cumulative_df = pd.DataFrame({"date": date_cumulative,
                                   "desc": desc_cumulative})
    

generate_figure(desc_cumulative_df["date"].iloc[90])
    
# Saving figures:
for date in desc_cumulative_df["date"]:
    generate_figure(date)
    
    
#generate_figure(desc_cumulative_df["date"].iloc[0])
## Generating latex script:
#f = open("crisis_timeline/crisis_timeline.tex", "w")
#
#for i, date in enumerate(desc_cumulative_df["date"]):
#    desc_cur = desc_cumulative_df[desc_cumulative_df.date == date].desc.iloc[0]
#    desc_cur = desc_cur.replace("%","\%")
#    desc_cur = desc_cur.replace("$","USD")
#    desc_cur = desc_cur.replace("€","EUR")
#    desc_cur = desc_cur.replace("£","GBP")
#    desc_cur = desc_cur.replace("¥","JPY")
#    desc_cur = desc_cur.replace("&","\&")
#    f.write("\\begin{figure}[htbp!] \n")
#    f.write("\centering \n")
#    f.write("\includegraphics[width = 0.8\\textwidth]{images/disaster-" + date.strftime("%Y-%m-%d") + ".png} \n")
#    f.write("\caption{Disaster risk measure on " + date.strftime("%Y-%m-%d") + "} \n")
#    f.write("\\begin{tablenotes} \n")
#    f.write("\small \n")
#    f.write("\item " + desc_cur + "\n")
#    f.write("\end{tablenotes} \n")
#    f.write("\end{figure} \n")
#    f.write("\n")
#    if ((i+1) % 3 == 0) & (i != 0):
#        f.write("\\newpage \n")
#        f.write("\n")
#    
#f.close()



    
    
def generate_figure(date):
    date_prev_day = ind_mean.date[ind_mean.date < date].iloc[-1]
    date_week_ago = ind_mean.date[ind_mean.date < date - pd.offsets.DateOffset(7)].iloc[-1]
    date_month_ago = ind_mean.date[ind_mean.date < date - pd.offsets.DateOffset(30)].iloc[-1]
    
    D_today_ts = np.array(ind_mean_mat[ind_mean_mat.date == date].D)
    D_today_av = ind_mean[ind_mean.date == date].D.iloc[0]
    
    D_prev_day_ts = np.array(ind_mean_mat[ind_mean_mat.date == date_prev_day].D)
    D_prev_day_av = ind_mean[ind_mean.date == date_prev_day].D.iloc[0]
    
    D_prev_week_ts = np.array(ind_mean_mat[ind_mean_mat.date == date_week_ago].D)
    D_prev_week_av = ind_mean[ind_mean.date == date_week_ago].D.iloc[0]
    
    D_prev_month_ts = np.array(ind_mean_mat[ind_mean_mat.date == date_month_ago].D)
    D_prev_month_av = ind_mean[ind_mean.date == date_week_ago].D.iloc[0]
    
    m_list = [30,60,90,120,150,180]
    av_list = [D_today_av, D_prev_day_av, D_prev_week_av, D_prev_month_av]
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize = (10,4))
    
    ax1.plot(m_list, D_today_ts, marker="s", color = "0", linewidth = 3, label = "Today: " + date.strftime("%Y-%m-%d"))
    ax1.plot(m_list, D_prev_day_ts, marker="o", color = "0.4", label = "Prev. Day: " + date_prev_day.strftime("%Y-%m-%d"))
    ax1.plot(m_list, D_prev_week_ts, linestyle = "-", marker="o", color = "0.75", label = "Prev. Week: " + date_week_ago.strftime("%Y-%m-%d"))
    ax1.plot(m_list, D_prev_month_ts, linestyle = "--", marker="o", color = "0.75", label = "Prev. Month: " + date_month_ago.strftime("%Y-%m-%d"))
    #ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax1.set_xticks(m_list)
    
    ax1.plot([195, 225], [av_list[0]]*2, color = "0", linewidth = 3)
    ax1.plot([195, 225], [av_list[1]]*2, color = "0.4")
    ax1.plot([195, 225], [av_list[2]]*2, linestyle = "-", color = "0.75")
    ax1.plot([195, 225], [av_list[3]]*2, linestyle = "--", color = "0.75")
    ax1.set_ylim((0, 0.25))
    
    # Doing SPX now:
    D_today_ts = np.array(spx[spx.date == date].D_clamp)
    D_today_av = spx_av[spx_av.date == date].D_clamp.iloc[0]
    
    D_prev_day_ts = np.array(spx[spx.date == date_prev_day].D_clamp)
    D_prev_day_av = spx_av[spx_av.date == date_prev_day].D_clamp.iloc[0]
    
    D_prev_week_ts = np.array(spx[spx.date == date_week_ago].D_clamp)
    D_prev_week_av = spx_av[spx_av.date == date_week_ago].D_clamp.iloc[0]
    
    D_prev_month_ts = np.array(spx[spx.date == date_month_ago].D_clamp)
    D_prev_month_av = spx_av[spx_av.date == date_month_ago].D_clamp.iloc[0]
    
    m_list = [30,60,90,120,150,180]
    av_list = [D_today_av, D_prev_day_av, D_prev_week_av, D_prev_month_av]
    
    ax2.plot(m_list, D_today_ts, marker="s", color = "0", linewidth = 3, label = "Today: " + date.strftime("%Y-%m-%d"))
    ax2.plot(m_list, D_prev_day_ts, marker="o", color = "0.4", label = "Prev. Day: " + date_prev_day.strftime("%Y-%m-%d"))
    ax2.plot(m_list, D_prev_week_ts, linestyle = "-", marker="o", color = "0.75", label = "Previous: " + date_week_ago.strftime("%Y-%m-%d"))
    ax2.plot(m_list, D_prev_month_ts, linestyle = "--", marker="o", color = "0.75", label = "Previous: " + date_month_ago.strftime("%Y-%m-%d"))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax2.set_xticks(m_list)
    
    ax2.plot([195, 225], [av_list[0]]*2, color = "0", linewidth = 3)
    ax2.plot([195, 225], [av_list[1]]*2, color = "0.4")
    ax2.plot([195, 225], [av_list[2]]*2, linestyle = "-", color = "0.75")
    ax2.plot([195, 225], [av_list[3]]*2, linestyle = "--", color = "0.75")
    ax2.set_ylim((0, 0.1))
    
    f.savefig("crisis_timeline/images/disaster-" +date.strftime("%Y-%m-%d") +".png", bbox_inches="tight")
    

d_diff = pd.pivot_table(D, columns = "m", index = "date").diff()

d_diff.iloc[:,0].plot()

pd.pivot_table(D, columns = "m", index = "date").iloc[:,0].plot()

pd.pivot_table(spx[spx.date >= "2007-01-01"], columns = "days", index = "date").plot()

################################################################
# Looking at the largest movements and looking for news at that 
# day
# Start with SPX, there is potentially less noise.
spx = spx[(spx.date >= "2007-01-01") & (spx.date < "2013-01-01")]
spx_av = spx_av[(spx_av.date >= "2007-01-01") & (spx_av.date < "2013-01-01")]

# Calculate summary statistics for differences
spx_av_diff = spx_av.set_index("date").diff()
diff_stats = spx_av_diff.describe(percentiles = [0.05, 0.95])
std_spx_av = diff_stats.loc["std","D_clamp"]
perc_95_spx_av = diff_stats.loc["95%","D_clamp"]
perc_5_spx_av = diff_stats.loc["5%","D_clamp"]

################################################################

start_date = "2007-01-01"
end_date = "2009-12-31"
D_both = pd.merge(spx_av[(spx_av.date >= start_date) & (spx_av.date <= end_date)].set_index("date").rename(columns = {"D_clamp":"SPX"}),
                  ind_mean[(ind_mean.date >= start_date) & (ind_mean.date <= end_date)].set_index("date").rename(columns = {"D":"IND"}),
                  left_index = True, right_index = True)

plt.rcParams["date.autoformatter.month"] = "%m-%d"
D_both.plot(figsize = (12,8))
date_list = ["2007-07-28", "2007-08-15", "2007-10-10", "2007-12-01", "2008-01-02",
             "2008-01-25", "2008-02-10", "2008-03-01", "2008-03-20", "2008-05-25",
             "2008-07-05", "2008-07-17", "2008-08-15", "2008-09-09", "2008-09-15", "2008-09-29", "2008-10-24",
             "2008-11-03", "2008-11-18", "2008-12-05", "2008-12-15",  "2009-01-01", "2009-01-15",
             "2009-01-31", "2009-02-28", "2009-03-10"]
date_list = [pd.to_datetime(x) for x in date_list]
for date in date_list:
    plt.axvline(x = date, color = "0.5", alpha = 0.9, linewidth = 0.75) 
plt.xticks(date_list, rotation='vertical', rotation_mode="anchor")

# Dates where the measure moved significantly, either greater
# than 2 standard deviations in absolute value, or in the tail
# 5 percents of all movements
large_moves_std = spx_av_diff[np.abs(spx_av_diff.D_clamp) >= 3*std_spx_av]


################################################################
# Dealing with 

filepath = 'crisis_chronology.txt'
line_arr = []
with open(filepath) as fp:
    line = fp.readline()
    line_arr.append(line)
    cnt = 1
    while line:
        line = fp.readline()
        line_arr.append(line)
        cnt += 1

# Collapsing the same date event into the same line\n",
dow_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
line_arr_new = []
i = 0
while i < len(line_arr):
    cur_new_line = line_arr[i].replace("\n", " ")
    j = i + 1
    while line_arr[j].split(" ")[0] not in [x+"," for x in dow_list]:
        cur_new_line += " " + line_arr[j].replace("\n", " ")
        j += 1
        if j == len(line_arr):
            break

    line_arr_new.append(cur_new_line)
    i = j

# Converting dates for each line:
date_list = [x.split(":")[0] for x in line_arr_new]
date_list = [x.split(", ")[1:] for x in date_list]
date_list_processed = []
for i, date_str in enumerate(date_list):
    if len(date_str) == 2:
        date_to_convert = date_str[0] + " " + date_str[1]
        date_list_processed.append(pd.to_datetime(date_to_convert))
    else:
        date_list_processed.append(pd.to_datetime(date_str[0]))
        
# Generating a pandas dataframe with all dates and descriptions:
pd.DataFrame({"date":date_list_processed, 
              "desc":[x.split(":")[1] for x in line_arr_new]}).to_csv("crisis_timeline/crisis_timeline_processed.csv", index = False)

        


    
df = pd.read_csv("estimated_data/disaster_sorts/port_sort_const_agg.csv")
df["form_date"] = pd.to_datetime(df["form_date"])
df = pd.pivot_table(df, index = ["permno", "form_date"], values = "port", columns = ["variable", "days"])
    
    
    
    
    
    

df = pd.read_csv("crsp_ret.csv")
df["date"] = pd.to_datetime(df["date"])
np.sum(df.groupby(["permno", "date"])["permco_mktcap"].count() > 1)






