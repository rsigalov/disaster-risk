import numpy as np
import pandas as pd
import wrds
import os

db = wrds.Connection()
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')


query_check = """
select *
from OPTIONM.OPPRCD2017
where secid = 108105
limit 100
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)

check_data.to_csv("data/some_names.csv")


query_check = """
select *
from OPTIONM.OPTION
where optionid = 115849688
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)

query_check = """
select *
from OPTIONM.OPTION
where optionid = 115666544
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)


query_check = """
select *
from OPTIONM.ZEROCD
where date >= '2017-01-01'
and days <= 100
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)
check_data.to_csv("data/zcb_cont_comp_rate.csv")


################################################
# Loading option and distribution data for Apple
# to get a sense of how dividends work

f = open("load_option_list.sql", "r")
query = f.read()
_secid_ = '101594'
query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('_secid_', _secid_)
opt_data = db.raw_sql(query)
opt_data.to_csv('data/opt_data_aapl.csv', index = False)

query_check = """
select 
    secid, ex_date, amount, distr_type
from OPTIONM.DISTRD
where ex_date >= '2017-01-01'
and ex_date < '2018-01-01'
and secid = 101594
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)

check_data.to_csv("data/dist_hist_aapl.csv")

################################################
# Loading data on dividend yield for SPX
query = """
select * 
from OPTIONM.IDXDVD 
where secid = '108105' 
and date >= '2017-01-01'
and date < '2018-01-01'
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)
df.to_csv("data/spx_dividend_yield.csv")


query = """
select *
from OPTIONM.OPPRCD2017
where secid = 101594
and date = '2017-01-03'
and exdate = '2017-02-03'
and strike_price = 100000
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)

############################################
# Loading option data on dates right after the Lehman collapse
# 1. Data on option prices:
f=open("load_option_list.sql", "r")
query_prices = f.read()

start_date = "'2008-09-16'"
end_date = "'2008-09-30'"
secid = '108105'
data_base = 'OPTIONM.OPPRCD2008'

query_prices = query_prices.replace('\n', ' ').replace('\t', ' ')
query_prices = query_prices.replace('_start_date_', start_date).replace('_end_date_', end_date)
query_prices = query_prices.replace('_secid_', secid)
query_prices = query_prices.replace('_data_base_', data_base)
df_prices = db.raw_sql(query_prices)

# 2. Data on index dividend yield:
query_div = """
select * 
from OPTIONM.IDXDVD 
where secid = '108105' 
and date >= '2008-09-16'
and date < '2008-09-30'
"""
query_div = query_div.replace('\n', ' ').replace('\t', ' ')
df_div = db.raw_sql(query_div)

# 3. Data on ZCB rates:
query_zcb = """
select *
from OPTIONM.ZEROCD
where date >= '2008-09-16'
and date <= '2008-09-30'
and days <= 100
"""

query_zcb = query_zcb.replace('\n', ' ').replace('\t', ' ')
df_zcb = db.raw_sql(query_zcb)

# 4. Saving data:
df_prices.to_csv('data/opt_data_lehman.csv', index = False)
df_div.to_csv('data/div_yield_lehman.csv', index = False)
df_zcb.to_csv('data/zcb_rates_lehman.csv', index = False)







f=open("load_option_list.sql", "r")
query_prices = f.read()

start_date = "'2008-01-01'"
end_date = "'2008-12-31'"
secid = "108105"
data_base = "OPTIONM.OPPRCD2008"

query_prices = query_prices.replace('\n', ' ').replace('\t', ' ')
query_prices = query_prices.replace('_start_date_', start_date).replace('_end_date_', end_date)
query_prices = query_prices.replace('_secid_', secid)
query_prices = query_prices.replace('_data_base_', data_base)
df_prices = db.raw_sql(query_prices)
df_prices.to_csv("data/opt_data_2008.csv", index = False)


query = """
select * 
from OPTIONM.IDXDVD 
where secid = '108105' 
and date >= '2008-01-01'
and date <= '2008-12-31'
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)
df.to_csv("data/spx_dividend_yield_2008.csv")

query_check = """
select *
from OPTIONM.ZEROCD
where date >= '2008-01-01'
and date <= '2008-12-31'
and days <= 100
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)
check_data.to_csv("data/zcb_cont_comp_rate_2008.csv")


############################################
# Getting data on companies and 
path_to_file = "/Users/rsigalov/Downloads/available_options.csv"
av_options = pd.read_csv(path_to_file)

av_options.drop('index_flag', axis = 1, inplace = True)
av_options.drop('sic', axis = 1, inplace = True)

# Getting not so liquid companies:
av_options["date"] = pd.to_datetime(av_options["date"].astype(int).astype(str), format = "%Y%m%d")

av_options[(av_options["date"] >= "1998-03-01") & (av_options["date"] < "1998-04-01")].sort_values(["open_interest"])

df_sub_1 = av_options[av_options["date"] == "1996-12-02"].sort_values('volume')
df_sub_2 = av_options[av_options["date"] == "2001-01-10"].sort_values('volume')
df_sub_3 = av_options[av_options["date"] == "2002-08-22"].sort_values('volume')
df_sub_4 = av_options[av_options["date"] == "2007-10-17"].sort_values('volume')
df_sub_5 = av_options[av_options["date"] == "2013-05-31"].sort_values('volume')


dates = ["1996-12-02", "2001-01-10", "2002-08-22", "2007-10-17", "2013-05-31"]
perc_list = [0.25, 0.5, 0.75, 0.9, 0.95]
opt_list = []

for date in dates:
    for perc in perc_list:
        df_sub_i = av_options[av_options["date"] == date].sort_values('volume')
        row_perc = df_sub_i.iloc[int(df_sub_i.shape[0]*perc) - 1]
        opt_list.append({"secid": row_perc["secid"], "date": date})
        
        
sec_list = "(" + ", ".join([str(x["secid"]) for x in opt_list]) + ")"

# Downloading option data for companies with different liquidity:
for i in range(len(opt_list)):
    print(i)
    if i == 0:
        f = open("load_option_list.sql", "r")
        query = f.read()
        
        start_date = "'"+opt_list[i]["date"]+"'"
        end_date = "'"+opt_list[i]["date"]+"'"
        secid = str(opt_list[i]["secid"])
        data_base = "OPTIONM.OPPRCD" + str(pd.to_datetime(opt_list[i]["date"]).year)
        
        query = query.replace('\n', ' ').replace('\t', ' ')
        query = query.replace('_start_date_', start_date).replace('_end_date_', end_date)
        query = query.replace('_secid_', secid)
        query = query.replace('_data_base_', data_base)
        df_prices = db.raw_sql(query)
    else:
        f = open("load_option_list.sql", "r")
        query = f.read()
        
        start_date = "'"+opt_list[i]["date"]+"'"
        end_date = "'"+opt_list[i]["date"]+"'"
        secid = str(opt_list[i]["secid"])
        data_base = "OPTIONM.OPPRCD" + str(pd.to_datetime(opt_list[i]["date"]).year)
        
        query = query.replace('\n', ' ').replace('\t', ' ')
        query = query.replace('_start_date_', start_date).replace('_end_date_', end_date)
        query = query.replace('_secid_', secid)
        query = query.replace('_data_base_', data_base)
        df_prices_to_append = db.raw_sql(query)
        
        df_prices = df_prices.append(df_prices_to_append)


df_prices.to_csv("data/test_opt_data.csv", index = False)

# Downloading data for distributions for these secids:
query = """
select 
    secid, ex_date, amount, distr_type
from OPTIONM.DISTRD
where secid in _secid_list_
"""

query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('_secid_list_', sec_list)
dist_data = db.raw_sql(query)

dist_data.to_csv("data/test_dist_data.csv", index = False)

# Downloading interest rates for all dates and all maturity:
query_zcb = """
select *
from OPTIONM.ZEROCD
where days <= 365*2
"""

query_zcb = query_zcb.replace('\n', ' ').replace('\t', ' ')
df_zcb = db.raw_sql(query_zcb)

df_zcb.to_csv("data/test_zcb_data.csv", index = False)

########################################################
# Getting data on top companies in 2008
########################################################
for t in range(2012, 2018, 1):
    print(t)
    query = """
        select secid, count(strike_price) as cnt 
        from OPTIONM.OPPRCD2008
        group by secid order by cnt desc
    """
    query = query.replace('\n', ' ').replace('\t', ' ')
    query = query.replace('OPTIONM.OPPRCD2008', 'OPTIONM.OPPRCD' + str(t))
    df_comps_to_append = db.raw_sql(query)
    if t == 2012:
        df_comps = df_comps_to_append
    else:
        df_comps = df_comps.append(df_comps_to_append)

df_comps = df_comps.groupby("secid")["cnt"].sum()
df_comps = df_comps.sort_values(ascending = False)
df_comps = df_comps.reset_index()

# 1. Removing the index:
query = """
select secid, issue_type
from OPTIONM.SECURD
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_issue_type = db.raw_sql(query)

# Merging on TOTAL number of options in 2017:
df_comps = pd.merge(df_comps, df_issue_type, on = "secid")
df_comps = df_comps[df_comps["issue_type"] == "0"]
df_comps.to_csv("data/ranked_equity_secid_2006_2011.csv", index = False)

########################################################
# Generating shell file to load data 
df_comps = pd.read_csv("data/ranked_equity_secid_2006_2011.csv")
top_comps = 100
group_size = 5
groups = int(top_comps/group_size)
start_index = 21

f = open("script_to_run_5", "w")

f.write("python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(df_comps["secid"][group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 2006 -e 2011 -o equity_" + str(i_group + start_index) + "\n")
    f.write("julia -p 7 fit_smiles.jl equity_" + str(i_group + start_index) + "\n")
    f.write("julia -p 7 est_parameters.jl equity_" + str(i_group + start_index) + "\n")
    f.write("rm data/opt_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("rm data/dist_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()


########################################################
# Doing the approach with volume of trading:
query = """
select secid, extract(year from date) as year, sum(volume) as volume, 
sum(open_interest) as op_int
from OPTIONM.OPVOLD
where cp_flag in ('C', 'P')
and date >= '1996-01-01' and date <= '2005-12-31'
group by extract(year from date), secid
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_volume = db.raw_sql(query)
df_volume = df_volume.sort_values("volume", ascending = False)

# Getting info on the type of the company:
query = """
select secid, issue_type
from OPTIONM.SECURD
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_issue_type = db.raw_sql(query)

# Merging on TOTAL number of options in 2017:
df_volume = pd.merge(df_volume, df_issue_type, on = "secid")
df_volume = df_volume[df_volume["issue_type"] == "0"]
df_volume = df_volume.drop("issue_type", axis = 1)
df_volume = df_volume.sort_values("volume", ascending = False)

# Looking at intersection of top 100 companies by volume with
# top 100 companies by number of options:

# Summing volume across years:
top_100_volume = df_volume.groupby(["secid"])["volume"].sum().sort_values(ascending = False).reset_index()

top_100_volume = top_100_volume.iloc[0:155]
top_100_number = df_comps.iloc[0:100]

top_100_comparison = pd.merge(top_100_volume, top_100_number, on = "secid", how = "left")
top_100_comparison[top_100_comparison["cnt"].isnull()]

secid_to_estimate = list(top_100_comparison[top_100_comparison["cnt"].isnull()]["secid"])

# Generating script_to_run_6
secid_to_estimate = top_100_volume["secid"]
top_comps = 100
group_size = 5
groups = int(top_comps/group_size)
start_index = 81

f = open("script_to_run_8", "w")

f.write("python load_zcb.py\n")
f.write("\n")

for i_group in range(groups):
    f.write("python loading_data.py -s ")
    f.write(",".join([str(x) for x in list(secid_to_estimate[group_size*i_group:(group_size + group_size*i_group)])]) + " ")
    f.write(" -b 1996 -e 2005 -o equity_" + str(i_group + start_index) + "\n")
    f.write("julia -p 7 fit_smiles.jl equity_" + str(i_group + start_index) + "\n")
    f.write("julia -p 7 est_parameters.jl equity_" + str(i_group + start_index) + "\n")
    f.write("rm data/opt_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("rm data/dist_data_equity_" + str(i_group + start_index) + ".csv\n")
    f.write("\n")
    
f.close()

# Getting intersecting comapnies in three samples that I consider:
query = """
select secid, extract(year from date) as year, sum(volume) as volume, 
sum(open_interest) as op_int
from OPTIONM.OPVOLD
where cp_flag in ('C', 'P')
group by extract(year from date), secid
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_volume = db.raw_sql(query)
df_volume = df_volume.sort_values("volume", ascending = False)



# Top-100 companies for 1996-2005
top_100_96_05 = df_volume[df_volume["year"].isin([str(x) for x in range(1996, 2006, 1)])].groupby(["secid"])["volume"].sum().sort_values(ascending = False).reset_index().iloc[0:100]
top_100_96_05["part1"] = 1
top_100_96_05 = top_100_96_05.drop("volume", axis = 1)

# Top-100 companies for 2006-2011
top_100_06_11 = df_volume[df_volume["year"].isin([str(x) for x in range(2006, 2012, 1)])].groupby(["secid"])["volume"].sum().sort_values(ascending = False).reset_index().iloc[0:100]
top_100_06_11["part2"] = 1
top_100_06_11 = top_100_06_11.drop("volume", axis = 1)

# Top-100 companies for 2012-2017
top_100_12_17 = df_volume[df_volume["year"].isin([str(x) for x in range(2012, 2018, 1)])].groupby(["secid"])["volume"].sum().sort_values(ascending = False).reset_index().iloc[0:100]
top_100_12_17["part3"] = 1
top_100_12_17 = top_100_12_17.drop("volume", axis = 1)

top_100 = pd.merge(top_100_96_05, top_100_06_11, on = "secid", how = "outer")
top_100 = pd.merge(top_100, top_100_12_17, on = "secid", how = "outer")

top_100[(top_100["part1"] == 1) & (top_100["part2"] == 1) & (top_100["part3"] == 1)].shape


