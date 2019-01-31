import wrds
db = wrds.Connection()


"""
Created on Thu Jan 17 22:23:53 2019

@author: rsigalov
"""

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



