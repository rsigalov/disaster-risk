"""
Loading data on return on S&P index and 
"""

import numpy as np
import pandas as pd
import wrds
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision/')

db = wrds.Connection(wrds_username = "rsigalov")

# 1. Downloading monthly index data:
query = """
    select 
        caldt as date,
        vwretd as vw_ret,
        sprtrn as sp_ret
    from crspa.msp500
"""

query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)

df.to_csv("estimated_data/crsp_data/crsp_monthly_index_returns.csv")

# 2. Downloading monthly individual company data:
query = """
    select 
        m.date, m.permno, m.permco, m.cusip,
        case when (m.ret in (-66.0, -77.0, -88.0, -99.0)) then NULL else m.ret end as ret,
        abs(prc) as prc, 
        shrout
    from crsp.msf as m
    left join crsp.msenames as n
        on m.permno = n.permno
        and m.date >= n.namedt
        and m.date <= n.nameendt
    left join crsp.msedelist as d
        on m.permno = d.permno
        and date_trunc('month', m.date) = date_trunc('month', d.dlstdt)
        where shrcd in (10, 11)
        and exchcd in (1,2,3)
"""

query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)

df.to_csv("estimated_data/crsp_data/crsp_monthly_returns.csv")



