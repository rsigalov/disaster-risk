import numpy as np
import pandas as pd
import wrds

db = wrds.Connection(wrds_username = "rsigalov")

# Downloading interest rates for all dates and all maturity:
query_zcb = """
select *
from OPTIONM.ZEROCD
where days <= 365*3
"""

query_zcb = query_zcb.replace('\n', ' ').replace('\t', ' ')
df_zcb = db.raw_sql(query_zcb)

df_zcb.to_csv("data/raw_data_new/zcb_data.csv", index = False)
