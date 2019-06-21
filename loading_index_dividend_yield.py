"""
Script to load index dividend yield
"""

import numpy as np
import pandas as pd
import wrds
import time

# Connecting to WRDS database:
db = wrds.Connection(wrds_username = "rsigalov")

query = """
select 
*
from OPTIONM.IDXDVD
where secid = 108105
"""

query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)

df.to_csv("data/raw_data/div_yield_spx.csv")

