"""
File to load all output and see which companies for which years
I already estimated
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
import os
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

import glob

# 1. Loading all data that I have in folder
for (i, filename) in enumerate(glob.glob("output/var_ests_equity_*")):
    if i == 0:
        df = pd.read_csv(filename)
    else:
        df = df.append(pd.read_csv(filename))
        
# 2. Leaving only year and secid:
df = df[["secid", "date"]]
df["date"] = pd.to_datetime(df["date"])
df["year"] = [x.year for x in df["date"]]
df = df.drop("date", axis = 1)
df = df.drop_duplicates()

# 3. Getting data on all companies:

