"""
This script looks into the industry composition of interpolated
disaster measures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading individual disaster measures:
disaster_level = pd.read_csv("data/sorting_variables/monthly_average_D_clamp.csv")
disaster_maturity = pd.read_csv("data/sorting_variables/monthly_average_D_clamp_by_maturity.csv")
disaster_level["maturity"] = "level"
disaster_maturity = disaster_maturity.rename(columns={"days": "maturity"})
disaster = pd.concat([disaster_maturity, disaster_level], ignore_index=True)
disaster["date"] = pd.to_datetime(disaster["date"])

# Loading data on CRSP SIC codes:
crsp_sic = pd.read_csv("data/crsp_ret.csv")
crsp_sic = crsp_sic[["permno", "date", "hsiccd"]].drop_duplicates()
crsp_sic["hsiccd"] = pd.to_numeric(crsp_sic["hsiccd"], errors = "coerce")
crsp_sic["date"] = pd.to_datetime(crsp_sic["date"])

disaster = pd.merge(disaster, crsp_sic, on=["permno", "date"], how="left")

# loading data on 12 FF industries 
ff_ind = pd.read_csv("data/ff_12_ind.csv")
ff_ind = ff_ind.drop(columns = "ind_num")
# ff_ind = pd.read_excel("data/ff_17_modified.xlsx", sheet_name="Sheet1")
disaster["ff_ind"] = None
for i_row in range(ff_ind.shape[0]):
    print(i_row)
    beg_sic = ff_ind.iloc[i_row, 1]
    end_sic = ff_ind.iloc[i_row, 2]
    ff_ind_name = ff_ind.iloc[i_row, 0]
    disaster.loc[
            (disaster["hsiccd"] >= beg_sic) & 
            (disaster["hsiccd"] <= end_sic), "ff_ind"] = ff_ind_name

disaster.loc[disaster["hsiccd"].notnull() & disaster.ff_ind.isnull(), "ff_ind"] = "Other"
disaster = disaster[disaster.ff_ind.notnull()]
disaster = disaster[["secid", "date", "maturity", "value", "hsiccd", "ff_ind"]]

# Averaging disaster series within industries on daily and monthly levels:
def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.975)) & (x >= np.quantile(x, 0.025))])

disaster = disaster[disaster["value"].notnull()]
ff_ind_disaster = disaster[disaster["maturity"] == 60].groupby(["date", "ff_ind"])["value"].apply(mean_with_truncation).rename("value")
ff_ind_disaster = pd.pivot_table(pd.DataFrame(ff_ind_disaster), columns = "ff_ind", index = "date", values="value")
ff_ind_disaster.plot(figsize = (10,8))
plt.show()

# Replacing negative values with zeros to do an area plot
mat = np.array(ff_ind_disaster)
mat = np.where(np.logical_or(mat < 0, np.isnan(mat)), 0.0, mat)
ff_ind_disaster_adj = pd.DataFrame(mat, index=ff_ind_disaster.index, columns=ff_ind_disaster.columns)

ff_ind_disaster_adj.div(ff_ind_disaster_adj.sum(axis=1), axis=0).rolling(5).mean().plot.area(figsize=(10, 8), stacked=True)
plt.show()