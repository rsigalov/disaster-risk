from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import wrds
import datetime
import time
import os
from os.path import isfile, join
pd.options.display.max_columns = 20
from pandas.tseries.offsets import *

# Libraries to import and unpack .zip file from Ken French's website with
# 5 Fama-French factors
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import crsp_comp


def main(argv=None):
	print("Start testing")
	db = wrds.Connection()

	# Loading the data on disaster characteristics:
	print("Loading disaster dataframe")
	disaster_df = crsp_comp.load_and_filter_ind_disaster(30, 5, 0)

	# print("Saving disaster dataframe")
	# disaster_df.to_csv("disaster_df_test.csv")

	# print("Loading test disaster dataframe")
	# disaster_df = pd.read_csv("disaster_df_test.csv")
	# disaster_df["date"] = pd.to_datetime(disaster_df["date"])

	print("Loading data on month-by-month CRSP-OM link")
	crsp_om_link = pd.read_csv("CRSP_OM_link_by_month.csv")
	crsp_om_link["date"] = pd.to_datetime(crsp_om_link["date"])

	print("Adding permno to disaster dataframe")
	disaster_df = pd.merge(disaster_df, crsp_om_link, on = ["secid", "date"])

	print("Doing portfolio sorts")
	ports = crsp_comp.monthly_portfolio_sorts(db, disaster_df, ["D_clamp"], 5)

	print("Saving results")
	ports["D_clamp"]["ret"].to_csv("ports_ret_test.csv")
	ports["D_clamp"]["bm"].to_csv("ports_bm_test.csv")
	ports["D_clamp"]["op"].to_csv("ports_op_test.csv")

	print("Done testing")




if __name__ == "__main__": sys.exit(main(sys.argv))