"""
Script to get volume from svi_params files and combine them in one
file. Primarily to download the data from AWS more easily
"""

from __future__ import print_function
from __future__ import division
import sys

import pandas as pd
import numpy as np
import os

def main(argv = None):
	path = "data/raw_data/"
	file_list = [x for x in os.listdir(path) if "svi_params" in x]
	columns_to_leave = ["secid", "obs_date", "exp_date", "T", "volume", "open_inetrest"]
	df = pd.DataFrame(columns = columns_to_leave)

	for ifile, file in enumerate(file_list):
		print("File %d out of %d" % (ifile, len(file_list)))
		df_to_append = pd.read_csv(path + file)
		df_to_append = df_to_append[columns_to_leave]
		df = df.append(df_to_append)

	df.to_csv(path + "option_volume_3.csv", index = False)

if __name__ == "__main__": sys.exit(main(sys.argv))