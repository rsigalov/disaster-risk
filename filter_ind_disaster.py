"""
This script filters and averages to monthly individual daily disaster series.
In particular, it takes file 

	estimated_data/interpolated_D/int_ind_disaster_days_<days>.csv

where <days> is user defined number of days. Then it averages individual
disaster measures to monthly level requiring at least <min_obs> observations
per month. Otherwise, it doesn't average the measures.

The script saves the resulting file as

	estimated_data/interpolated_D/mon_ind_disaster_days_<days>.csv
"""

from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd

import crsp_comp

def main(argv=None):
	days = argv[1]
	if len(argv) > 2:
		suffix = argv[2]
		disaster_df = crsp_comp.load_and_filter_ind_disaster(int(days), 5, 0, suffix)
		disaster_df.to_csv("estimated_data/interpolated_D/mon_ind_disaster_" + suffix + "_days_" + days + ".csv")
	else:
		disaster_df = crsp_comp.load_and_filter_ind_disaster(int(days), 5, 0)
		disaster_df.to_csv("estimated_data/interpolated_D/mon_ind_disaster_days_" + days + ".csv")



if __name__ == "__main__": sys.exit(main(sys.argv))