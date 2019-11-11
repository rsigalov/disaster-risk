from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
import wrds
import time
from arch.univariate import ARX
pd.options.display.max_columns = 20
from numpy.linalg import inv
import crsp_comp

import rolling_disaster_betas as rdb

def main(argv = None):

	# == Parameters == #
	s = time.time()

	# == Establish WRDS connection == #
	db = wrds.Connection()

	print("\nGetting CRSP returns\n")
	# == Get CRSP monthly data, filling in delisted returns == #
	crsp = crsp_comp.get_monthly_returns(db, start_date = '1986-01-01',
	                                end_date = '2017-12-31', balanced = True)

	# Getting Cremers factor:
	cremers_df = pd.read_csv("data/cremers_factors.csv")
	cremers_df.drop(columns = "VOL", inplace = True)

	# Aggregating to daily data
	cremers_df["date"] = pd.to_datetime(cremers_df["date"])
	cremers_df["date_mon"] = cremers_df["date"] + pd.offsets.MonthEnd(0)
	cremers_df["JUMP"] = cremers_df["JUMP"] + 1
	cremers_df = pd.DataFrame(cremers_df.groupby("date_mon")["JUMP"].prod())
	cremers_df["JUMP"] = cremers_df["JUMP"] - 1

	# == Merge monthly returns on Cremers factor with CRSP returns == #
	crsp_with_fac = pd.merge(
		crsp, cremers_df, 
		left_on = ['date_eom'], right_index = True, how = 'left')

	print("\nComputing betas with respect to all factors\n")
	# == Compute betas with respect to innovations in each factor == #

	for i, f in enumerate(cremers_df.columns):
	    print("Factor %d out of %d"%(i+1, len(cremers_df.columns)))
	    beta_f = crsp_with_fac.groupby('permno')['ret', f].\
	                apply(rdb.RollingOLS, 24, 18)
	    crsp_with_fac = crsp_with_fac.join(beta_f)

	# == Output permno-date-betas == #
	output_df = crsp_with_fac[['permno', 'date_eom'] + \
	                          ['beta_' + x for x in cremers_df.columns]]

	output_df.to_csv(
		'estimated_data/disaster_risk_betas/cremers_jump_betas.csv', index = False)
	print('Computed betas with respect to disaster risk factors ' +\
	      'in %.2f minutes' %((time.time() - s) / 60))

	# Now using sorts with respect to cremers factors to sort portfolios:
	ncuts = 5
	roll_betas = pd.read_csv("estimated_data/disaster_risk_betas/cremers_jump_betas.csv")
	roll_betas.rename(columns = {"date_eom": "date"}, inplace = True)
	roll_betas["date"] = pd.to_datetime(roll_betas["date"])
	roll_betas = roll_betas[roll_betas.date >= "1997-07-31"]
	roll_betas = roll_betas[roll_betas.date <= "2015-11-30"]

	print("\n---- Computing portfolio characteristics ----\n")
	ports = crsp_comp.monthly_portfolio_sorts(db, roll_betas, ["beta_JUMP"], ncuts)

	print("\n---- Saving Returns, BM and OP ----\n")
	ports["beta_JUMP"]["ret"].to_csv("estimated_data/disaster_sorts/port_sort_cremers_jump_ret.csv")
	ports["beta_JUMP"]["bm"].to_csv("estimated_data/disaster_sorts/port_sort_cremers_jump_bm.csv")
	ports["beta_JUMP"]["op"].to_csv("estimated_data/disaster_sorts/port_sort_cremers_jump_op.csv")


if __name__ == "__main__": sys.exit(main(sys.argv))


