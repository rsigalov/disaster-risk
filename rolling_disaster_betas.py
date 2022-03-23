""" This is the docstring for rolling_disaster_betas.py.
This file uses CRSP monthly data to compute rolling betas with respect to
several disaster-risk factors.

Authors: Roman Sigalov and Emil Siriwardane

Affiliations: Both at Harvard Business School

"""

##  -----------------------------------------------------------------------
##                                import
##  -----------------------------------------------------------------------

from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import pandas as pd
# import wrds
import time
from arch.univariate import ARX
pd.options.display.max_columns = 20
from numpy.linalg import inv
import crsp_comp

##  -----------------------------------------------------------------------
##                               functions
##  -----------------------------------------------------------------------
def RollingOLS_helper(df, betas, window, min_periods):
    r'''
    Helper function to speed up rolling regression
    '''

    N = len(df)

    for t in range(0, N):
        # Drop nans, then check if length of non-nan dataframe is sufficient
        df_sub = df[max(0, t - window + 1):t+1]
        nanidx = ~np.isnan(df_sub).any(axis=1)
        df_tmp = df_sub[nanidx]
        #non_missing = len(df_tmp)


        if len(df_tmp) < min_periods:
        	# If less than min_periods periods are not NaN
        	# set beta to NaN
            beta = -100
        else:
            # Creating Y and X arrays
            Y = np.array(df_tmp[:, 0])
            X = np.hstack((np.ones((len(df_tmp),1)),
                           np.array(df_tmp[:, 1:])))

            params = inv((X.T.dot(X))).dot(X.T).dot(Y)
            beta = params[1:]

        # Store betas and idiosyncratic volatilities into array
        betas[t, :] = beta

    return betas

def RollingOLS(df, window, min_periods = None):
    r'''
    Function to run rolling regressions. Speed improvements over pandas rolling

    Args:
        df: Two-column dataframe where the first column is the y-variable and
            the second column is the x-variable
        window: Integer for the window over which to run rolling regressions
        min_periods: Integer of the minimum number of periods required

    Returns: Dataframe containing rolling betas

    '''

    xvars = df.columns[1:]
    index = df.index

    # If now min_periods is supplied then set to window size
    if min_periods is None:
        min_periods = window

    # Arrays to store betas
    a = -100 * np.ones(np.shape(df[xvars]))
    betas = RollingOLS_helper(df.values, a, window, min_periods)

    betas[betas == -100] = np.nan

    return pd.DataFrame(betas, index = index,
                        columns = ['beta_' + x for x in xvars])

def get_disaster_factors(innovation_method, agg_freq = "mon", resample = True):
    r'''
    Function to get various disaster risk factors and their innovations.

    Args:
        innovation_method: String for how to compute innovations in disaster
                           risk factors.
                               'AR' uses an AR1 model
                               'fd' uses first-differences
        agg_freq: can be either "mon" or "week"

    Returns:
        df: Dataframe where index is date and columns are various disaster
            risk factors
        df_innov: Dataframe containing innovations to disaster risk factors
    '''

    if agg_freq == "mon":
        agg_freq = "date_mon"
    elif agg_freq == "week":
        agg_freq = "date_week"
    else:
        raise ValueError("agg_freq should be either 'mon' or 'week'")

    # == Check inputs == #
    if innovation_method not in ['AR', 'fd']:
        raise ValueError("innovation_method must be either 'AR' or 'fd'")

    # == Read in raw data == #
    raw_f = pd.read_csv("estimated_data/disaster_risk_measures/" +\
                        "disaster_risk_measures.csv")
    raw_f['date'] = pd.to_datetime(raw_f['date'])
    raw_f = raw_f[raw_f.agg_freq == agg_freq]
    # raw_f = raw_f[raw_f.variable.isin(["D_clamp", "rn_prob_5", "rn_prob_20", "rn_prob_80"]) & 
    #               raw_f.maturity.isin(["level", "30", "180"])]
    raw_f = raw_f[raw_f.variable.isin(["D_clamp"]) & 
                  raw_f.maturity.isin(["level"]) &
                  (raw_f.level == "Ind")]

    # == Create variable names == #
    raw_f['name'] = raw_f['level'] + '_' + raw_f['variable'] +\
                    '_' + raw_f['maturity'].astype(str)

    # == Create pivot table, then resample to end of month == #
    pdf = raw_f.pivot_table(index = 'date', columns = 'name',
                            values = 'value')
    if resample:
        pdf = pdf.resample('M').last()

    # == Compute innovations in each factor == #
    if innovation_method == 'fd':
        df = pdf.diff()
    elif innovation_method == 'AR':
        df = pd.DataFrame(index = pdf.index, columns = pdf.columns)
        for col in df.columns:
            ar = ARX(pdf[col], lags = [1]).fit()
            df.loc[ar.resid.index, col] = ar.resid.values
        df = df.astype(float)

    return pdf, df

##  ----------------------------------------------
##  function  ::  main
##  ----------------------------------------------
def main(argv = None):

    # == Parameters == #
    s = time.time()
    wrds_un = 'ens'     # WRDS username
    imethod = 'fd'      # Innovations to factors using first-differences

    # == Establish WRDS connection == #
    db = wrds.Connection()

    # == Constructing Monhtly betas == #
    print("\n === Constructing Monthly betas === \n")
    print("\nGetting CRSP returns\n")
    # == Get CRSP monthly data, filling in delisted returns == #
    crsp = crsp_comp.get_monthly_returns(db, start_date = '1986-01-01',
                                    end_date = '2017-12-31', balanced = True)

    # Getting a zoo of factors:
    print("\nGetting a zoo of factors\n")
    dis_fac, dis_fac_innov = get_disaster_factors(
        innovation_method = imethod, agg_freq = "mon", resample = True)

    # == Merge returns with factor innovations == #
    crsp_with_fac = pd.merge(crsp, dis_fac_innov,
                             left_on = ['date_eom'],
                             right_index = True,
                             how = 'left')

    print("\nComputing betas with respect to all factors\n")

    # == Compute betas with respect to innovations in each factor == #
    for i, f in enumerate(dis_fac_innov.columns):
        print("Factor %d out of %d"%(i+1, len(dis_fac_innov.columns)))
        beta_f = crsp_with_fac.groupby('permno')['ret', f].\
                    apply(RollingOLS, 24, 18)
        crsp_with_fac = crsp_with_fac.join(beta_f)

    # == Output permno-date-betas == #
    output_df = crsp_with_fac[['permno', 'date_eom'] + \
                              ['beta_' + x for x in dis_fac_innov.columns]]

    output_df.to_csv('estimated_data/disaster_risk_betas/' +\
                     'disaster_risk_betas.csv', index = False)

    # # == Constructing Monhtly betas == #
    # print("\n === Constructing Weekly betas === \n")
    # print("\nGetting CRSP returns\n")
    # # == Get CRSP monthly data, filling in delisted returns == #
    # crsp = crsp_comp.get_weekly_returns(
    #     db, start_date = '1986-01-01', end_date = '2017-12-31')

    # # Getting a zoo of factors:
    # print("\nGetting a zoo of factors\n")
    # dis_fac, dis_fac_innov = get_disaster_factors(
    #     innovation_method = imethod, agg_freq = "week", resample = False)
    

    # # == Merge returns with factor innovations == #
    # crsp_with_fac = pd.merge(crsp, dis_fac_innov,
    #                          left_on = ['date'],
    #                          right_index = True,
    #                          how = 'left')

    # print("\nComputing betas with respect to all factors\n")

    # # == Compute betas with respect to innovations in each factor == #
    # for i, f in enumerate(dis_fac_innov.columns):
    #     print("Factor %d out of %d"%(i+1, len(dis_fac_innov.columns)))
    #     beta_f = crsp_with_fac.groupby('permno')['ret', f].\
    #                 apply(RollingOLS, 52*2, int(52*2*0.75))
    #     crsp_with_fac = crsp_with_fac.join(beta_f)

    # # == Output permno-date-betas == #
    # output_df = crsp_with_fac[['permno', 'date'] + \
    #                           ['beta_' + x for x in dis_fac_innov.columns]]

    # output_df.to_csv('estimated_data/disaster_risk_betas/' +\
    #                  'disaster_risk_betas_week.csv', index = False)

    # print("\n === Constructing Weekly betas === \n")

    print('Computed betas with respect to disaster risk factors ' +\
          'in %.2f minutes' %((time.time() - s) / 60))

##  -----------------------------------------------------------------------
##                             main program
##  -----------------------------------------------------------------------

if __name__ == "__main__": sys.exit(main(sys.argv))
