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
import wrds
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
           
def get_disaster_factors(innovation_method, level_filter = None, 
                         var_filter = None, day_filter = None,
                         extrapolation_filter = None):
    r'''
    Function to get various disaster risk factors and their innovations. 
    
    Args:
        innovation_method: String for how to compute innovations in disaster 
                           risk factors. 
                               'AR' uses an AR1 model 
                               'fd' uses first-differences
        level_filter: List of filters to apply to whether disaster risk comes 
                      from sp_500 or individual firms (ind)
        var_filter: List of filters to apply to the disaster risk measure 
                    (D, rn_prob_2sigma, rn_prob_20, rn_prob_40, rb_prob_60)
        day_filter: List of filters to apply to duration of options that 
                    went into measure (30, 60, 120)
        extrapolation_filter: whether the extrapolate a measure when smiles
                    don't straddle X days (Y, N)
                    
    
    Returns:
        df: Dataframe where index is date and columns are various disaster
            risk factors
        df_innov: Dataframe containing innovations to disaster risk factors
    '''
    
    # == Check inputs == #
    if innovation_method not in ['AR', 'fd']:
        raise ValueError("innovation_method must be either 'AR' or 'fd'")
    
    # == Read in raw data == #
    raw_f = pd.read_csv("../estimated_data/disaster_risk_measures/" +\
                        "combined_disaster_df.csv")
    raw_f['date_eom'] = pd.to_datetime(raw_f['date'])
    raw_f.drop('date', axis = 1, inplace = True)
    
    # == Focus only on direct (for S&P 500) and filtered mean aggregation == #
    raw_f = raw_f[raw_f.agg_type.isin(['direct', 'mean_all'])]
    
    # == Apply other filters == #
    if level_filter is not None:
        raw_f = raw_f[raw_f['level'].isin(level_filter)]
    if var_filter is not None:
        raw_f = raw_f[raw_f['var'].isin(var_filter)]
    if day_filter is not None:
        raw_f = raw_f[raw_f['days'].isin(day_filter)]
    if extrapolation_filter is not None:
        raw_f = raw_f[raw_f['extrapolation'].isin(extrapolation_filter)]
        
    # == Create variable names == #
    raw_f['name'] = raw_f['level'] + '_' + raw_f['var'] +\
                    '_' + raw_f['days'].astype(str) + '_' + raw_f['extrapolation']
                    
    # == Create pivot table, then resample to end of month == #
    pdf = raw_f.pivot_table(index = 'date_eom', columns = 'name', 
                            values = 'value')
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
    # db = wrds.Connection(wrds_username=wrds_un)
    db = wrds.Connection()
    
    # == Get CRSP monthly data, filling in delisted returns == #
    crsp = crsp_comp.get_monthly_returns(db, start_date = '1986-01-01',
                                    end_date = '2017-01-01', balanced = True)
    
    # == Read in relevant factors and compute their innovations == #
    dis_fac, dis_fac_innov = get_disaster_factors(innovation_method = imethod,
                                                  level_filter = [argv[3]],
                                                  var_filter = [argv[1]],
                                                  day_filter = [float(argv[2])],
                                                  extrapolation_filter = ['N'])

    # == Merge returns with factor innovations == #
    crsp_with_fac = pd.merge(crsp, dis_fac_innov,
                             left_on = ['date_eom'],
                             right_index = True,
                             how = 'left')
    
    # == Compute betas with respect to innovations in each factor == #
    for f in dis_fac_innov.columns:
        beta_f = crsp_with_fac.groupby('permno')['ret', f].\
                    apply(RollingOLS, 24, 18)
        crsp_with_fac = crsp_with_fac.join(beta_f)
        
    # == Output permno-date-betas == #
    output_df = crsp_with_fac[['permno', 'date_eom'] + \
                              ['beta_' + x for x in dis_fac_innov.columns]]
    output_df.to_csv('../estimated_data/disaster_risk_betas/' +\
                     'disaster_risk_betas.csv', index = False)
    print('Computed betas with respect to disaster risk factors ' +\
          'in %.2f minutes' %((time.time() - s) / 60))
    
##  -----------------------------------------------------------------------
##                             main program
##  -----------------------------------------------------------------------

if __name__ == "__main__": sys.exit(main(sys.argv))
            