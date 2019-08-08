""" This is the docstring for portfolio_sorts.py. 
This file loads a vector of characteristics related to disaster risk
(e.g., betas with respect to several factors and direct measures) and then
creates portfolios sorted on those characteristics. It also computes the 
average book-to-market ratio of firms in each portfolio at the time of 
formation.

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
import crsp_comp as ccm
# import 
pd.options.display.max_columns = 20


##  -----------------------------------------------------------------------
##                               functions
##  -----------------------------------------------------------------------

def Saver(pdict, dloc):
  r'''
  Function to save dictionary of portfolios to disk. 

  Args:
    pdict: Dictionary of dictionaries. The first level is the characteristic
           used to create the portfolios. The second level contains:
            1. A dataframe of portfolio returns
            2. A dataframe of average book-to-market ratios at formation
            3. A dataframe containing portfolio constituents at formation date
    floc: String of directory where to save files

  Returns: None

  Note: Files will be saved with the following name structure:
          characterstic_datatype.csv. In the case of constituents, we 
          compress them, so they end with .csv.gz
  '''

  for key in pdict.keys():
    for dt in pdict[key].keys():
       # Define filename, without extension
      fn = dloc + '/' + key + '_' + dt 

      if dt == "constituents":
        pdict[key][dt].to_csv(fn +'.csv.gz',index = False,compression = 'gzip')
      else:
        pdict[key][dt].to_csv(fn + '.csv')

        
##  ----------------------------------------------
##  function  ::  main
##  ----------------------------------------------
def main(argv=None):
    
    # == Parameters == #
    s = time.time()    
    wrds_un = 'ens'     # WRDS username
    db = wrds.Connection() # WRDS connection
    ncuts = 5   # Number of portfolios to use
    smoothing = 6 # Number of months to smooth market capitalization for BMs
    
    # == Load in betas with respect to disaster risk factors == #
    df = pd.read_csv("../estimated_data/disaster_risk_betas/" +\
                     "disaster_risk_betas.csv").dropna()
    df['date'] = pd.to_datetime(df['date_eom'])
    df.drop('date_eom', 1, inplace = True)
    sort_cols = [x for x in df.columns if x not in ['permno', 'date']]
    
    # == For monthly portfolios based on disaster risk == #
    disaster_portfolios = ccm.monthly_portfolio_sorts(db, df, sort_cols, ncuts, 
                                                      smoothing)

    # == Output portfolios to disk == #
    Saver(disaster_portfolios, '../estimated_data/portfolios')
    print('Computed betas with respect to disaster risk factors ' +\
          'in %.2f minutes' %((time.time() - s) / 60))

        
    
##  -----------------------------------------------------------------------
##                             main program
##  -----------------------------------------------------------------------

if __name__ == "__main__": sys.exit(main(sys.argv))
            