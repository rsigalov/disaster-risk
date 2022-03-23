""" This is the docstring for crsp_comp.py.
This file contains a suite of functions that interact with the CRSP-COMPUSTAT
database on WRDS.

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
# import wrds # Replaced WRDS module with a plain PostgreSQL module psycopg2
import datetime
import time
import psycopg2
import os
from os.path import isfile, join
pd.options.display.max_columns = 20
from pandas.tseries.offsets import *

# Libraries to import and unpack .zip file from Ken French's website with
# 5 Fama-French factors
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

##  -----------------------------------------------------------------------
##                               functions
##  -----------------------------------------------------------------------


def connectToWRDS():
	# Setting up the connection to load CRSP and Compustat data (if was not loaded previously)
	with open("account_data/wrds_user.txt") as f:
		wrds_username = f.readline()

	with open("account_data/wrds_pass.txt") as f:
		wrds_password = f.readline()

	conn = psycopg2.connect(
		host="wrds-pgdata.wharton.upenn.edu",
		port = 9737,
		database="wrds",
		user=wrds_username,
		password=wrds_password)

	return conn

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


def balance_monthly_data(df):
    r'''
    Helper function for balancing panel of CRSP return data

    Args:
        df: Dataframe of CRSP returns

    Returns:
        balance_df: Dataframe containing permnos, years, and month for balanced
        panel

    '''

    # Get balanced panel
    df['date_eom'] = pd.to_datetime(df['date'])
    df.set_index('date_eom', inplace = True)
    bdf = df.groupby('permno').resample('M')['y','m'].last().reset_index()

    # Fill in missing years and months
    midx = pd.isnull(bdf.y)
    bdf.loc[midx, 'y'] = [x.year for x in bdf[midx].date_eom]
    bdf.loc[midx, 'm'] = [x.month for x in bdf[midx].date_eom]

    return pd.merge(bdf, df, on = ['permno', 'y', 'm'], how = 'left')

def compute_value_weights(p_df, scols):
    r'''
    Helper function to compute value weights for portfolio sorts on different
    characteristics.

    Args:
        p_df: Dataframe containing, for each PERMNO-Date, the portfolio bin that
              pertaining to a given characteristic. One of the columns must be:
              'permco_mktcap', which provides the basis for computing value
              weights. The other column must be titled 'date'
        scols: List of columns in p_df corresponding to characteristics
    '''

    for char in scols:
        bin_mktcap = p_df.groupby(['date', char])['permco_mktcap'].sum().\
                            reset_index().\
                            rename(columns = {'permco_mktcap': 'tot_' + char})

        p_df = pd.merge(p_df, bin_mktcap, on = ['date', char],
                        how = 'left')
        p_df[char + '_vw'] = p_df['permco_mktcap'] / p_df['tot_' + char]
        p_df.drop('tot_' + char, axis = 1, inplace = True)

    p_df.drop('permco_mktcap', 1, inplace = True)

    return p_df

def get_weekly_returns(conn, start_date, end_date, load_override=False):
    r'''
        Same as get_monthly_returns but loads weekly returns
    '''

    # Checking if the right CRSP file is in the working directory
    # and if it is simply load it without the need to go to WRDS:
    all_files = [f for f in os.listdir("data")]

    if ("crsp_week_ret.csv" in  all_files) and not load_override:
        df = pd.read_csv("data/crsp_week_ret.csv")
        # Convert all date columns to a proper format
        df["date"] = pd.to_datetime(df["date"])
        return df
    else:
        # == Ensure dates are in proper format == #
        sd = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        ed = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # Downloading daily CRSP returns:
        query = """
        with daily_df as (
            select 
                a.cusip, a.permno, a.permco, a.date, 
                date_trunc('week', a.date)::date as date_week,
                log(1+a.ret) as log_ret,
                abs(a.prc * a.shrout) as mktcap
            from crsp.dsf as a 
                join crsp.dsenames c on a.permno = c.permno 
            where a.ret is not null and prc is not null and 
                a.date >= c.namedt and a.date <= c.nameendt and 
                date >= '%s' and date <= '%s' and
                c.shrcd in (10, 11) and c.exchcd in (1, 2, 3)
        ), mkt_cap_rn as (
            select 
                cusip, permno, permco, date_week, mktcap,
                row_number() over (partition by cusip, permno, permco, date_week order by date desc) as rn
            from daily_df
        ), mkt_cap_last as(
            select
                cusip, permno, permco, date_week, mktcap
            from mkt_cap_rn
            where rn = 1
        ), week_ret as (
            select 
                cusip, permno, permco, date_week,
                sum(log_ret) as cum_log_ret
            from daily_df
            group by cusip, permno, permco, date_week
        )
        select 
            w.cusip, w.permno, w.permco, w.date_week as date, exp(w.cum_log_ret)-1 as ret, m.mktcap
        from week_ret as w
        left join mkt_cap_last as m
            on w.cusip = m.cusip and 
                w.permno = w.permno and 
                w.permco = m.permco and 
                w.date_week = m.date_week
            
        """ % (sd, ed)
        query = query.replace("\t", " ").replace("\n", " ")
        crspm_raw  = pd.read_sql_query(query, conn)
        crspm_raw["date"] = pd.to_datetime(crspm_raw["date"])

        # == For PERMCO-DATES with multiple PERMNOs, aggregate market cap and
        #    use PERMNO with largest mkt cap as "main" PERMNO for each PERMCO == #
        crspm_raw = crspm_raw.sort_values(['permco', 'date', 'mktcap'])
        crsp_mkt = crspm_raw.groupby(['permco', 'date'])['mktcap'].sum().reset_index()
        crspm_raw = crspm_raw.groupby(['permco', 'date']).last().reset_index()
        crspm_raw = pd.merge(
            crspm_raw, crsp_mkt, left_on = ['permco', 'date'], right_on = ['permco', 'date'])
        crspm_raw.drop('mktcap_x', 1, inplace = True)
        crspm_raw.rename(columns = {'mktcap_y': 'permco_mktcap'}, inplace = True)

        # == Deal with delisted returns == #
        query = """
        select 
            date_trunc('week', dlstdt)::date as dlstdt,
            permno, dlstcd, 
            case when dlret is null and dlstcd >= 400 and dlstcd <= 591 then -0.3 else dlret end as dlret
        from crsp.dsedelist
        """
        query = query.replace("\t", " ").replace("\n", " ")
        delist = pd.read_sql_query(query, conn)
        delist["dlstdt"] = pd.to_datetime(delist["dlstdt"])

        # Merge delisted returns with CRSP
        crspm_raw = pd.merge(crspm_raw,
                             delist[['permno','dlstdt', 'dlret']],
                             left_on = ['permno', 'date'],
                             right_on = ['permno', 'dlstdt'], how = 'left')
        idx = ((pd.isnull(crspm_raw.ret)) & (pd.isnull(crspm_raw.dlret)))

        # Total returns
        crspm_raw['ret_gross'] = crspm_raw['ret'] + 1
        crspm_raw['dlret_gross'] = crspm_raw['dlret'] + 1
        crspm_raw['ret_adj'] = crspm_raw[['ret_gross']].fillna(1).multiply(\
                               crspm_raw['dlret_gross'].fillna(1),
                               axis = 'index') - 1
        crspm_raw.loc[idx,'ret_adj'] = np.nan     

        # == Drop old returns and rename adjusted return columns
        crspm_raw.drop(labels = ['ret_gross','dlret_gross','dlstdt','dlret', 'ret'], axis = 1, inplace = True)
        crspm_raw.rename(columns = {'ret_adj': 'ret'}, inplace = True)

        crspm_raw.to_csv("data/crsp_week_ret.csv", index = False)

        return crspm_raw


def get_monthly_returns(conn, start_date, end_date, balanced = True, load_override=False):
    r'''
    Function to download monthly returns from CRSP and then compute delisted
    returns following Shumway (1997). Observations are at the PERMNO-month
    level. For PERMCO-months with multiple PERMNOS, we take the largest PERMNO
    available on that date.

    Args:
        conn: Database connection using wrds module
        start_date: String of start date for returns. Default to minimum
        end_date: String of end date for returns. Default to maximum
        balanced: Boolean indicating whether to balance the panel for each
                  PERMNO

    Returns:
        df: Dataframe with the data, after cleaning
    '''

    # Checking if the right CRSP file is in the working directory
    # and if it is simply load it without the need to go to WRDS:
    all_files = [f for f in os.listdir("data")]

    if ("crsp_ret.csv" in all_files) and not load_override:
        df = pd.read_csv("data/crsp_ret.csv")
        # Convert all date columns to a proper format
        df["date"] = pd.to_datetime(df["date"])
        df["date_eom"] = pd.to_datetime(df["date_eom"])
        return df
    else:
        # == Ensure dates are in proper format == #
        sd = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        ed = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # == Download CRSP Monthly == #
        query = "select a.cusip, a.permno, a.permco, a.date, " +\
                "a.ret, a.retx, a.prc, a.shrout, c.hsiccd, " +\
                "abs(a.prc * a.shrout) as mktcap, " +\
                "extract(MONTH from a.date) as m, " +\
                "extract(YEAR from a.date) as y " +\
                "from crsp.msf as a " +\
                "join crsp.dsenames c " +\
                "on a.permno = c.permno " +\
                "where a.ret is not null and prc is not null and " +\
                "a.date >= c.namedt and a.date <= c.nameendt and " +\
                "date >= '%s' and date <= '%s' and " %(sd, ed) +\
                "c.shrcd in (10, 11) and c.exchcd in (1, 2, 3)"
        crspm_raw = pd.read_sql_query(query, conn)
        crspm_raw.loc[:, 'date'] = pd.to_datetime(crspm_raw.loc[:, 'date'])

        # == For PERMCO-DATES with multiple PERMNOs, aggregate market cap and
        #    use PERMNO with largest mkt cap as "main" PERMNO for each PERMCO == #
        crspm_raw = crspm_raw.sort_values(['permco', 'date', 'mktcap'])
        crsp_mkt = crspm_raw.groupby(['permco', 'date'])['mktcap'].sum().\
                                                                      reset_index()
        crspm_raw = crspm_raw.groupby(['permco', 'date']).last().reset_index()
        crspm_raw = pd.merge(crspm_raw, crsp_mkt, left_on = ['permco', 'date'],
                                        right_on = ['permco', 'date'])
        crspm_raw.drop('mktcap_x', 1, inplace = True)
        crspm_raw.rename(columns = {'mktcap_y': 'permco_mktcap'}, inplace = True)

        # == Deal with delisted returns == #
        query = "select *, extract(MONTH from dlstdt) as m, " +\
                "extract(YEAR from dlstdt) as y from crsp.msedelist"
        delist = pd.read_sql_query(query, conn)

        # Follow Shumway (1997) by filling in missing delisted returns with
        # delist codes 400-591 as -30%
        shumway_idx = ((pd.isnull(delist.dlret)) & (delist.dlstcd >= 400) & \
                       (delist.dlstcd <= 591))
        delist[shumway_idx] = -0.3

        # Merge delisted returns with CRSP
        crspm_raw = pd.merge(crspm_raw,
                             delist[['permno', 'y', 'm', 'dlret', 'dlretx']],
                             left_on = ['permno', 'y', 'm'],
                             right_on = ['permno', 'y', 'm'], how = 'left')
        idx = ((pd.isnull(crspm_raw.ret)) & (pd.isnull(crspm_raw.dlret)))

        # Total returns
        crspm_raw['ret_gross'] = crspm_raw['ret'] + 1
        crspm_raw['dlret_gross'] = crspm_raw['dlret'] + 1
        crspm_raw['ret_adj'] = crspm_raw[['ret_gross']].fillna(1).multiply(\
                               crspm_raw['dlret_gross'].fillna(1),
                               axis = 'index') - 1
        crspm_raw.loc[idx,'ret_adj'] = np.nan            #Entries with no ret/dlret

        # Ex-Dividend Returns
        crspm_raw['retx_gross'] = crspm_raw['retx'] + 1
        crspm_raw['dlretx_gross'] = crspm_raw['dlretx'] + 1
        crspm_raw['retx_adj'] = crspm_raw[['retx_gross']].fillna(1).multiply(\
                               crspm_raw['dlretx_gross'].fillna(1),
                               axis = 'index') - 1
        crspm_raw.loc[idx, 'retx_adj'] = np.nan          #Entries with no ret/dlret

        # == Drop old returns and rename adjusted return columns
        crspm_raw.drop(labels = ['ret', 'retx', 'dlret', 'dlretx', 'ret_gross',
                                 'dlret_gross', 'retx_gross', 'dlretx_gross'],
                       axis = 1, inplace = True)
        crspm_raw.rename(columns = {'ret_adj': 'ret',
                                    'retx_adj': 'retx'},
                        inplace = True)

        # == Return panel == #
        if balanced:
            # Balanced panel for permno-month == #
            df = balance_monthly_data(crspm_raw)
        else:
            df = crspm_raw

        df.to_csv("data/crsp_ret.csv", index=False)

        return df

def monthly_portfolio_sorts(conn, df, sort_cols, ncuts, smoothing = 6):
    r'''
    Function to conduct monthly portfolio sorts based on PERMNO-Month
    characteristics. The function value-weights and equal-weights, and for both,
    only includes PERMNO-dates where value-weighted is possible.

    Args:
        conn: WRDS database connection
        df: Dataframe containing date, PERMNO, and characteristics. If
        multiple characteristics are required, does portfolio sorts on each.
        sort_cols: List of columns in df containing characteristics
        ncuts: Integer for how many bins to create for the portfolio sorts
        smoothing: Integer for window to smooth market capitalizations for
                   book-to-markets

    Returns:
        A dictionary where the keys are the variable that was sorted on
        and the values are dataframes containing the portfolio returns, as well
        as the number of total PERMNOs for each month. In all cases,
        the portfolio number is increasing in characteristic value (e.g.,
        Portfolio 1 is the firms with the lowest characteristic)

    '''

    # == Get CRSP Monthly Data, including delisted returns == #
    sd = df['date'].min() -  MonthEnd(smoothing + 1)
    ed = df['date'].max()
    crsp = get_monthly_returns(conn, sd, ed, balanced = False)

    # Ensure CRSP data is at the end of each month for mergers later
    crsp.loc[:, 'date'] = crsp.loc[:, 'date'] + MonthEnd(0)

    # == Lag characteristics for portfolio sorts so that returns in month m
    #    are based on characteristics in month m - 1 #
    crsp['form_date'] = crsp['date'] - MonthEnd()

    # == Get associated valuation ratios for each date in CRSP == #
    crsp = get_bm_ratios(conn, crsp, smoothing, comp_lag = 3)

    # == Compute cuts == #
    psorts = df.groupby('date')[sort_cols].transform(\
                                        lambda q: pd.qcut(q, ncuts,
                                        labels = np.arange(1, ncuts + 1)))

    # Merge back PERMNO-date information
    # print(psorts.head())
    # print(df[['date', 'permno']].head())
    psorts = psorts.join(df[['date', 'permno']])

    # == Get value-weights and book-to-markets  == #
    psorts = pd.merge(psorts, crsp[['date', 'permno', 'permco_mktcap', 'bm', 'op']],
                      left_on = ['date', 'permno'],
                      right_on = ['date', 'permno'],
                      how = 'inner')

    # # Compute value-weights in each bin for each characteristic
    # psorts = compute_value_weights(psorts, sort_cols)

    # == Merge with returns and compute portfolio returns == #
    crsp_char = pd.merge(crsp, psorts, left_on = ['permno', 'form_date'],
                         right_on = ['permno', 'date'],
                         how = 'inner')
    crsp_char.rename(columns = {'date_x': 'date',
                                'bm_y': 'bm',
                                'op_y': 'op',
                                'permco_mktcap_y': "permco_mktcap"}, inplace = True)
    crsp_char.drop(['date_y', 'bm_x', 'op_x', "permco_mktcap_x"], 1, inplace = True)

    # == Assemble portfolios == #
    portfolios = assemble_portfolios(crsp_char, sort_cols, ncuts)

    return portfolios

def assemble_portfolios(df, scols, ncuts):
    r'''
    Compute average realized portfolio returns and average book-to-market
    ratios at formation

    Args:
        df: Dataframe of PERMNOS-Date-Characteristic bins, plus returns,
            book-to-market and operating profitability
        scols: Columns in the dataframe containing sorting characteristics
        ncuts: Number of portfolios

    Returns:
        storage_dict: Dictionary whose keys are characteristics. Each
                      characteristics is associated with:
                            1. Realized portfolio returns
                            2. Average book-to-markets at formation
                            3. Portfolio constituents at formation

    '''
    storage_dict = {}

    for char in scols:

        #Initialize dictionary entries
        storage_dict[char] = {}

        # Compute EW returns and counts, then join
        ew_ret = calc_ew_char_ports(df, char, "ret", ncuts)
        vw_ret = calc_vw_char_ports(df, char, "ret", ncuts)
        storage_dict[char]['ret'] = ew_ret.join(vw_ret)

        # Compute EW returns and counts, then join
        ew_bm = calc_ew_char_ports(df, char, "bm", ncuts)
        vw_bm = calc_vw_char_ports(df, char, "bm", ncuts)
        storage_dict[char]['bm'] = ew_bm.join(vw_bm)

        # Compute EW returns and counts, then join
        ew_op = calc_ew_char_ports(df, char, "op", ncuts)
        vw_op = calc_vw_char_ports(df, char, "op", ncuts)
        storage_dict[char]['op'] = ew_op.join(vw_op)

        # Store portfolio constituents
        storage_dict[char]['constituents'] = \
                                      df[['form_date', 'permno', char]].dropna()

    return storage_dict

def calc_ew_char_ports(df, char, variable, ncuts):
    ew = df.groupby(['date', char])[variable].mean().reset_index().\
                    pivot(index = 'date', columns = char, values = variable)
    ew_count = df.groupby(['date']).\
                apply(lambda x: len(x[[char, variable]].dropna()))
    ew_count.name = 'ew_count'
    ew = pd.DataFrame(ew_count).join(ew)
    ew.rename(columns = dict([(x, 'ew_' + str(x) ) for x in \
                              np.arange(1, ncuts + 1)]),
             inplace = True)
    return ew

def calc_vw_char_ports(df, char, variable, ncuts):
    vw = df.groupby(['date', char]).apply(lambda x: wavg(x, variable, "permco_mktcap")).\
                        rename("ret_vw").\
                        reset_index()
    vw = pd.pivot_table(vw, index = "date", columns = char, values = "ret_vw")
    vw_count = df.groupby(['date']).\
                    apply(lambda x: len(x[variable].dropna()))
    vw_count.name = "vw_count"
    vw.rename(columns = dict([(x, 'vw_' + str(x) ) for x in \
                                  np.arange(1, ncuts + 1)]),
              inplace = True)
    vw = pd.merge(vw, vw_count, left_index = True, right_index = True)
    return vw

def get_bm_ratios(conn, crsp_df, smoothing, comp_lag = 3, load_override=False):
    r'''
    Takes panel of CRSP returns and market valuations and computes
    book-to-market ratios using COMPUSTAT data, with a user-defined
    input to determine how much to lag accounting data from market data.
    Book equity comes from COMPUSTAT Quarterly, Annual, and then from
    Fama, French, and Davis (2000), in that order.

    Args:
        conn: WRDS database connection
        crsp_df: Dataframe of CRSP data
        smoothing: Window to smooth market capitalizations for smooth BMs
        comp_lag: Integer for desired lag (months) in accounting data.
                  Default = 3.

    Returns:
        crsp_df: Original dataframe, augmented with book-to-market ratio, both
                 using current market cap and smoothed market cap
    '''

    # == Get COMPUSTAT Annual and Quarterly == #
    # First checking if there is saved data with the same lag in the
    # working directory:
    all_files = [f for f in os.listdir("data")]
    name_crsp_ccm = f"crsp_ccm_{comp_lag}.csv"

    if (name_crsp_ccm in all_files) and not load_override:
        crsp_ccm = pd.read_csv(f"data/crsp_ccm_{comp_lag}.csv")
        crsp_ccm["date"] = pd.to_datetime(crsp_ccm["date"])
        crsp_ccm["form_date"] = pd.to_datetime(crsp_ccm["form_date"])
        return crsp_ccm
    else:
        compa = get_comp_ann(conn, comp_lag)
        compq = get_comp_qtr(conn, comp_lag)

        # == Davis, Fama, and French  (2000) book equity == #
        dff_be = pd.read_csv('data/DFF_BE.csv')
        dff_be['dff_known_date_dt'] = [datetime.datetime(x, 6, 30) for \
                                       x in dff_be.YEAR.values]
        dff_be.drop('YEAR', 1, inplace = True)
        dff_be.rename(columns = {'BE': 'DFF_BE'}, inplace = True)

        # == For each PERMNO-Date, get the latest book value from each source == #
        crsp_ccm = get_latest_accounting(crsp_df, compa,
                                         acct_date = 'known_date_ann',
                                         acct_permno = 'lpermno')
        crsp_ccm = get_latest_accounting(crsp_ccm, compq,
                                         acct_date = 'known_date_qtr',
                                         acct_permno = 'lpermno')
        crsp_ccm = get_latest_accounting(crsp_ccm, dff_be,
                                         acct_date = 'dff_known_date_dt',
                                         acct_permno = 'PERMNO')

        # == Use COMPQ, COMPA, then Fama-French-Davis book value == #
        be_star = np.where(pd.isnull(crsp_ccm['beq']), crsp_ccm['be'], crsp_ccm['beq'])
        be_star = np.where(pd.isnull(be_star), crsp_ccm['DFF_BE'], be_star)
        crsp_ccm['be_star'] = be_star
        crsp_ccm.loc[crsp_ccm.be_star < 0, :] = np.nan

        crsp_ccm.drop(columns = ['be', 'beq', 'DFF_BE', 'dff_known_date_dt',
                                 'known_date_qtr', 'known_date_ann'],
                      axis = 1, inplace = True)

        # == Compute rolling market capitalization. Note that this ignores
        #    unbalanced panels and just uses last # of observations == #
        rolling_mkt = crsp_ccm.groupby(['permno'])['permco_mktcap'].\
                                apply(lambda x: x.rolling(window = smoothing,
                                    min_periods = smoothing).mean())
        rolling_mkt.name = 'rolling_mkt'
        crsp_ccm = crsp_ccm.join(rolling_mkt)

        # == Compute book-to-market == #
        crsp_ccm['bm'] = crsp_ccm['be_star'] * 1000 / crsp_ccm['rolling_mkt']
        crsp_ccm.drop('be_star', 1, inplace = True)

        # Computing latest available operating profitability:
        crsp_ccm["op"] = np.where(pd.isnull(crsp_ccm['opq']), crsp_ccm['op'], crsp_ccm['opq'])
        crsp_ccm.drop(columns = "opq", inplace = True)

        # Computing latest available YoY asset growth
        crsp_ccm["at_growth"] = np.where(pd.isnull(crsp_ccm['atq_growth']), crsp_ccm['at_growth'], crsp_ccm['atq_growth'])
        crsp_ccm.drop(columns = "atq_growth", inplace = True)

        # Saving financials data for future use:
        crsp_ccm.to_csv(f"data/crsp_ccm_{comp_lag}.csv", index=False)

        return crsp_ccm

def get_latest_accounting(crsp_df, acct_df, acct_date, acct_permno):
    r'''
    Function to get, for each PERMNO-Date, the latest accounting data
    contained in be_df.

    Args:
        crsp_df: Dataframe with CRSP data. We assume PERMNOS are contained in
                 the variable 'permnos' and date column 'date'
        acct_df: Dataframe with the accounting data. It must have column for
                 PERMNOS and dates
        acct_date: Column indicating date column in accounting dataframe to use
                   when finding latest accounting data
        acct_permno: Column indicating PERMNO column in accounting dataframe
    '''

    # == Do some setup == #
    crsp_date = 'date'
    crsp_permno = 'permno'
    original_length = len(crsp_df)

    # Rename accounting PERMNO column to same as in CRSP.
    acct_df.rename(columns = {acct_permno: crsp_permno}, inplace = True)

    # == Get latest accounting data == #
    ccm = pd.merge(crsp_df[[crsp_permno, crsp_date]], acct_df,
                   left_on = [crsp_permno],
                   right_on = [crsp_permno],
                   how = 'left')
    ccm = ccm[ccm[acct_date] <= ccm[crsp_date]].\
                sort_values([crsp_permno, crsp_date])
    ccm = ccm.groupby([crsp_permno, crsp_date]).last().reset_index()

    # Ensure no stale data
    ddiff = (ccm[crsp_date] - ccm[acct_date]) / pd.offsets.Day(1)
    #ccm = ccm[ddiff.astype(int) <= 366]      #Avoids stale

    # == Check to ensure each PERMNO-Date pair is unique and no lost data  == #
    if len(ccm) != len(ccm.drop_duplicates([crsp_permno, crsp_date])):
        raise ValueError("Duplicate PERMNO-Date Pairs")

    # == Return with original data == #
    return pd.merge(crsp_df, ccm, on = [crsp_permno, crsp_date], how = 'left')

def link_crsp_compustat(conn, comp_df, datate_col):
    r'''
    Given a dataset of COMPUSTAT accounting data (Quarterly or Annual), get
    associated PERMNO using CRSP-COMPUSTAT linking table

    Args:
        conn: WRDS database object
        comp_df: Dataframe of Compustat data. Must have a 'gvkey' column
        datadate_col: Column in COMPUSTAT data that has datadate column

     Output:
        comp_df: The input dataframe, augmented with PERMNOs (lpermno)
    '''

    # == Get linking table = =#
    query = "select * from crsp.ccmxpf_linktable where " +\
            "linktype in ('LU','LC','LD','LF','LN','LO','LS'," +\
            "'LX') and USEDFLAG = 1"
    link_table = pd.read_sql_query(query, conn)
    link_table['linkdt_dt'] = pd.to_datetime(link_table['linkdt'])
    link_table['linkenddt_dt'] = pd.to_datetime(link_table['linkenddt'])

    # == Merge == #
    comp_df = pd.merge(comp_df, link_table, left_on = ['gvkey'],
                                 right_on = ['gvkey'], how = 'inner')

    dt_check = ((comp_df.linkdt_dt <= comp_df[datate_col]) | \
                 pd.isnull(comp_df.linkdt_dt)) & \
               ((comp_df[datate_col] <= comp_df.linkenddt_dt) \
               | pd.isnull(comp_df.linkenddt_dt))

    comp_df = comp_df[dt_check]

    # == For permno-date pairs with multiple obs, take primary link == #
    comp_df = comp_df.sort_values(['lpermno', datate_col, 'linkprim'])
    comp_df = comp_df.groupby(['lpermno', datate_col,
                               'linkprim']).last().reset_index()
    comp_df.drop(['linkprim', 'liid', 'linktype', 'usedflag', 'linkdt',
                  'linkenddt', 'linkdt_dt', 'linkenddt_dt'], 1, inplace = True)

    return comp_df.drop_duplicates(subset = ['lpermno', datate_col])

def get_comp_qtr(conn, comp_lag):
    r'''
    Gets data for book values from COMPUSTAT quarterly

    Args:
        conn: WRDS database connection
        comp_lag: Integer for desired lag (months) in accounting data.
                  Default = 3.

    Returns:
        comp_qtr: Dataframe with gvkey-date-accounting info
    '''
    query = """
            select
                gvkey, datadate as datadate_qtr,
                fyearq, fyr, fqtr, atq, txditcq, ceqq, pstkq, seqq, ltq, dlttq,
                saleq as sale,
                lag(atq, 4) over (partition by gvkey order by datadate) as lag_atq,
                coalesce(cogsq, 0) as cogs,
                coalesce(xsgaq, 0) as xsga,
                coalesce(xintq, 0) as xint,
                case when cogsq is not null then 1 else 0 end +
                    case when xsgaq is not null then 1 else 0 end +
                    case when xintq is not null then 1 else 0 end as cnt_op_non_zero
            from compm.fundq
            where
                datafmt = 'STD' and
                indfmt = 'INDL' and
                popsrc = 'D' and
                consol = 'C'
        """
    query = query.replace("\t", " ").replace("\n", " ")
    comp_qtr_raw = pd.read_sql_query(query, conn)
    comp_qtr_raw["datadate_qtr"] = pd.to_datetime(comp_qtr_raw["datadate_qtr"])

    # Set known date and beginning of fiscal year
    comp_qtr_raw["known_date_qtr"] = comp_qtr_raw["datadate_qtr"] + \
                                                              MonthEnd(comp_lag)
    comp_qtr_raw["begfy"] = comp_qtr_raw["datadate_qtr"] - MonthBegin(12)


    # == For gvkeys with multiple obs on a datadate, take the one with the
    # latest fyearq.  == #
    comp_qtr_raw = comp_qtr_raw.sort_values(['gvkey', 'datadate_qtr', 'fyearq'])
    comp_qtr = comp_qtr_raw.drop_duplicates(subset = ['gvkey', 'datadate_qtr'],
                                            keep = 'last')

    # == Drop firms who do not have two years of data (FF,JFE, 1993) == #
    gvkey_count = comp_qtr.groupby('gvkey').cumcount() + 1
    gvkey_count.name = 'count'
    comp_qtr = comp_qtr.join(gvkey_count)
    comp_qtr = comp_qtr[comp_qtr['count'] > 8]

    # == Compute book value of equity == #
    comp_qtr['psq'] = comp_qtr['pstkq']
    comp_qtr['psq'].fillna(0, inplace = True)
    comp_qtr['txditcq'].fillna(0, inplace = True)

    comp_qtr['ceq_psq'] = comp_qtr['ceqq'] + comp_qtr['psq']
    comp_qtr['at_ltq'] = comp_qtr['atq'] - comp_qtr['ltq']

    seq = np.where(pd.isnull(comp_qtr["seqq"]), comp_qtr["ceq_psq"],
                   comp_qtr["seqq"])
    seq = np.where(pd.isnull(seq), comp_qtr["at_ltq"], seq)

    comp_qtr['beq'] = seq - comp_qtr.psq + comp_qtr.txditcq

    comp_qtr.dropna(subset = ['beq'], inplace = True)

    # Using book equity compute operating profitability as
    # (SALE - COGS - XINT - XSGA)/BE if satisfies the filters
    comp_qtr["opq"] = np.where(
        (comp_qtr["beq"] > 0) & (comp_qtr["sale"] > 0) & (comp_qtr["cnt_op_non_zero"]),
        (comp_qtr["sale"] - comp_qtr["cogs"] - comp_qtr["xint"] - comp_qtr["xsga"])/comp_qtr["beq"],
        np.nan)
    comp_qtr.drop(columns = ["sale", "cogs", "xint", "xsga", "cnt_op_non_zero"], inplace = True)

    # Calculating asset growth rate
    comp_qtr["atq_growth"] = np.where(
        comp_qtr["lag_atq"] > 0, (comp_qtr["atq"]-comp_qtr["lag_atq"])/comp_qtr["lag_atq"], np.nan)

    # == Add PERMNOS for CRSP link == #
    comp_qtr = link_crsp_compustat(conn, comp_qtr, 'datadate_qtr')

    # # Saving file with quarterly compustat so that we don't need
    # # to download the data everytime we run this
    # comp_qtr.to_csv("comp_qtr.csv", index = False)

    return comp_qtr[['lpermno','known_date_qtr', 'beq', 'opq', "atq_growth"]]

def get_comp_ann(conn, comp_lag):
    r'''
    Gets data for book values from COMPUSTAT annual

    Args:
        conn: WRDS database connection
        comp_lag: Integer for desired lag (months) in accounting data.
                  Default = 3.

    Returns:
        comp_ann: Dataframe with gvkey-date-accounting info

    '''

    # == Get raw data == #
    query = """
            select
                gvkey, datadate as datadate_ann,
                fyear, fyr, at, pstkl, txditc, pstkrv, ceq, pstk, seq, lt,
                sale,
                lag(at) over (partition by gvkey order by datadate) as lag_at,
                coalesce(cogs, 0) as cogs,
                coalesce(xsga, 0) as xsga,
                coalesce(xint, 0) as xint,
                extract(YEAR from datadate) as year,
                case when cogs is not null then 1 else 0 end +
                    case when xsga is not null then 1 else 0 end +
                    case when xint is not null then 1 else 0 end as cnt_op_non_zero
            from
                compm.funda
            where
                datafmt = 'STD' and
                indfmt = 'INDL' and
                popsrc = 'D' and
                consol = 'C'
        """
    query = query.replace("\t", " ").replace("\n", " ")
    comp_ann_raw = pd.read_sql_query(query, conn)
    comp_ann_raw["datadate_ann"] = pd.to_datetime(comp_ann_raw["datadate_ann"])

    # Set known date and beginning of fiscal year
    comp_ann_raw["known_date_ann"] = comp_ann_raw["datadate_ann"] + MonthEnd(comp_lag)
    comp_ann_raw["begfy"] = comp_ann_raw["datadate_ann"] - MonthBegin(12)

    # == For gvkeys with multiple obs in a calendar year, take last one.
    # This is for firms who change fiscal-year end within the year == #
    comp_ann_raw = comp_ann_raw.sort_values(['gvkey', 'year', 'datadate_ann'])
    comp_ann = comp_ann_raw.drop_duplicates(subset = ['gvkey', 'year'],
                                            keep = 'last')

    # == Drop firms who do not have two years of data (FF,JFE, 1993) == #
    gvkey_count = comp_ann.groupby('gvkey').cumcount() + 1
    gvkey_count.name = 'count'
    comp_ann = comp_ann.join(gvkey_count)
    comp_ann = comp_ann[comp_ann['count'] > 2]

    # == Compute book equity.  Incorporate Preferred Stock (PS) values use
    # == the redemption value of PS, or the liquidation value the par value
    # == (in that order) (FF, JFE, 1993, p. 8).  Use Balance Sheet Deferred
    # == Taxes TXDITC if available  == #

    comp_ann['ps'] = np.where(comp_ann['pstkrv'].isnull(), comp_ann['pstkl'],
                              comp_ann['pstkrv'])
    comp_ann['ps'] = np.where(comp_ann['ps'].isnull(), comp_ann['pstk'],
                              comp_ann['ps'])
    comp_ann['ps'] = np.where(comp_ann['ps'].isnull(), 0, comp_ann['ps'])

    comp_ann['txditc'].fillna(0, inplace = True)

    #Now compute book equity
    comp_ann['ceq_ps'] = comp_ann['ceq'] + comp_ann['ps']
    comp_ann['at_lt'] = comp_ann['at'] - comp_ann['lt']

    se = np.where(pd.isnull(comp_ann["seq"]), comp_ann["ceq_ps"],
                  comp_ann["seq"])
    se = np.where(pd.isnull(se), comp_ann["at_lt"], se)

    comp_ann['be'] = se - comp_ann.ps + comp_ann.txditc

    comp_ann.dropna(subset = ['be'], inplace = True)

    # Using book equity compute operating profitability as
    # (SALE - COGS - XINT - XSGA)/BE if satisfies the filters
    comp_ann["op"] = np.where(
        (comp_ann["be"] > 0) & (comp_ann["sale"] > 0) & (comp_ann["cnt_op_non_zero"]),
        (comp_ann["sale"] - comp_ann["cogs"] - comp_ann["xint"] - comp_ann["xsga"])/comp_ann["be"],
        np.nan)
    comp_ann.drop(columns = ["sale", "cogs", "xint", "xsga", "cnt_op_non_zero"], inplace = True)

    comp_ann["at_growth"] = np.where(
        comp_ann["lag_at"] > 0, (comp_ann["at"]-comp_ann["lag_at"])/comp_ann["lag_at"], np.nan)

    # == Add PERMNOS for CRSP link == #
    comp_ann = link_crsp_compustat(conn, comp_ann, 'datadate_ann')

    return comp_ann[['lpermno', 'known_date_ann', 'be', 'op', "at_growth"]]


def load_FF():
    r'''
        Gets 5 Fama-French factors directly from Ken French's website

        Returns:
            ff: dataframe with 5 Fama-French factors and date as index.
                Returns are already divided by a hundred (originally)
    '''

    resp = urlopen("http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    f = open("ff.csv", "w")

    indicator = False
    counter = 0
    for line in zipfile.open(zipfile.namelist()[0]).readlines():
        line_text = line.decode('utf-8')

        if line_text[0] == ",":
            indicator = True

        if line_text[0] == "\r":
            indicator = False
            counter += 1

        if not indicator and counter == 2:
            break

        if indicator:
            f.write(line_text)

    f.close()

    ff = pd.read_csv("ff.csv")
    ff = ff.rename({ff.columns[0]: "date"}, axis = 1)
    ff = ff.rename({"Mkt-RF": "MKT"}, axis = 1)
    ff["date"] = pd.date_range(start = "1963-07-01", freq = "M", periods = ff.shape[0])
    ff = ff.set_index("date")/100

    return ff

def get_ff_ind(conn, df):
    '''
    This function assigns FF 12 industry to permnos in the data frame:

    Inputs:
        conn: connection to WRDS to get MSENAMES table with SIC code
        df: dataframe with 'form_date' and 'permno' columns to merge with
        MSENAMES to get appropriate SIC code

    Outputs:
        df_ind: same dataframe as df but with a type of industry column

    Requirements:
        ff_12_ind.csv: file with ff 12 start and end SIC
    '''

    query = """
        select permno, siccd, namedt, nameendt
        from crsp.msenames
    """
    query = query.replace("\n","").replace("\t","")
    crsp_sic = pd.read_sql_query(query, conn)
    crsp_sic["namedt"] = pd.to_datetime(crsp_sic["namedt"])
    crsp_sic["nameendt"] = pd.to_datetime(crsp_sic["nameendt"])

    df_ind = pd.merge(df, crsp_sic, on = "permno")
    df_ind = df_ind[(df_ind.form_date >= df_ind.namedt) & (df_ind.form_date <= df_ind.nameendt)]
    df_ind = df_ind.drop(columns = ["namedt", "nameendt"])

    def find_between(x, x1, x2, y, y_def):
        ind_between = (x >= x1) & (x <= x2)
        if sum(ind_between) == 0:
            return y_def
        else:
            return y[ind_between][0]

    # Loading data on FF industries:
    ff_ind = pd.read_csv("data/ff_12_ind.csv")

    df_ind["ff_ind"] = df_ind["siccd"].apply(
        lambda x: find_between(
            x, np.array(ff_ind.start_sic), np.array(ff_ind.end_sic), np.array(ff_ind.ind_desc), "Other"))

    return df_ind


# def load_and_filter_ind_disaster(days, min_obs_in_month, min_share_month, suffix = None):

#     ########################################################################
#     # Loading interpolated measures according to the specified number of days
#     # of interpolation
#     if suffix is None:
#         file_name = "estimated_data/interpolated_D/int_ind_disaster_days_" + str(days) + ".csv"
#     else:
#         file_name = "estimated_data/interpolated_D/int_ind_disaster_" + suffix + "_days_" + str(days) + ".csv"

#     D_df = pd.read_csv(file_name)

#     # Dealing with dates:
#     D_df["date"] = pd.to_datetime(D_df["date"])
#     D_df["date_eom"] = D_df["date"] + pd.offsets.MonthEnd(0)
#     D_df = D_df.drop("date", axis = 1)

#     ########################################################################
#     # Limiting to companies with at least 15 observations in a month in at least 80%
#     # months in the sample from January 1996 to December 2017.
#     def min_month_obs(x):
#         return x["D_clamp"].count() > min_obs_in_month

#     D_filter_1 = D_df.groupby(["secid", "date_eom"]).filter(min_month_obs)
#     D_mon_mean = D_filter_1.groupby(["secid", "date_eom"]).mean().reset_index()

#     num_months = len(np.unique(D_mon_mean["date_eom"]))
#     def min_sample_obs(x):
#         return x["D_clamp"].count() > num_months * min_share_month

#     D_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)
#     D_filter = D_filter.rename(columns = {"date_eom": "date"})

#     return D_filter
