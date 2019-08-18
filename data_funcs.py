import pandas as pd
import numpy as np
import wrds

def get_crsp_permno(df):
    r'''
        This function takes in a datafame df that should have a secid
        and date columns (and have potentially other columns) and
        uses WRDS OM/CRSP linking table to assign an appropriate
        PERMNO to each (secid, date) observation. Returns the same
        dataframe with additional PERMNO column
    '''

    # Loading WRDS OM/CRSP linking table
    om_crsp = pd.read_csv("om_crsp_wrds_linking_table.csv").rename(columns = {"PERMNO":"permno"})
    om_crsp["sdate"] = pd.to_datetime(om_crsp["sdate"], format = "%Y%m%d")
    om_crsp["edate"] = pd.to_datetime(om_crsp["edate"], format = "%Y%m%d")

    # Merging OM dataset with PERMNO
    df = pd.merge(df, om_crsp, on = "secid", how = "left")
    df = df[(df.date <= df.edate) & (df.date >= df.sdate)]
    df = df.drop(columns = ["sdate", "edate", "score"])

    return df

def get_cs_indicator(df):
    r'''
    This fuction assigns a common share indicator that is equal to
    1 when the (secid, date) is classified as Common Share by either
    OptionMetrics (issue_type = "0") or by CRSP (SHRCD in [10,11]
    and EXCHCD in [1,2,3]).

    Args:
        df: dataframe that has secid and date columns (potentially
        has other columns)

    Returns:
        df: Same dataframe with added common share indicator

    Requirement:
        om_crsp_wrds_linking_table.csv: a file with OM/CRSP linking
        table should be in the working directory
    '''

    # Setting up a connection with WRDS PostgreSQL server
    db = wrds.Connection()

    # Loading data on CRSP classification
    query = """
    select permno, namedt, nameendt, shrcd, exchcd
    from crsp.msenames
    """.replace('\n', ' ').replace('\t', ' ')
    df_crsp_names = db.raw_sql(query, date_cols = ["namedt", "nameendt"])

    # Loading issue type from OptionMetrics:
    query = """
    select secid, issue_type
    from OPTIONM.SECURD
    """.replace('\n', ' ').replace('\t', ' ')
    df_issue_type = db.raw_sql(query)

    # Loading WRDS OM/CRSP linking table
    om_crsp = pd.read_csv("om_crsp_wrds_linking_table.csv").rename(columns = {"PERMNO":"permno"})
    om_crsp["sdate"] = pd.to_datetime(om_crsp["sdate"], format = "%Y%m%d")
    om_crsp["edate"] = pd.to_datetime(om_crsp["edate"], format = "%Y%m%d")

    # (1) For each (secid, date) getting the best link from OM-CRSP linking table:
    df_crsp_merge = pd.merge(df[["secid", "date"]], om_crsp, on = "secid", how = "left")
    df_crsp_merge = df_crsp_merge[
        (df_crsp_merge.date <= df_crsp_merge.edate) &
        (df_crsp_merge.date >= df_crsp_merge.sdate)]
    df_crsp_merge = df_crsp_merge.drop(columns = ["sdate", "edate", "score"])

    # (2) Now linking with CRSP MSENAMES by PERMNO and filtering by
    # shrcd and exchcd
    df_crsp_merge = pd.merge(df_crsp_merge, df_crsp_names, on = "permno", how = "left")
    df_crsp_merge = df_crsp_merge[
        (df_crsp_merge.date >= df_crsp_merge.namedt) &
        (df_crsp_merge.date <= df_crsp_merge.nameendt)]
    df_crsp_merge = df_crsp_merge[
        df_crsp_merge.shrcd.isin([10,11]) &
        df_crsp_merge.exchcd.isin([1,2,3])]
    df_crsp_merge["crsp_cs"] = 1
    df_crsp_merge = df_crsp_merge.drop(columns = ["namedt", "nameendt", "shrcd", "exchcd"])

    # (3) Merging with OM issue_types
    df = pd.merge(df, df_issue_type, on = "secid", how = "left")

    # (4) Merging with common share observations from CRSP:
    df = pd.merge(df, df_crsp_merge, on = ["secid", "date"], how = "left")

    # (5) Setting union indicator:
    df["cs"] = np.where((df["issue_type"] == "0") | (df["crsp_cs"] == 1), 1, 0)
    df = df.drop(columns = ["crsp_cs", "issue_type"])

    return df

def mean_with_truncation(x):
    return np.mean(x[(x <= np.nanquantile(x, 0.975)) & (x >= np.nanquantile(x, 0.025))])

def filter_ind_disaster(df, variable, min_obs_in_month, min_share_month):
    def min_month_obs(x):
        return x[variable].count() >= min_obs_in_month

    def min_sample_obs(x):
        return x[variable].count() > num_months * min_share_month

    df["date"] = pd.to_datetime(df["date"])
    df["date_eom"] = df["date"] + pd.offsets.MonthEnd(0)

    df_filter_1 = df.groupby(["secid", "date_eom"]).filter(min_month_obs)
    df_mon_mean = df_filter_1.groupby(["secid", "date_eom"]).mean().reset_index()

    num_months = len(np.unique(df_mon_mean["date_eom"]))

    df_filter = df_mon_mean.groupby("secid").filter(min_sample_obs)
    df_filter = df_filter.rename(columns = {"date_eom": "date"})

    return df_filter[["secid", "date", variable]]


def filter_min_obs_per_month(df, variable, min_obs_in_month):
    def min_month_obs(x):
        return x[variable].count() >= min_obs_in_month

    df["date"] = pd.to_datetime(df["date"])
    df["date_eom"] = df["date"] + pd.offsets.MonthEnd(0)

    df_filter_1 = df.groupby(["secid", "date_eom"]).filter(min_month_obs)

    return df_filter_1
