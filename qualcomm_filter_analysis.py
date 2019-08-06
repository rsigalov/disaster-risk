import numpy as np
import pandas as pd
import wrds
import os
os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")
db = wrds.Connection(wrds_username = "rsigalov")

# Counting smiles with at least 5 observations:
def count_smiles_min_obs(df, days, min_obs, return_df = False):
    
    # Making sure to use only OTM options
    opt_df_tmp = df[
            (df.cp_flag == "C") & (df.strike_price >= df.close) |
            (df.cp_flag == "P") & (df.strike_price <= df.close)]

    opt_num_per_obs = opt_df_tmp.groupby(["secid", "date", "exdate"])["strike_price"].count().reset_index().rename({"strike_price":"num_opts"}, axis = 1)
    opt_num_per_obs["date"] = pd.to_datetime(opt_num_per_obs["date"])
    opt_num_per_obs["exdate"] = pd.to_datetime(opt_num_per_obs["exdate"])
    opt_df_tmp = pd.merge(opt_df_tmp, opt_num_per_obs, on = ["secid", "date", "exdate"], how = "left")
    
    opt_df_tmp = opt_df_tmp[opt_df_tmp["num_opts"] >= min_obs]
    opt_df_tmp["days"] = [x.days - 1 for x in opt_df_tmp["exdate"] - opt_df_tmp["date"]]
    min_days_df = opt_df_tmp.groupby(["secid", "date"])["days"].min().rename("min_days").reset_index()
    max_days_df = opt_df_tmp.groupby(["secid", "date"])["days"].max().rename("max_days").reset_index()
    
    min_max_days_df = pd.merge(min_days_df, max_days_df, on = ["secid", "date"])
    min_max_days_df["inside"] = (min_max_days_df["min_days"] <= days) & (min_max_days_df["max_days"] >= days)
    
    if return_df:
        return min_max_days_df
    else:
        print("%d/%d dates with maturity of %d days inside" % (min_max_days_df["inside"].sum(),
                                                               min_max_days_df["inside"].count(),
                                                               days))




secid = 101558
min_obs_in_smile_filter = 4
days = 30

raw_num_within = 0
total_obs_BB_OTM = 0
total_obs_all_filters = 0
total_obs_all_filters_all = 0

tracking_df = pd.DataFrame(columns = ["secid", "date", "min_days", "max_days", "inside"])

for year in range(1996, 2018, 1):
    print("-------------------------------------")
    print("Year " + str(year))
    print("")

    # Loading all data on options:
    query = """
    select 
        secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
        open_interest, impl_volatility, delta, ss_flag
    from _table_
    where secid = _secid_
    and date <= '2015-05-31'
    """
    
    query = query.replace("_table_", "OPTIONM.OPPRCD" + str(year)).replace("_secid_", str(secid))
    query = query.replace('\n', ' ').replace('\t', ' ')
    opt_df = db.raw_sql(query)
    
    if opt_df.shape[0] > 0:
    
        # Loading data on spot prices:
        query = """
        select 
            secid, date, close
        from optionm.SECPRD
        where secid = _secid_
        and date >= _start_date_
        and date <= _end_date_
        """
        
        query = query.replace("_start_date_", "'" + str(year) + "-01-01'").replace("_end_date_", "'" + str(year) + "-12-31'").replace("_secid_", str(secid))
        query = query.replace('\n', ' ').replace('\t', ' ')
        prc_df = db.raw_sql(query)
        
        # Converting dates:
        opt_df["date"] = pd.to_datetime(opt_df["date"])
        opt_df["exdate"] = pd.to_datetime(opt_df["exdate"])
        prc_df["date"] = pd.to_datetime(prc_df["date"])
    
        # Merging data on options and spot prices:
        opt_df = pd.merge(opt_df, prc_df[["secid", "date", "close"]],
                          on = ["secid", "date"], how = "left")
        # Dividing strikes by 1000 to make them comparable with spot prices:
        opt_df["strike_price"] = opt_df["strike_price"]/1000
        
        # Leaving only OTM options:
        opt_df = opt_df[
                (opt_df.cp_flag == "C") & (opt_df.strike_price >= opt_df.close) |
                (opt_df.cp_flag == "P") & (opt_df.strike_price <= opt_df.close)]
        
        # Filtering by best bid:
        opt_df = opt_df[opt_df["best_bid"] > 0]
        
        min_max_days_df = count_smiles_min_obs(opt_df, 30, 4, return_df = True)
        
        # Calculating number of smiles with maturities that cover 30 days
        min_max_days_df["inside"] = (min_max_days_df["min_days"] <= days) & (min_max_days_df["max_days"] >= days)
        total_obs_BB_OTM += np.sum(min_max_days_df["inside"])
        
        tracking_df = tracking_df.append(min_max_days_df)
        
        print(total_obs_BB_OTM)
        print("")
        
        # Next applying all filters used in actual analysis and calculating the number
        # of applicable smiles:
        # Loading data on distributions
        query = """
        select
            secid, payment_date, amount, distr_type
        from OPTIONM.DISTRD
        where secid = _secid_
        and currency = 'USD'
        """
        
        query = query.replace('\n', ' ').replace('\t', ' ')
        query = query.replace('_secid_', str(secid))
        dist_df = db.raw_sql(query)
        dist_df = dist_df[dist_df["distr_type"].isin([1,2,3,4,"1", "3", "4", "5", "%"])]
        dist_df["payment_date"] = pd.to_datetime(dist_df["payment_date"])
        
        # Loading data on ZCB rates:
        query = """
        select *
        from OPTIONM.ZEROCD
        where days <= 365*3
        and date >= _start_date_
        and date <= _end_date_
        """
        query = query.replace("_start_date_", "'" + str(year) + "-01-01'").replace("_end_date_", "'" + str(year) + "-12-31'")
        query = query.replace('\n', ' ').replace('\t', ' ')
        zcb_df = db.raw_sql(query)
        zcb_df["rate"] = zcb_df["rate"]/100
        zcb_df["date"] = pd.to_datetime(zcb_df["date"])
        
        # Common Filters among PUTS and CALLS:
        opt_df = opt_df[opt_df["open_interest"] > 0]
        opt_df = opt_df[opt_df["best_offer"] - opt_df["best_bid"] > 0]
        opt_df = opt_df[(opt_df["ss_flag"] == 0) | (opt_df["ss_flag"] == "0")]
        opt_df = opt_df[~(opt_df["delta"].isnull())]
        opt_df = opt_df[~(opt_df["impl_volatility"].isnull())]
        opt_df["dtm"] = [x.days - 1 for x in opt_df["exdate"] - opt_df["date"]]
        opt_df = opt_df[opt_df["dtm"] <= 3*365]
        opt_df = opt_df[opt_df["dtm"] > 0]
        
        # CALL specific filters (unrelated to dividends):
        opt_c_df = opt_df[opt_df["cp_flag"] == "C"]
        opt_c_df = opt_c_df[(opt_c_df["best_offer"] + opt_c_df["best_bid"])/2 < opt_c_df["close"]]
                
        # PUT specific filters (unrelated to dividends):
        opt_p_df = opt_df[opt_df["cp_flag"] == "P"]
        opt_p_df = opt_p_df[(opt_p_df["best_offer"] + opt_p_df["best_bid"])/2 < opt_p_df["strike_price"]]
        opt_p_df = opt_p_df[(opt_p_df["best_offer"] + opt_p_df["best_bid"])/2 >= np.maximum(0, opt_p_df["strike_price"] - opt_p_df["close"])]
        
            # 5. Combining Calls and Puts:
        opt_df = opt_c_df.append(opt_p_df)
        
        # 6. For each date calculating the present value of future dividends
        dist_df = dist_df[dist_df["distr_type"].isin(["1", "3", "4", "5", 1, 3, 4, 5, "%"])]
        
        # Grouping option data by (secid, date, exdate):
        dist_pv_df = pd.DataFrame(columns = ["secid", "date", "exdate", "dist_pv"])
        
        for name, opt_sub_df in opt_df.groupby(["secid", "date", "exdate"]):
            
            dist_inside_df = dist_df[
                    (dist_df["secid"] == name[0]) &
                    (dist_df["payment_date"] <= name[2]) & 
                    (dist_df["payment_date"] >= name[1])]
            
            if dist_inside_df.shape[0] == 0:
                dist_pv = 0
            else:
                dist_days = np.array([x.days - 1 for x in dist_inside_df["payment_date"] - name[1]])
                dist_amounts = np.array(dist_inside_df["amount"])
                
                # Interpolating zero coupon rate for dist_days:
                zcb_date_df = zcb_df[zcb_df["date"] == name[1]]
                if zcb_date_df.shape[0] == 0:
                    subzcb = zcb_df[zcb_df.date <= name[1]]
                    prev_obs_date = subzcb.date.iloc[-1]
                    zcb_date_df = zcb_df[zcb_df.date == prev_obs_date]
                    
                interp_int_rate = np.interp(dist_days, zcb_date_df["days"], zcb_date_df["rate"])
                
                # Calculating present value of dividends:
                dist_pv = np.sum(np.exp(-interp_int_rate * dist_days/365) * dist_amounts)
                
            to_append_df = pd.DataFrame(
                    {"secid":[name[0]], "date":[name[1]], "exdate":[name[2]], "dist_pv":[dist_pv]})
            
            dist_pv_df = dist_pv_df.append(to_append_df)
            
        # Merging PV of dividends back to the options data:
        opt_df = pd.merge(opt_df, dist_pv_df, on = ["secid", "date", "exdate"], how = "left")
        
        # 7. Interpolating interest rate for each option date:
        interp_int_rate_df = pd.DataFrame(columns = ["date", "exdate", "rate"])
        for name, opt_sub_df in opt_df.groupby(["date", "exdate"]):
            days_to_interp = (opt_sub_df["exdate"].iloc[0] - opt_sub_df["date"].iloc[0]).days
            
            zcb_date_df = zcb_df[zcb_df["date"] == name[0]]
            
            if zcb_date_df.shape[0] == 0:
                    subzcb = zcb_df[zcb_df.date <= name[0]]
                    prev_obs_date = subzcb.date.iloc[-1]
                    zcb_date_df = zcb_df[zcb_df.date == prev_obs_date]
                    
            interp_int_rate = np.interp(days_to_interp, zcb_date_df["days"], zcb_date_df["rate"])
            to_append = pd.DataFrame({"date":[name[0]], "exdate":[name[1]], "rate":[interp_int_rate]})
            interp_int_rate_df = interp_int_rate_df.append(to_append)
            
        # merging interpolated interest rate back to options data:
        opt_df = pd.merge(opt_df,interp_int_rate_df, on = ["date", "exdate"], how = "left")
            
        # 8. Calculating the last filter puts and call separately:
        opt_df["T"] = [(x.days - 1)/365 for x in opt_df["exdate"] - opt_df["date"]]
        opt_df["call_min"] = np.maximum(0, opt_df["close"] - opt_df["strike_price"]*np.exp(-opt_df["rate"] * opt_df["T"]) - opt_df["dist_pv"])
        opt_df["put_min"] = np.maximum(0, opt_df["strike_price"]*np.exp(-opt_df["rate"] * opt_df["T"]) + opt_df["dist_pv"] - opt_df["close"])
        
        # Filtering by minimum price for calls and puts
        opt_df = opt_df[
                (opt_df["cp_flag"] == "C") & ((opt_df["best_bid"] + opt_df["best_offer"])/2 >= opt_df["call_min"]) |
                (opt_df["cp_flag"] == "P") & ((opt_df["best_bid"] + opt_df["best_offer"])/2 >= opt_df["put_min"])]
        
        min_max_days_df = count_smiles_min_obs(opt_df, 30, 4, return_df = True)
        
        # Calculating number of smiles with maturities that cover 30 days
        min_max_days_df["inside"] = (min_max_days_df["min_days"] <= days) & (min_max_days_df["max_days"] >= days)
        total_obs_all_filters += np.sum(min_max_days_df["inside"])
            
        total_obs_all_filters_all += min_max_days_df.shape[0]
        
#        tracking_df = tracking_df.append(min_max_days_df)
    
    
db = wrds.Connection(wrds_username = "rsigalov")

year = 2005
query = """
select 
    date, 
    min(exdate - date) as min_maturity,
    max(exdate - date) as max_maturity
from _table_
where secid = _secid_
group by date
"""

query = query.replace("_table_", "OPTIONM.OPPRCD" + str(year)).replace("_secid_", str(secid))
query = query.replace('\n', ' ').replace('\t', ' ')
opt_df = db.raw_sql(query)
   

query = """
    with security_table as (
    select
        secid, date, close as under_price
    from optionm.SECPRD
    where secid = _secid_
    and date >= _start_date_
    and date <= _end_date_
), combined_table as (
(   select
        o.secid, o.date, o.exdate, o.strike_price
    from _data_base_ as o
    left join security_table as s
        on o.secid = s.secid and o.date = s.date
        where o.secid = _secid_
    and o.cp_flag = 'C'
    and o.open_interest > 0
    and o.best_bid > 0
    and o.best_offer - o.best_bid > 0
    and o.ss_flag = '0'
    and o.delta is not null
    and o.impl_volatility is not null
    and o.strike_price/1000 > s.under_price
    and (o.best_offer + o.best_bid)/2 < s.under_price
    and o.exdate - o.date <= 365 * 1
    and o.exdate - o.date > 0
    and o.date >= _start_date_
    and o.date <= _end_date_
    order by o.exdate, o.strike_price
) union (
    select
       o.secid, o.date, o.exdate, o.strike_price
    from _data_base_ as o
    left join security_table as s
        on o.secid = s.secid and o.date = s.date
    where o.secid = _secid_
    and o.cp_flag = 'P'
    and o.open_interest > 0
    and o.best_bid > 0
    and o.best_offer - o.best_bid > 0
    and o.ss_flag = '0'
    and o.delta is not null
    and o.impl_volatility is not null
    and o.strike_price/1000 < s.under_price
    and (o.best_offer + o.best_bid)/2 < o.strike_price/1000
    and (o.best_offer + o.best_bid)/2 >= GREATEST(0, o.strike_price/1000 - s.under_price)
    and o.exdate - o.date <= 365 * 1
    and o.exdate - o.date > 0
    and o.date >= _start_date_
    and o.date <= _end_date_
    order by o.exdate, o.strike_price
)
), cnt_table as (
    select 
        date, exdate,
        count(strike_price) as cnt_opts
    from combined_table
    group by date, exdate
    order by date, exdate
)
select 
    date,
    min(exdate - date) as min_maturity,
    max(exdate - date) as max_maturity
from cnt_table
where cnt_opts >= 4
group by date
"""
 
query = query.replace("_data_base_", "OPTIONM.OPPRCD" + str(year)).replace("_secid_", str(secid))
query = query.replace("_start_date_", "'" + str(year) + "-01-01'").replace("_secid_", str(secid))
query = query.replace("_end_date_", "'" + str(year) + "-12-31'")
query = query.replace('\n', ' ').replace('\t', ' ')
opt_df = db.raw_sql(query)



opt_df.set_index("date")["min_maturity"].plot()
tracking_df[(tracking_df.date >= "2005-01-01") & (tracking_df.date <= "2005-12-31")].set_index("date")["min_days"].plot()







