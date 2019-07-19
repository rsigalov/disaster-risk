import numpy as np
import pandas as pd
import wrds
import os
os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")
db = wrds.Connection(wrds_username = "rsigalov")

# Loading Emil's top secid counts:
top_secids = pd.read_csv("/Users/rsigalov/Desktop/secid_counts.csv")
top_secids = top_secids.sort_values("c", ascending = False)
top_secids.to_csv("/Users/rsigalov/Desktop/secid_counts.csv", index = False)


# Loading data on a particular secid and year to see whether the problem lies:
secid = 101062
year = 1996
min_obs_in_smile_filter = 5

for year in range(1996, 2018, 1):
    print("-------------------------------------")
    print("Starting year " + str(year))
    print("")

    # Loading data on options:
    query = """
    select 
        secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
        open_interest, impl_volatility, delta, ss_flag
    from _table_
    where secid = _secid_
    """
    
    query = query.replace("_table_", "OPTIONM.OPPRCD" + str(year)).replace("_secid_", str(secid))
    query = query.replace('\n', ' ').replace('\t', ' ')
    opt_df = db.raw_sql(query)
    
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
    
    
    
    # Loading data on Qualcomm
    #opt_df = pd.read_csv("data/qualcomm_sample_data.csv")
    #prc_df = pd.read_csv("data/qualcomm_sample_price_data.csv")
    #zcb_df = pd.read_csv("data/qualcomm_sample_zcb_data.csv")
    #zcb_df["rate"] = zcb_df["rate"]/100
    #dist_df = pd.read_csv("data/qualcomm_sample_dist_data.csv")
    
    ## Converting dates
    #opt_df["date"] = pd.to_datetime(opt_df["date"], format = "%Y%m%d")
    #opt_df["exdate"] = pd.to_datetime(opt_df["exdate"], format = "%Y%m%d")
    #prc_df["date"] = pd.to_datetime(prc_df["date"], format = "%Y%m%d")
    #zcb_df["date"] = pd.to_datetime(zcb_df["date"], format = "%Y%m%d")
    #dist_df["payment_date"] = pd.to_datetime(dist_df["payment_date"], format = "%Y%m%d")
    
    opt_df["date"] = pd.to_datetime(opt_df["date"])
    opt_df["exdate"] = pd.to_datetime(opt_df["exdate"])
    prc_df["date"] = pd.to_datetime(prc_df["date"])
    zcb_df["date"] = pd.to_datetime(zcb_df["date"])
    dist_df["payment_date"] = pd.to_datetime(dist_df["payment_date"])
    
    # Doing filtering
    # 1. Merging the data on spot prices with option data:
    opt_df = pd.merge(opt_df, prc_df[["secid", "date", "close"]],
                      on = ["secid", "date"], how = "left")
    
    
    
    # Leaving only out-of-the-money options:
    opt_df["strike_price"] = opt_df["strike_price"]/1000
    opt_df = opt_df[
            (opt_df.cp_flag == "C") & (opt_df.strike_price >= opt_df.close) |
            (opt_df.cp_flag == "P") & (opt_df.strike_price <= opt_df.close)]
    
    # Counting number of days where can interpolate
    print("After OTM restriction:")
    count_smiles_min_obs(opt_df, 30, min_obs_in_smile_filter)
    print("")
    
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
        
        print("%d/%d dates with maturity of %d days inside" % (min_max_days_df["inside"].sum(),
                                                               min_max_days_df["inside"].count(),
                                                               days))
        
        if return_df:
            return min_max_days_df
    
    
    # 3. Common filters for call and put options:
    opt_df = opt_df[opt_df["open_interest"] > 0]
    opt_df = opt_df[opt_df["best_offer"] - opt_df["best_bid"] > 0]
    opt_df = opt_df[(opt_df["ss_flag"] == 0) | (opt_df["ss_flag"] == "0")]
    opt_df = opt_df[~(opt_df["delta"].isnull())]
    opt_df = opt_df[~(opt_df["impl_volatility"].isnull())]
    opt_df["dtm"] = [x.days - 1 for x in opt_df["exdate"] - opt_df["date"]]
    opt_df = opt_df[opt_df["dtm"] <= 365]
    opt_df = opt_df[opt_df["dtm"] > 0]
    
    print("After common Put/Call restrictions")
    count_smiles_min_obs(opt_df, 30, min_obs_in_smile_filter)
    print("")
    
    # 3. Dealing with call specific filters:
    opt_c_df = opt_df[opt_df["cp_flag"] == "C"]
    opt_c_df = opt_c_df[opt_c_df["strike_price"] >= opt_c_df["close"]]
    opt_c_df = opt_c_df[(opt_c_df["best_offer"] + opt_c_df["best_bid"])/2 < opt_c_df["close"]]
    
    
    # 4. Dealing with put options:
    opt_p_df = opt_df[opt_df["cp_flag"] == "P"]
    opt_p_df = opt_p_df[opt_p_df["strike_price"] <= opt_p_df["close"]]
    opt_p_df = opt_p_df[(opt_p_df["best_offer"] + opt_p_df["best_bid"])/2 < opt_p_df["strike_price"]]
    opt_p_df = opt_p_df[(opt_p_df["best_offer"] + opt_p_df["best_bid"])/2 >= np.maximum(0, opt_p_df["strike_price"] - opt_p_df["close"])]
    
    print("After Put/Call specific restrictions")
    count_smiles_min_obs(opt_df, 30, min_obs_in_smile_filter)
    print("")
    
    # 5. Combining Calls and Puts:
    opt_df = opt_c_df.append(opt_p_df)
    
    # 6. For each date calculating the present value of future dividends
    dist_df = dist_df[dist_df["distr_type"].isin(["1", "3", "4", "5",1, 3, 4, 5, "%"])]
    
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
    
    print("After Dividend Restrictions")
    count_smiles_min_obs(opt_df, 30, min_obs_in_smile_filter)
    print("")
    
    opt_df = opt_df[opt_df["best_bid"] > 0]
    print("After best_bid > 0 restriction")
    count_smiles_min_obs(opt_df, 30, min_obs_in_smile_filter)
    print("")
    
        
####################################################
# Leaving only the best bid and OTM options filters
####################################################  

db = wrds.Connection(wrds_username = "rsigalov")

secid = 101328
min_obs_in_smile_filter = 4
days = 30

total_obs = 0
for year in range(2017, 2018, 1):
    print("-------------------------------------")
    print("Starting year " + str(year))
    print("")

    # Loading data on options:
    query = """
    select 
        secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
        open_interest, impl_volatility, delta, ss_flag
    from _table_
    where secid = _secid_
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
        opt_df["strike_price"] = opt_df["strike_price"]/1000
        
        # Leaving only OTM options:
        opt_df = opt_df[
                (opt_df.cp_flag == "C") & (opt_df.strike_price >= opt_df.close) |
                (opt_df.cp_flag == "P") & (opt_df.strike_price <= opt_df.close)]
        
        # Filtering by best bid:
        opt_df = opt_df[opt_df["best_bid"] > 0]
        
        # Calculating number of options in each smile
        opt_num_per_obs = opt_df.groupby(["secid", "date", "exdate"])["strike_price"].count().reset_index().rename({"strike_price":"num_opts"}, axis = 1)
        
        # Converting dates
        opt_num_per_obs["date"] = pd.to_datetime(opt_num_per_obs["date"])
        opt_num_per_obs["exdate"] = pd.to_datetime(opt_num_per_obs["exdate"])
        
        # Merging number of options for each smile back to options data
        opt_df = pd.merge(opt_df, opt_num_per_obs, on = ["secid", "date", "exdate"], how = "left")
        
        # Removing smiles where the number of options is less than min_filter
        opt_df = opt_df[opt_df["num_opts"] >= min_obs_in_smile_filter]
        opt_df["days"] = [x.days - 1 for x in opt_df["exdate"] - opt_df["date"]]
        
        # Among the remaining smiles calculating the minimum and maximum maturities for each (secid, date)
        min_days_df = opt_df.groupby(["secid", "date"])["days"].min().rename("min_days").reset_index()
        max_days_df = opt_df.groupby(["secid", "date"])["days"].max().rename("max_days").reset_index()
        min_max_days_df = pd.merge(min_days_df, max_days_df, on = ["secid", "date"])
        
        # Calculating number of smiles with maturities that cover 30 days
        min_max_days_df["inside"] = (min_max_days_df["min_days"] <= days) & (min_max_days_df["max_days"] >= days)
        total_obs += np.sum(min_max_days_df["inside"])
        
        print(total_obs)
        print("")
        
print("Total observations: %d" % total_obs)
    
    
    
    
    
    
    
    