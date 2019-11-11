"""
This file replicates Cremers et al. JUMP and VOL factors
"""

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)

os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

## Loading CME data:
#df = pd.read_csv("data/raw_data/opt_data_CME_ITM_OTM.csv")

# Loading gull OptionMetrics data (including In-The-Money options)
df = pd.read_csv("data/raw_data/opt_data_spx_OM_ITM_OTM.csv")

df["date"] = pd.to_datetime(df["date"])
df["exdate"] = pd.to_datetime(df["exdate"])
df = df[df.date >= "1988-01-01"]
df["date_mon"] = df["date"] + pd.offsets.MonthEnd(0)
df["date_next_mon"] = df["date_mon"] + pd.offsets.MonthEnd(1)
df["date_following_mon"] = df["date_mon"] + pd.offsets.MonthEnd(2)
df["exdate_mon"] = df["exdate"] + pd.offsets.MonthEnd(0)

# Getting the next trading day as available in OptionMetrics to
# calculate returns later on
next_trading_day = pd.DataFrame({"date": np.unique(df.sort_values("date")["date"])})
next_trading_day["date_next"] = next_trading_day["date"].shift(-1)

# For each date taking options that expire in the next month and
# options that expire in the following month. Filtering to get
# options for each date that are closest to being ATM
def options_closest_to_ATM(opt_df, spot):    
    opt_df_pc = opt_df.groupby(["strike_price"]).date.count()
    opt_df_pc = opt_df_pc[opt_df_pc > 1]
    
    if opt_df_pc.shape[0] > 0:
        i_strike_min_dist = np.argmin(np.abs(np.array(opt_df_pc.index) - spot))
        strike_min_dist = opt_df_pc.index[i_strike_min_dist]
        return opt_df[opt_df.strike_price == strike_min_dist]
    else: # If not matches return an empty DataFrame
        return pd.DataFrame(columns = opt_df.columns)

df_next_full = pd.DataFrame(columns = df.columns)
df_follow_full = pd.DataFrame(columns = df.columns)
missing_dates = []

# For each date picking options with maturity next and the following months. Among
# these options selecting a pair of call and put with a strike closest to spot price
for i_date, date in enumerate(next_trading_day.date):
    
    if i_date % 500 == 0:
        print("%d out of %d" % (i_date, next_trading_day.date.shape[0]))
    
    df_sub = df[df.date == date]
    df_next = df_sub[df_sub.exdate_mon == df_sub.date_next_mon]
    df_follow = df_sub[df_sub.exdate_mon == df_sub.date_following_mon]
    
    # In later parts of the sample there can be multiple maturity dates
    # for a given month. I will pick the latest one among the
    # available ones
    if (df_next.shape[0] > 0) & (df_follow.shape[0] > 0):
        df_next = df_next[df_next.exdate == np.max(np.unique(df_next.exdate))]
        df_follow = df_follow[df_follow.exdate == np.max(np.unique(df_follow.exdate))]
        
        spot = df_sub["spot"].iloc[0]
    
        to_append_next = options_closest_to_ATM(df_next, spot)
        to_append_follow = options_closest_to_ATM(df_follow, spot)
        
        # If there are both next and the following month straddles append the
        # corresponding options to the full DataFrame
        if (to_append_next.shape[0] >= 2) & (to_append_follow.shape[0] >= 2):
            df_next_full = df_next_full.append(to_append_next)
            df_follow_full = df_follow_full.append(to_append_follow)
        else:
            missing_dates.append(date)
    else:
        missing_dates.append(date)
    

# Calculating forward prices for all options in the sample:
#df_next_full["forward"] = np.exp(-(df_next_full["div_yield"] - df_next_full["int_rate"])*df_next_full["T"])*df_next_full["spot"]
#df_follow_full["forward"] = np.exp(-(df_follow_full["div_yield"] - df_follow_full["int_rate"])*df_follow_full["T"])*df_follow_full["spot"]

# Calculating implied vols
def BS_call_price(S0, q, r, K, sigma, T):
    
    d1 = (np.log(S0/K) + (r - q + np.power(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    p1 = np.exp(-q*T) * S0 * norm.cdf(d1)
    p2 = np.exp(-r*T) * K * norm.cdf(d2)
    
    return p1 - p2
    
def BS_put_price(S0, q, r, K, sigma, T):
    
    d1 = (np.log(S0/K) + (r - q + np.power(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    p1 = norm.cdf(-d2) * K * np.exp(-r*T)
    p2 = norm.cdf(-d1) * S0 * np.exp(-q*T)

    return p1 - p2

def calculate_implied_vol(row):
    F = row["forward"]
    r = row["int_rate"]
    K = row["strike_price"]
    T = row["T"]
    P = row["price"]
    opt_type = row["cp_flag"]
    
    def to_minimize(x):
        if opt_type == "C":
            return (BS_call_price(F, 0, r, K, x, T) - P)**2
        elif opt_type == "P":
            return (BS_put_price(F, 0, r, K, x, T) - P)**2
        else:
            raise ValueError("cp_flag should be either 'C' or 'P'")
        
    res = minimize(to_minimize, [0.18], method='nelder-mead', 
                   options={'xtol': 1e-8, 'disp': False})
    
    if res.success:
        return res.x[0]
    else:
        return np.nan

# Don't need to calculate implied volatility for OptionMetrics data
# since they already provide it. Uncomment this for CME data
        
#impl_vol_next = df_next_full.apply(calculate_implied_vol, axis = 1)
#df_next_full["impl_volatility"] = impl_vol_next
#impl_vol_follow = df_follow_full.apply(calculate_implied_vol, axis = 1)
#df_follow_full["impl_volatility"] = impl_vol_follow

df_next_full["type"] = "short"
df_follow_full["type"] = "long"

df_full = df_next_full.append(df_follow_full)


# Calculating position in longer dated option (y in Cremers paper)
def BS_call_delta(S0, q, r, K, sigma, T):
    d1 = (np.log(S0/K) + (r - q + np.power(sigma, 2)/2)*T)/(sigma * np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1)

def BS_put_delta(S0, q, r, K, sigma, T):
    d1 = (np.log(S0/K) + (r - q + np.power(sigma, 2)/2)*T)/(sigma * np.sqrt(T))
    return -np.exp(-q*T)*norm.cdf(-d1)

def BS_vega(S0, q, r, K, sigma, T):
    d1 = (np.log(S0/K) + (r - q + np.power(sigma, 2)/2)*T)/(sigma*np.sqrt(T))
    return S0*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
    
def BS_gamma(S0, q, r, K, sigma, T):
    d1 = (np.log(S0/K) + (r - q + np.power(sigma, 2)/2)*T)/(sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.pdf(d1)/(S0*sigma*np.sqrt(T))

df_full = pd.merge(df_full, next_trading_day, on = "date", how = "left")
df_full = pd.merge(
        df_full, 
        df[["date", "exdate", "strike_price", "cp_flag", "price"]].rename({"price": "price_next"}, axis = 1), 
        left_on = ["date_next", "exdate", "strike_price", "cp_flag"], 
        right_on = ["date", "exdate", "strike_price", "cp_flag"], how = "left")
df_full = df_full.drop("date_y", axis = 1).rename({"date_x":"date"}, axis = 1)


# Function that takes a sub-df that contains two options that correspond to
# a straddle and caluclates the y position in the long straddle to make
# the strategy gamma or vega neutral
def calculate_jump_vol_ret(sub_df):
    # Calculating gamma and vega for each of the straddles. Since both 
    # gamma and vega are the the same for puts and calls of the same
    # strike and maturity, we will just use the call to calculate
    # each of them
    
    spot = sub_df.spot.iloc[0]
    div_yield = sub_df.div_yield.iloc[0]
    
    # Splitting dataframe into short and long straddle options:
    sub_df = sub_df.sort_values(["type", "cp_flag"])
    sub_df_short = sub_df[sub_df.type == "short"]
    sub_df_long = sub_df[sub_df.type == "long"]
    
    # Getting parameters for the short straddle
    r_short = sub_df_short.int_rate.iloc[0]
    K_short = sub_df_short.strike_price.iloc[0]
    sigma_short = sub_df_short.impl_volatility.iloc[0]
    T_short = sub_df_short["T"].iloc[0]
    price_call_short = sub_df_short["price"].iloc[0]
    price_put_short = sub_df_short["price"].iloc[1]
    
    # Getting parameters for the long straddle
    r_long = sub_df_long.int_rate.iloc[0]
    K_long = sub_df_long.strike_price.iloc[0]
    sigma_long = sub_df_long.impl_volatility.iloc[0]
    T_long = sub_df_long["T"].iloc[0]
    price_call_long = sub_df_long["price"].iloc[0]
    price_put_long = sub_df_long["price"].iloc[1]
    
    # Calculating betas of options
    beta_call_short = BS_call_delta(spot, div_yield, r_short, K_short, sigma_short, T_short)*spot/price_call_short
    beta_put_short = (price_call_short/price_put_short)*beta_call_short - spot/price_put_short
    
    beta_call_long = BS_call_delta(spot, div_yield, r_long, K_long, sigma_long, T_long)*spot/price_call_long
    beta_put_long = (price_call_long/price_put_long)*beta_call_long - spot/price_put_long
    
    # Calculating weight on call and put to make each straddle market neutral
    theta_short = beta_put_short/(beta_put_short - beta_call_short)
    theta_long = beta_put_long/(beta_put_long - beta_call_long)
    
    # Calculating gamma of short and long straddle positions:
    gamma_short = BS_gamma(spot, div_yield, r_short, K_short, sigma_short, T_short)
    gamma_long = BS_gamma(spot, div_yield, r_long, K_long, sigma_long, T_long)
    y_zero_gamma = gamma_long/gamma_short
    
    vega_short = BS_vega(spot, div_yield, r_short, K_short, sigma_short, T_short)
    vega_long = BS_vega(spot, div_yield, r_long, K_long, sigma_long, T_long)    
    y_zero_vega = vega_short/vega_long
    
    # Calculating returns on individual options
    price_call_short_next = sub_df_short["price_next"].iloc[0]
    price_put_short_next = sub_df_short["price_next"].iloc[1]
    
    price_call_long_next = sub_df_long["price_next"].iloc[0]
    price_put_long_next = sub_df_long["price_next"].iloc[1]
    
    ret_vec = np.array([price_call_short_next/price_call_short - 1,
                        price_put_short_next/price_put_short - 1,
                        price_call_long_next/price_call_long - 1,
                        price_put_long_next/price_put_long - 1])
    
    # Calculating weights for zero vega strategy (JUMP):
    w_zero_vega = np.array([theta_short,
                            1 - theta_short,
                            -theta_long * y_zero_vega,
                            -(1 - theta_long) * y_zero_vega])

    ret_zero_vega = np.inner(w_zero_vega, ret_vec)
    
    # Calculating weights for zero gamma strategy (VOL):
    w_zero_gamma = np.array([-theta_short * y_zero_gamma,
                             -(1 - theta_short) * y_zero_gamma,
                             theta_long,
                             (1 - theta_long)])

    ret_zero_gamma = np.inner(w_zero_gamma, ret_vec)
    
    # Calculating straddle returns:

    ret_short_straddle = np.inner(np.array([theta_short, 1 - theta_short]),
                                  np.array([price_call_short_next/price_call_short - 1,
                                           price_put_short_next/price_put_short - 1]))
    
    ret_long_straddle = np.inner(np.array([theta_long, 1 - theta_long]),
                                 np.array([price_call_long_next/price_call_long - 1,
                                          price_put_long_next/price_put_long - 1]))
    
    # Calculating vega for zero gamma position and gamma for zero vega position:
    vega = -vega_short*y_zero_gamma + vega_long
    gamma = gamma_short - y_zero_vega*gamma_long

    return (theta_short, theta_long, y_zero_gamma, y_zero_vega, vega, gamma, ret_zero_gamma, ret_zero_vega, ret_short_straddle, ret_long_straddle)

jump_vol_ret = df_full.groupby("date").apply(calculate_jump_vol_ret).apply(pd.Series)
jump_vol_ret.columns = ["theta_short", "theta_long", "y_zero_gamma", "y_zero_vega", "vega", "gamma", "ret_zero_gamma", "ret_zero_vega", "ret_short_straddle", "ret_long_straddle"]

# Adjusting dates:
jump_vol_ret = pd.merge(jump_vol_ret, next_trading_day, left_index = True,
                        right_on = "date")
jump_vol_ret.drop("date", axis = 1, inplace = True)
jump_vol_ret.rename({"date_next": "date"}, axis = 1, inplace = True)
jump_vol_ret.set_index("date", inplace = True)

##########################################################################
# Making a table with summary stats on zero-vega (JUMP factor) returns:
sum_stats_full = jump_vol_ret[["ret_zero_vega"]].describe()
sum_stats_full = sum_stats_full.loc[["mean", "std"]]
sum_stats_full.loc["mean"] = sum_stats_full.loc["mean"]*252
sum_stats_full.loc["std"] = sum_stats_full.loc["std"] * np.sqrt(252)

sum_stats_short = jump_vol_ret[jump_vol_ret.index <= "2011-12-31"][["ret_zero_vega"]].describe()
sum_stats_short = sum_stats_short.loc[["mean", "std"]]
sum_stats_short.loc["mean"] = sum_stats_short.loc["mean"]*252
sum_stats_short.loc["std"] = sum_stats_short.loc["std"] * np.sqrt(252)

sum_stats = pd.concat([sum_stats_full, sum_stats_short], axis = 1)
sum_stats.columns = ["full", "short"]
sum_stats.loc["sharpe"] = sum_stats.loc["mean"]/sum_stats.loc["std"]

char_list = ["Mean", "SD", "Sharpe"]
path = "estimated_data/cremers/jump_sum_stats.tex"
f = open(path, "w")
f.write("\\begin{tabular}{lcc}\n")
f.write("\\toprule \n")
f.write("Sample: & Full & Thru Dec-11  \\\\ \n")
f.write("\hline \\\\[-1.8ex] \n")
for i_row in range(sum_stats.shape[0]):
    vars_to_write = [char_list[i_row]] + list(sum_stats.iloc[i_row])
    f.write("{} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
    
f.write("\\bottomrule \n")
f.write("\end{tabular} \n")  
f.close()

##########################################################################
# Table with correlation



##########################################################################
# Making a table with correlations of JUMP with disaster risk series:

# Using daily data from SPX options to calculate daily correlations:
df_sp_daily = pd.read_csv("estimated_data/disaster-risk-series/spx_OM_daily_disaster.csv")
df_sp_daily = pd.melt(df_sp_daily, id_vars = ["date", "days"])

measure_list = ["D_clamp",  "rn_prob_5", "rn_prob_10", "rn_prob_15", "rn_prob_20"]

days_list = [30, 60, 120, 180]
corr_agg_daily_df = pd.DataFrame(columns = ["measure", "days", "corr"])

for days in days_list:
    for measure in measure_list:
        # Subsetting disaster series:
        D = df_sp_daily[(df_sp_daily.variable == measure) & (df_sp_daily.days == days)].set_index("date")["value"].diff()
        
        # Merging with Cremers' JUMP:
        D = pd.merge(D, jump_vol_ret[["ret_zero_vega"]], left_index = True, right_index = True)

        # Calculating and writing correlation:
        corr_agg_daily_df = corr_agg_daily_df.append(
                pd.DataFrame({"measure": [measure], 
                              "days": [days], 
                              "corr": [D.corr().loc["value", "ret_zero_vega"]]}))


corr_daily_stats = pd.pivot_table(corr_agg_daily_df, index = "measure", columns = "days", values = "corr")
corr_daily_stats = corr_daily_stats.rename({"rn_prob_5":"rn_prob_05"}, axis = 0)
corr_daily_stats = corr_daily_stats.sort_index()

####################################################################################
# Saving Cremers et al factors:
cremers_factors = jump_vol_ret[["ret_zero_vega", "ret_zero_gamma"]].rename(
        {"ret_zero_vega":"JUMP",
         "ret_zero_gamma":"VOL"}, axis = 1)    
cremers_factors.to_csv("data/cremers_factors.csv")



######################################################################################
# Comparing with disaster measure derived from SPX options on daily level
cremers_factors = cremers_factors[["JUMP"]]
df_sp_daily = pd.read_csv("estimated_data/disaster-risk-series/spx_OM_daily_disaster.csv")
df_sp_daily = pd.melt(df_sp_daily, id_vars = ["date", "days"])

measure_list = ["D_clamp", "rn_prob_5",
                "rn_prob_10", "rn_prob_15", "rn_prob_20"]

days_list = [30, 60, 120, 180]
corr_agg_daily_df = pd.DataFrame(columns = ["measure", "days", "corr"])

for days in days_list:
    for measure in measure_list:
        # Subsetting disaster series:
        D = df_sp_daily[(df_sp_daily.variable == measure) & (df_sp_daily.days == days)].set_index("date")["value"].diff()
        
        # Merging with Cremers' JUMP:
        D = pd.merge(D, cremers_factors, left_index = True, right_index = True)

        # Calculating and writing correlation:
        corr_agg_daily_df = corr_agg_daily_df.append(
                pd.DataFrame({"measure": [measure], 
                              "days": [days], 
                              "corr": [D.corr().loc["value", "JUMP"]]}))


corr_daily_stats = pd.pivot_table(corr_agg_daily_df, index = "measure", columns = "days", values = "corr")
corr_daily_stats = corr_daily_stats.rename({"rn_prob_5":"rn_prob_05"}, axis = 0)
corr_daily_stats = corr_daily_stats.sort_index()

################################################################
# Calculating correlation of D with P(r<-X) for different x
df_sp_daily = pd.read_csv("estimated_data/disaster-risk-series/spx_OM_daily_disaster.csv")
df_sp_daily = pd.melt(df_sp_daily, id_vars = ["date", "days"])
df_sp_daily = df_sp_daily[
        ((df_sp_daily.variable == "D_clamp") & (df_sp_daily.days == 30)) | 
        (df_sp_daily.variable.isin(["rn_prob_5", "rn_prob_10", "rn_prob_15", "rn_prob_20"]) & (df_sp_daily.days.isin([30, 60,120,180])))]

corr_df_1 = pd.pivot_table(df_sp_daily, index = "date", columns = ["variable", "days"], values = "value").diff().corr()
corr_df_1 = pd.pivot_table(corr_df_1.loc[("D_clamp",30)].reset_index(), index = "variable", columns = "days")
corr_df_1 = corr_df_1[corr_df_1.index != "D_clamp"]
corr_df_1.rename({"rn_prob_5":"rn_prob_05"}, inplace = True)
corr_df_1.sort_index(inplace = True)

corr_df_2 = corr_daily_stats[corr_daily_stats.index != "D_clamp"]

######################################################################################
# Comparing with disaster measure derived from SPX options on monthly level
cremers_factors["date_mon"] = cremers_factors.index + pd.offsets.MonthEnd(0)
cremers_factors_mon = cremers_factors.copy().set_index("date_mon")
cremers_factors_mon = (cremers_factors_mon + 1).groupby("date_mon").prod() - 1
cremers_factors_mon = cremers_factors_mon["JUMP"]

df_sp_mon = pd.read_csv("estimated_data/disaster-risk-series/spx_OM_monthly_disaster.csv")
df_sp_mon = pd.melt(df_sp_mon, id_vars = ["date", "days"])

measure_list = ["D_clamp", "rn_prob_5", "rn_prob_10", "rn_prob_15", "rn_prob_20"]
days_list = [30, 60, 120, 180]
corr_agg_mon_df = pd.DataFrame(columns = ["measure", "days", "corr"])

for days in days_list:
    for measure in measure_list:
        # Subsetting disaster series:
        D = df_sp_mon[(df_sp_mon.variable == measure) & (df_sp_mon.days == days)].set_index("date")["value"].diff()
        
        # Merging with Cremers' JUMP:
        D = pd.merge(D, cremers_factors_mon, left_index = True, right_index = True)

        # Calculating and writing correlation:
        corr_agg_mon_df = corr_agg_mon_df.append(
                pd.DataFrame({"measure": [measure], 
                              "days": [days], 
                              "corr": [D.corr().loc["value", "JUMP"]]}))

corr_mon_stats = pd.pivot_table(corr_agg_mon_df, index = "measure", columns = "days", values = "corr")
corr_mon_stats = corr_mon_stats.rename({"rn_prob_5":"rn_prob_05"}, axis = 0)
corr_mon_stats = corr_mon_stats.sort_index()


######################################################################################
# Comparing with disaster measure derived from individual options on monthly level
df_D = pd.read_csv("estimated_data/disaster-risk-series/combined_disaster_df.csv")
df_D = df_D[(df_D.level == "union_cs") & (df_D.agg_type == "mean_filter")]

measure_list = ["D_clamp", "rn_prob_20", "rn_prob_40", "rn_prob_60", "rn_prob_80"]
days_list = [30, 60, 120, 180]

corr_ind_mon_df = pd.DataFrame(columns = ["measure", "days", "corr"])

for days in days_list:
    for measure in measure_list:
        # Subsetting disaster series:
        D = df_D[(df_D["var"] == measure) & (df_D.days == days)].set_index("date")["value"].diff()
        
        # Merging with Cremers' JUMP:
        D = pd.merge(D, cremers_factors_mon, left_index = True, right_index = True)

        # Calculating and writing correlation:
        corr_ind_mon_df = corr_ind_mon_df.append(
                pd.DataFrame({"measure": [measure], 
                              "days": [days], 
                              "corr": [D.corr().loc["value", "JUMP"]]}))


corr_mon_ind_stats = pd.pivot_table(corr_ind_mon_df, index = "measure", columns = "days", values = "corr")
corr_mon_ind_stats = corr_mon_ind_stats.rename({"rn_prob_5":"rn_prob_05"}, axis = 0)
corr_mon_ind_stats = corr_mon_ind_stats.sort_index()



row_name_list = ["$P(r < -5\%)$", "$P(r < -10\%)$", "$P(r < -15\%)$", "$P(r < -20\%)$"]
path = "estimated_data/cremers/comp_daily_corr.tex"
f = open(path, "w")
f.write("\\begin{tabular}{lcccc}\n")

f.write("\\multicolumn{4}{l}{\\textbf{30 day clamped $\\mathbb{D}$}}  \\\\ \\\\[-1.8ex]\n")
f.write("\\hline \\\\[-1.8ex] \n")
f.write("days: & 30 & 60 & 120 & 180 \\\\ \\\\[-1.8ex]\n")
f.write("\hline \\\\[-1.8ex] \n")
for i_row in range(corr_df_1.shape[0]):
    vars_to_write = [row_name_list[i_row]] + list(corr_df_1.iloc[i_row])
    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
f.write("\\bottomrule \n\\\\[-1.2ex]")

f.write("\\multicolumn{4}{l}{\\textbf{JUMP}} \\\\ \\\\[-1.8ex]\n")
f.write("\\hline \\\\[-1.8ex]\n")
f.write("days: & 30 & 60 & 120 & 180\\\\ \\\\[-1.8ex]\n")
f.write("\hline \\\\[-1.8ex] \n")
for i_row in range(corr_df_2.shape[0]):
    vars_to_write = [row_name_list[i_row]] + list(corr_df_2.iloc[i_row])
    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
f.write("\\bottomrule \n")

f.write("\end{tabular} \n")  
f.close()



############################################################################
# Saving output to Latex tables:
#row_name_list = ["clamped $\\mathbb{D}$", "$P(r < -5\%)$", "$P(r < -10\%)$", "$P(r < -15\%)$", "$P(r < -20\%)$"]
#path = "estimated_data/cremers/jump_daily_corr.tex"
#f = open(path, "w")
#f.write("\\begin{tabular}{lcccc}\n")
#f.write("\\toprule \n")
#f.write("days: & 30 & 60 & 120 & 180\\\\ \n")
#f.write("\hline \\\\[-1.8ex] \n")
#for i_row in range(corr_daily_stats.shape[0]):
#    vars_to_write = [row_name_list[i_row]] + list(corr_daily_stats.iloc[i_row])
#    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#    
#f.write("\\bottomrule \n")
#f.write("\end{tabular} \n")  
#f.close()
#
#
#row_name_list = ["clamped $\\mathbb{D}$", "$P(r < -5\%)$", "$P(r < -10\%)$", "$P(r < -15\%)$", "$P(r < -20\%)$"]
#path = "estimated_data/cremers/jump_mon_corr.tex"
#f = open(path, "w")
#f.write("\\begin{tabular}{lcccc}\n")
#f.write("\\toprule \n")
#f.write("days: & 30 & 60 & 120 & 180\\\\ \n")
#f.write("\hline \\\\[-1.8ex] \n")
#for i_row in range(corr_mon_stats.shape[0]):
#    vars_to_write = [row_name_list[i_row]] + list(corr_mon_stats.iloc[i_row])
#    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#    
#f.write("\\bottomrule \n")
#f.write("\end{tabular} \n")  
#f.close()
#
#
#corr_mon_daily = pd.concat([corr_daily_stats, corr_mon_stats], axis = 1)
#row_name_list = ["clamped $\\mathbb{D}$", "$P(r < -5\%)$", "$P(r < -10\%)$", "$P(r < -15\%)$", "$P(r < -20\%)$"]
#path = "estimated_data/cremers/jump_daily_mon_corr.tex"
#f = open(path, "w")
#f.write("\\begin{tabular}{lcccc|cccc}\n")
#f.write("\\toprule \n")
#f.write(" & \\multicolumn{4}{c}{Daily Frequency} & \\multicolumn{4}{c}{Monthly Frequency} \\\\ \\\\[-1.8ex]\n")
#f.write("\cline{2-9} \\\\[-1.8ex] \n")
#f.write("days: & 30 & 60 & 120 & 180 & 30 & 60 & 120 & 180\\\\ \n")
#f.write("\hline \\\\[-1.8ex] \n")
#for i_row in range(corr_mon_daily.shape[0]):
#    vars_to_write = [row_name_list[i_row]] + list(corr_mon_daily.iloc[i_row])
#    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}  & {:.3f}  & {:.3f}  & {:.3f}  \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#    
#f.write("\\bottomrule \n")
#f.write("\end{tabular} \n")  
#f.close()
#
#
#row_name_list = ["clamped $\\mathbb{D}$", "$P(r < -20\%)$", "$P(r < -40\%)$", "$P(r < -60\%)$", "$P(r < -80\%)$"]
#path = "estimated_data/cremers/jump_mon_ind_corr.tex"
#f = open(path, "w")
#f.write("\\begin{tabular}{lcccc}\n")
#f.write("\\toprule \n")
#f.write("days: & 30 & 60 & 120 & 180\\\\ \n")
#f.write("\hline \\\\[-1.8ex] \n")
#for i_row in range(corr_mon_ind_stats.shape[0]):
#    vars_to_write = [row_name_list[i_row]] + list(corr_mon_ind_stats.iloc[i_row])
#    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#    
#f.write("\\bottomrule \n")
#f.write("\end{tabular} \n")  
#f.close()
#
#
#row_name_list = ["$P(r < -5\%)$", "$P(r < -10\%)$", "$P(r < -15\%)$", "$P(r < -20\%)$"]
#path = "estimated_data/cremers/comp_daily_corr.tex"
#f = open(path, "w")
#f.write("\\begin{tabular}{lcccc}\n")
#f.write("\\multicolumn{5}{l}{\\textbf{Correlation of JUMP with other measures}}")
#f.write("\\toprule \n")
#f.write("days: & 30 & 60 & 120 & 180\\\\ \n")
#f.write("\hline \\\\[-1.8ex] \n")
#for i_row in range(corr_df_1.shape[0]):
#    vars_to_write = [row_name_list[i_row]] + list(corr_df_1.iloc[i_row])
#    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#f.write("\\bottomrule \n\\\\[-1.2ex]")
#
#f.write("\\multicolumn{5}{l}{\\textbf{Correlation of 30 day clamped $\\mathbb{D}$ with other measures}}")
#f.write("\\toprule \n")
#f.write("days: & 30 & 60 & 120 & 180\\\\ \n")
#f.write("\hline \\\\[-1.8ex] \n")
#for i_row in range(corr_df_2.shape[0]):
#    vars_to_write = [row_name_list[i_row]] + list(corr_df_2.iloc[i_row])
#    f.write("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\\\[-1.8ex]\n".format(*vars_to_write))
#f.write("\\bottomrule \n")
#f.write("\end{tabular} \n")  
#f.close()













