cd "/Users/rsigalov/Documents/PhD/disaster-risk-revision"

clear all

import delimited estimated_data/final_regression_dfs/rn_prob_decline_ret.csv

gen date = date(month_lead,"YMD")
gen crisis = (date >= td(1,1,2007) & date <= td(31,12,2009))
