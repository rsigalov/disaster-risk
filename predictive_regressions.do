cd "/Users/rsigalov/Documents/PhD/disaster-risk-revision"

clear all

import delimited estimated_data/final_regression_dfs/rn_prob_decline_ret.csv

gen date = date(month_lead,"YMD")
gen crisis = (date >= td(1,1,2007) & date <= td(31,12,2009))

gen temp = 1
gen dummy_decline_20 = (ret < -0.2)
gen dummy_decline_40 = (ret < -0.4)

* Regression with the full sample:
eststo a, title("Base (All)"): reg dummy_decline_20 rn_prob_20mon, r
eststo b, title("Cluster (All)"): reghdfe dummy_decline_20 rn_prob_20mon, absorb(temp) cluster(secid date)
eststo c, title("No Crisis (All)"): reghdfe dummy_decline_20 rn_prob_20mon if crisis == 0, absorb(temp) cluster(secid date)

esttab a b c using "/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/ind_prob_model_1.tex", cells(b(fmt(a3)) se(fmt(a3) par)) ///
	stats(r2 N, labels(R-squared "N. of obs")) ///
	label varlabels(_cons Constant rn_prob_20mon "20\% rn prob") mtitles booktabs replace

* Regression with the filtered sample:
eststo d, title("Base (Filter)"): reg dummy_decline_20 rn_prob_20mon if dummy_filter == 1, r
eststo e, title("Cluster (Filter)"): reghdfe dummy_decline_20 rn_prob_20mon if dummy_filter == 1, absorb(temp) cluster(secid date)
eststo f, title("No Crisis (Filter)"): reghdfe dummy_decline_20 rn_prob_20mon if crisis == 0 & dummy_filter == 1, absorb(temp) cluster(secid date)

esttab d e f using "/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/ind_prob_model_2.tex", cells(b(fmt(a3)) se(fmt(a3) par)) ///
	stats(r2 N, labels(R-squared "N. of obs")) ///
	label varlabels(_cons Constant rn_prob_20mon "20\% rn prob") mtitles booktabs replace
	
* Repeating the same exercise for risk neutral probability:
eststo g, title("Base (All)"): reg dummy_decline_40 rn_prob_40mon, r
eststo h, title("Cluster (All)"): reghdfe dummy_decline_40 rn_prob_40mon, absorb(temp) cluster(secid date)
eststo i, title("No Crisis (All)"): reghdfe dummy_decline_40 rn_prob_40mon if crisis == 0, absorb(temp) cluster(secid date)

esttab g h i using "/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/ind_prob_model_3.tex", cells(b(fmt(a3)) se(fmt(a3) par)) ///
	stats(r2 N, labels(R-squared "N. of obs")) ///
	label varlabels(_cons Constant rn_prob_40mon "40\% rn prob") mtitles booktabs replace

eststo j, title("Base (Filter)"): reg dummy_decline_40 rn_prob_40mon if dummy_filter == 1, r
eststo k, title("Cluster (Filter)"): reghdfe dummy_decline_40 rn_prob_40mon if dummy_filter == 1, absorb(temp) cluster(secid date)
eststo l, title("No Crisis (Filter)"): reghdfe dummy_decline_40 rn_prob_40mon if crisis == 0 & dummy_filter == 1, absorb(temp) cluster(secid date)

esttab j k l using "/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/ind_prob_model_4.tex", cells(b(fmt(a3)) se(fmt(a3) par)) ///
	stats(r2 N, labels(R-squared "N. of obs")) ///
	label varlabels(_cons Constant rn_prob_40mon "40\% rn prob") mtitles booktabs replace
