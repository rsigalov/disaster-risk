clear *
set more off

* Setting up working directory
global root = "/Users/rsigalov/Documents/PhD/disaster-risk-revision"
cd "${root}/"

* Reading nonfarm
insheet using "${root}/estimated_data/macro_announcements/nonfarm_react.csv", clear
gen date = date(ann_date, "YMD")
format date %td
gen crisis = ((date >= td(1,1,2000)) & (date < td(1,1,2004))) | ((date >= td(1,1,2007)) & (date < td(1,1,2010)))
gen not_out = (diff_ind >= -2) & (diff_ind <= 2) & (surprise_level >= -2) & (surprise_level <= 2)

eststo a, title("Ind, Full"): reg diff_ind surprise_level, r 
eststo b, title("Ind, Crisis"): reg diff_ind surprise_level if crisis == 1, r 
eststo c, title("Ind, Not Out"): reg diff_ind surprise_level if not_out == 1, r 
eststo d, title("SPX"): reg diff_spx surprise_level, r 

esttab a b c d using "SS_tables/macro_announcements.tex", cells(b(fmt(a3)) se(fmt(a3) par)) ///
	stats(r2 N, labels(R-squared "N. of obs")) ///
	label varlabels(_cons Constant surprise_level Surprise) booktabs replace
