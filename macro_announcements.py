import pandas as pd
import numpy as np
from statsmodels.iolib.summary2 import summary_col

def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.975)) & (x >= np.quantile(x, 0.025))])

def estimate_reaction(int_d, int_spx, ann_df, days_before, days_after):
    before_list = []
    after_list = []
    before_spx_list = []
    after_spx_list = []
    for i_ann in range(ann_df.shape[0]):
        print("Announcement %d out of %d" % (i_ann, ann_df.shape[0]))
        ann_date = ann_df.ann_date.iloc[i_ann]
            
        begin_date = ann_date + pd.offsets.DateOffset(-days_before)
        end_date = ann_date + pd.offsets.DateOffset(days_after)
        
        # Calcuating individual disaster measure before and after announcements
        int_d_sub = int_d[(int_d.date >= begin_date) & (int_d.date <= end_date)]
        int_d_sub_before = int_d_sub[int_d_sub.date < ann_date]
        int_d_sub_after = int_d_sub[int_d_sub.date >= ann_date]
        
        if (int_d_sub_before.shape[0] > 0) & (int_d_sub_after.shape[0] > 0):
            before_list.append(mean_with_truncation(np.array(int_d_sub_before["D_clamp"])))
            after_list.append(mean_with_truncation(np.array(int_d_sub_after["D_clamp"])))
        else:
            before_list.append(np.nan)
            after_list.append(np.nan)
            
        int_spx_sub = int_spx[(int_spx.date >= begin_date) & (int_spx.date <= end_date)]
        int_spx_sub_before = int_spx_sub[int_spx_sub.date < ann_date]
        int_spx_sub_after = int_spx_sub[int_spx_sub.date >= ann_date]
        
        if (int_d_sub_before.shape[0] > 0) & (int_d_sub_after.shape[0] > 0):
            before_spx_list.append(mean_with_truncation(np.array(int_spx_sub_before["D_clamp"])))
            after_spx_list.append(mean_with_truncation(np.array(int_spx_sub_after["D_clamp"])))
        else:
            before_spx_list.append(np.nan)
            after_spx_list.append(np.nan)
        
    to_return = ann_df.copy()
    
    to_return["before"] = before_list
    to_return["after"] = after_list
    to_return["before_spx"] = before_spx_list
    to_return["after_spx"] = after_spx_list
    to_return["diff_ind"] = to_return["after"] - to_return["before"]
    to_return["diff_spx"] = to_return["after_spx"] - to_return["before_spx"]
    to_return["surprise_level"] = to_return["actual"] - to_return["survey_average"]

    return to_return
    

# Loading data on macro announcements
ism_pmi_ann = pd.read_csv("data/ism_pmi_announcements.csv")
nonfarm_ann = pd.read_csv("data/nonfarm_payroll_announcements.csv")

ism_pmi_ann = ism_pmi_ann[ism_pmi_ann.survey_average.notnull()]
nonfarm_ann = nonfarm_ann[nonfarm_ann.survey_average.notnull()]

ism_pmi_ann["ann_date"] = pd.to_datetime(ism_pmi_ann["ann_date"], format = "%m/%d/%y")
ism_pmi_ann["ref_month"] = pd.to_datetime(ism_pmi_ann["ref_month"], format = "%m/%d/%y")

nonfarm_ann["ann_date"] = pd.to_datetime(nonfarm_ann["ann_date"], format = "%m/%d/%y")
nonfarm_ann["ref_month"] = pd.to_datetime(nonfarm_ann["ref_month"], format = "%m/%d/%y")

ism_pmi_ann = ism_pmi_ann[ism_pmi_ann.ann_date <= "2017-12-01"]
nonfarm_ann = nonfarm_ann[nonfarm_ann.ann_date <= "2017-12-01"]

# Loading individual disaster measures:
int_d = pd.DataFrame(columns = ["secid", "date", "D_clamp"])

for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    int_d_to_append = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
    int_d_to_append = int_d_to_append[["secid", "date", "D_clamp"]]
    int_d_to_append["days"] = days
    int_d = int_d.append(int_d_to_append)

int_d["date"] = pd.to_datetime(int_d["date"])
int_d = int_d[int_d.D_clamp.notnull()]

# Loading data on SPX options:
int_spx = pd.DataFrame(columns = ["secid", "date", "D_clamp"])

for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    to_append = pd.read_csv("estimated_data/interpolated_D/int_D_spx_days_" + str(days) + ".csv")
    to_append = to_append[["secid", "date", "D_clamp"]]
    to_append["days"] = days
    int_spx = int_spx.append(to_append)

int_spx["date"] = pd.to_datetime(int_spx["date"])

# Estimating reactions of individual and spx disaster measures:
nonfarm_react_7_7 = estimate_reaction(int_d, int_spx, nonfarm_ann, 7, 7)
nonfarm_react_3_3 = estimate_reaction(int_d, int_spx, nonfarm_ann, 3, 3)
nonfarm_react_1_1 = estimate_reaction(int_d, int_spx, nonfarm_ann, 1, 1)

pmi_react_7_7 = estimate_reaction(int_d, int_spx, ism_pmi_ann, 7, 7)
pmi_react_3_3 = estimate_reaction(int_d, int_spx, ism_pmi_ann, 3, 3)
pmi_react_1_1 = estimate_reaction(int_d, int_spx, ism_pmi_ann, 1, 1)

# Standardizing variables:
for react_df in [nonfarm_react_7_7, nonfarm_react_3_3, nonfarm_react_1_1,
                 pmi_react_7_7, pmi_react_3_3, pmi_react_1_1]:
    for variable in ["diff_ind", "diff_spx", "surprise_level"]:
        react_df.loc[:, variable] = react_df.loc[:, variable] - np.mean(react_df.loc[:, variable])
        react_df.loc[:, variable] = react_df.loc[:, variable]/np.std(react_df.loc[:, variable])

# Saving macro announcement reactions:
nonfarm_react_3_3.to_csv("estimated_data/macro_announcements/nonfarm_react.csv")
pmi_react_3_3.to_csv("estimated_data/macro_announcements/pmi_react.csv")


# Putting everything in a table:
reg_df = nonfarm_react_3_3
reg1 = smf.ols(formula = "diff_ind ~ surprise_level", data = reg_df).fit(cov_type = "HC3")
reg_df = nonfarm_react_3_3[
        ((nonfarm_react_3_3.ann_date >= "2000-01-01") & (nonfarm_react_3_3.ann_date <= "2003-12-31")) |
        ((nonfarm_react_3_3.ann_date >= "2007-01-01") & (nonfarm_react_3_3.ann_date <= "2009-12-31"))]
reg2 = smf.ols(formula = "diff_ind ~ surprise_level", data = reg_df).fit(cov_type = "HC3")
reg_df = nonfarm_react_3_3[
        (nonfarm_react_3_3.surprise_level >= -2) & (nonfarm_react_3_3.surprise_level <= 2) &
        (nonfarm_react_3_3.diff_ind >= -2) & (nonfarm_react_3_3.diff_ind <= 2)]
reg3 = smf.ols(formula = "diff_ind ~ surprise_level", data = reg_df).fit(cov_type = "HC3")
reg_df = nonfarm_react_3_3
reg4 = smf.ols(formula = "diff_spx ~ surprise_level", data = reg_df).fit(cov_type = "HC3")

regs = [reg1, reg2, reg3, reg4]

path = "SS_tables/macro_announcements.tex"
f = open(path, "w")
f.write("\small")
f.write("\\begin{tabular}{lcccc}\n")
f.write("\\toprule \n")
f.write("  & \\multicolumn{3}{c}{Individual} & SPX \\\\ \n")
f.write("\cline{2-5} \n")
f.write(" Sample: & Full & Crisis & No Outliers & Full \\\\ \n")
f.write("  & (1) & (2) & (3) & (4) \\\\ \\\\[-1.8ex] \n")
f.write("\hline \\\\[-1.8ex] \n")
        
f.write("Surprise & {:.3f}   & {:.3f}   & {:.3f} & {:.3f}  \\\\ \n".format(*[x.params[1] for x in regs]))
f.write("           & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f})  \\\\ \\\\[-1.8ex] \n".format(*list([x.bse[1] for x in regs])))

f.write("Constant & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  \\\\ \n".format(*[x.params[0] for x in regs]))
f.write("           & ({:.3f}) & ({:.3f}) & ({:.3f}) & ({:.3f})  \\\\ \\\\[-1.8ex] \n".format(*[x.bse[0] for x in regs]))

f.write("$R^2$      & {:.3f}  & {:.3f} & {:.3f} & {:.3f}  \\\\ \\\\[-1.8ex] \n".format(*[x.rsquared for x in regs]))

f.write("\\bottomrule \n")
f.write("\end{tabular}")  
f.close()




# Looking at the scatterplot:
# 1. Getting outliers:
outliers_diff = list(nonfarm_react_3_3.sort_values("diff_ind").ann_date[0:5])
outliers_diff = outliers_diff + list(nonfarm_react_3_3.sort_values("diff_ind", ascending = False).ann_date[0:7])

outliers_surprise = list(nonfarm_react_3_3.sort_values("surprise_level").ann_date[0:6])
outliers_surprise = outliers_surprise + list(nonfarm_react_3_3.sort_values("surprise_level", ascending = False).ann_date[0:6])

outliers = outliers_surprise + outliers_diff

# 2. Plotting and setting dates to outliers:
nonfarm_ann_out = nonfarm_react_3_3[nonfarm_react_3_3.ann_date.isin(outliers)]
nonfarm_ann_not_out = nonfarm_react_3_3[~nonfarm_react_3_3.ann_date.isin(outliers)]

fig, ax = plt.subplots(figsize = (8,6))
ax.scatter(nonfarm_ann_not_out["surprise_level"], nonfarm_ann_not_out["diff_ind"], color = "0.85")
ax.scatter(nonfarm_ann_out["surprise_level"], nonfarm_ann_out["diff_ind"], color = "0.1")

for i in range(nonfarm_ann_out.shape[0]):
    row = nonfarm_ann_out.iloc[i]
    ax.annotate(
            row.ann_date.strftime('%Y-%m-%d'), 
            (row.surprise_level, row.diff_ind))
    
ax.set_xlabel("Nonfarm Payroll Surprise (Standardized)")
ax.set_ylabel("Change in Individual Disaster Measure (Standardized)")
plt.tight_layout()
plt.savefig("SS_figures/macro_announcements.pdf")


# Removing observations with either x or y beyond 2 st.dev from the mean
# and running the same regressions to see if there is an effect if we 
# remove outliers
not_out_df = nonfarm_react_3_3[(nonfarm_react_3_3.surprise_level <= 2) & (nonfarm_react_3_3.surprise_level >= -2) &
                  (nonfarm_react_3_3.diff_ind <= 2) & (nonfarm_react_3_3.diff_ind >= -2)]

smf.ols(formula = "diff_ind ~ surprise_level", data = nonfarm_react_3_3).fit(cov_type = "HC3").summary()
smf.ols(formula = "diff_ind ~ surprise_level", data = not_out_df).fit(cov_type = "HC3").summary()

# Plotting side by side:
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize = (10,5))
sns.regplot(data = nonfarm_react_3_3, x = "surprise_level", y = "diff_ind", ax = ax1)
sns.regplot(data = not_out_df, x = "surprise_level", y = "diff_ind", ax = ax2)












#####################################################
## Doing option pricing via Fourier transform
#####################################################
#I = complex(0,1)
#T = 1
#sigma = 0.18
#mu = 0.05
#alpha = 10
#
#def psi(v):
#    return np.exp(-T*sigma**2/2 * (v**2 + 1j*v))
#
#def Ftransform(v):
#    return psi(v - 1j*alpha)/(1j*v + alpha)
#
## Parameters for DFT:
#
#L = 1.5*2**10
#N = 2**10
#Delta = L/(N-1)
#v = -L/2 + Delta * np.arange(N)
#k0 = -2
#k = k0 + 2*math.pi/(N*Delta) * np.arange(N)
#
## Simpson rule weights:
#w = np.zeros(v.shape)
#w[0] = 1/3
#for k in range(1,int(N/2),1):
#    w[2*k-1] = 4/3
#    w[2*k] = 2/3
#    
#w[-1] = 4/3
#
## Constructing sequence for FFT:
#m = np.array(range(N))
#a = np.exp(-1j * m * Delta * k0) * w * Ftransform(v)
#
#zhat = L/(2 * math.pi * (N-1)) * np.exp(1j * k * L/2) * np.fft.fft(a)
#
#
#
#
##### Using somebody's code for calculation option prices:
#def psi(v):
#    return np.exp(-T*sigma**2/2 * (v**2 + I*v))
#
#def carr_madan_fft_call_pricer(N, d_u, alpha, r, t, S0, q, chf_ln_st):
#    d_k = 2 * np.pi / (N * d_u)
#    beta = np.log(S0) - d_k * N / 2
#    u_arr = np.arange(N) * d_u
#    k_arr = beta + np.arange(N) * d_k
#    delta_arr = np.zeros(N)
#    delta_arr[0] = 1
#    w_arr = d_u / 3 * (3 + (-1) ** (np.arange(N) + 1) - delta_arr)
#    call_chf = (np.exp(-r * t) / ((alpha + 1j * u_arr) * (alpha + 1j * u_arr + 1))) * chf_ln_st(
#        u_arr - (alpha + 1) * 1j)
#    x_arr = np.exp(-1j * beta * u_arr) * call_chf * w_arr
#    fft_prices = (np.fft.fft(x_arr))
#    call_prices = (np.exp(-alpha * k_arr) / np.pi) * fft_prices.real
#    return np.exp(k_arr), call_prices
#
#N = 2 ** 10
#d_u = 0.01
#alpha = 1
#S0 = 100
#t = 1
#r = 0.05
#q = 0
#sigma = 0.3
#
#k, C = carr_madan_fft_call_pricer(N, d_u, alpha, r, t, S0, q, psi)
#
#
#
##np.fft.fft()
##
##np.fft.fftfreq(n)
#
#
#
#
#
#
#


