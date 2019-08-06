print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation
using CSV
using Dates
using Plots
cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")


# 1. Reading files from the directory with disaster D-measure and monthly
# (standardized decline probabilities).
dir_path = "estimated_data/V_IV/"
file_list = readdir(dir_path)
file_list = file_list[map(x -> occursin("var_ests_final", x), file_list)]

df = CSV.read(string(dir_path, file_list[1]); datarow = 2, delim = ",")

for i in 1:length(file_list)
    if i == 1
        global df = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        df = df[:, [:secid, :date, :T]]
    else
        df_append = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        df_append = df_append[:, [:secid, :date, :T]]
        append!(df, df_append)
    end
end

# Calculating the share of companies on each day that have maturities
# covering 30 days (or other prespecified number of days):
function calc_mon_share_inside(df, days)
    inside_days = by(df, [:date, :secid],
        inside_ind = :T => x -> (maximum(x) >= days/365) & (minimum(x) <= days/365))
    inside_days[:date_mon] = Dates.lastdayofmonth.(inside_days[:date])
    inside_num = by(inside_days, :date_mon, inside_num = :inside_ind => sum)
    tot_num = by(inside_days, :date_mon, tot_num = :inside_ind => length)
    share_inside = join(inside_num, tot_num, on = [:date_mon])

    share_inside[:share_inside] = share_inside.inside_num./share_inside.tot_num
    sort!(share_inside, :date_mon)
    return share_inside
end

days = 30

df_share_30 = calc_mon_share_inside(df, days)
Plots.plot(df_share_30.date_mon, df_share_30.share_inside, label = "All options",
    title = string(days, " days"), legend = :topleft)

# Loading ranking of secids in terms of volume and number of options
# and looking how the share looks among them
secid_rank = CSV.read("data/secid_option_ranking.csv")
secid_rank = secid_rank[.!ismissing.(secid_rank.number_rank), :]

secid_top_cnt = secid_rank[secid_rank.number_rank .<= 100, :secid]
secid_top_volume = secid_rank[secid_rank.volume_rank .<= 100, :secid]

df_top = df[map(x -> x in secid_top_cnt, df.secid), :]
df_share_top_30 = calc_mon_share_inside(df_top, days)
Plots.plot!(df_share_top_30.date_mon, df_share_top_30.share_inside, label = "Top by count")

df_top = df[map(x -> x in secid_top_volume, df.secid), :]
df_share_top_30 = calc_mon_share_inside(df_top, days)
Plots.plot!(df_share_top_30.date_mon, df_share_top_30.share_inside, label = "Top by volume")



inside_days = by(df, [:date, :secid],
    inside_ind = :T => x -> (maximum(x) >= days/365) & (minimum(x) <= days/365))

inside_cnt = by(inside_days, :date, inside_cnt = :inside_ind => sum)
sort!(inside_cnt, :date)
Plots.plot(inside_cnt.date, inside_cnt.inside_cnt, label = "Number of firms with 30 days inside", legend = :topleft)

tot_num = by(inside_days, :date, tot_num = :inside_ind => length)
sort!(tot_num, :date)
Plots.plot!(tot_num.date, tot_num.tot_num, label = "Total number of firms")

inside_days[:date_mon] = Dates.lastdayofmonth.(inside_days[:date])
inside_num = by(inside_days, :date_mon, inside_num = :inside_ind => sum)
tot_num = by(inside_days, :date_mon, tot_num = :inside_ind => length)
share_inside = join(inside_num, tot_num, on = [:date_mon])

share_inside[:share_inside] = share_inside.inside_num./share_inside.tot_num
sort!(share_inside, :date_mon)
share_inside
sort!(df[df.secid .== 101125,:], :date)



# Calculating how many firs have complete panel:
inside_days[:date_mon] = Dates.lastdayofmonth.(inside_days.date)
secid_mon_ind = by(inside_days, [:secid, :date_mon], inside_ind = :inside_ind => x -> sum(x) >= 1)
mon_at_least_one = by(secid_mon_ind, :date_mon, num_firms = :inside_ind => sum)
# sort!(mon_at_least_one, :date_mon)
# Plots.plot(mon_at_least_one.date_mon, mon_at_least_one.num_firms)

# Calculating number of firms that have a full panel (or more than a share of it):
firms_num_months_inside = by(secid_mon_ind, :secid, num_months = :inside_ind => sum)
sort!(firms_num_months_inside, :num_months, rev = true)
Plots.plot(firms_num_months_inside.num_months)
tot_num_months = length(unique(secid_mon_ind.date_mon))
Plots.hline!([tot_num_months * 0.8])

secid_mon_ind[(secid_mon_ind.secid .== 109348) .& (.!secid_mon_ind.inside_ind),:]



################################################################################
#### Secid with the most observations that allows us to interpolate:
################################################################################
# Calculating minimum and maximum maturities for each (secid, date):
min_max_mat = by(df, [:secid, :date], [:T] => x -> (min_mat = minimum(x.T), max_mat = maximum(x.T)))

# Calculating number of observations for each secid where 30 days is inside
# the maturity list:
num_obs_per_secid = by(
    min_max_mat[(min_max_mat.min_mat .<= 29/365) .& (min_max_mat.max_mat .>= 29/365),:],
    :secid, :secid => length)
sort!(num_obs_per_secid, :secid_length, rev = true)

# Calculating ranking for my number of observations:
num_obs_per_secid[:rank] = 1:size(num_obs_per_secid)[1]

# Loading Emil's data on number of observations:
emils_num_obs = CSV.read("/Users/rsigalov/Desktop/secid_counts.csv")
sort!(emils_num_obs, :c, rev = true)
names!(emils_num_obs, [:secid, :emils_count])
emils_num_obs[:emils_rank] = 1:size(emils_num_obs)[1]

# Merging the two rankings:
num_obs = join(emils_num_obs, num_obs_per_secid, on = :secid, kind = :left)

################################################################################
#### Calculating number of months for each secid where it has at least X days
#### with interpolation data
################################################################################
days = 45
X = 10

inside_day_ind = by(df, [:secid, :date],
    inside_ind = :T => x -> (minimum(x) <= (days-1)/365) & (maximum(x) >= (days-1)/365))
inside_day_ind[:date_mon] = Dates.lastdayofmonth.(inside_day_ind.date)
inside_min_mon_ind = by(inside_day_ind, [:secid, :date_mon],
    min_inside = :inside_ind => x -> sum(x) >= 5)
inside_tot_secid = by(inside_min_mon_ind, :secid, tot_months = :min_inside => sum)
sort!(inside_tot_secid,:tot_months, rev = true)

# Plotting figure
df_to_plot = by(inside_min_mon_ind, :date_mon, num_firms = :min_inside => sum)
sort!(df_to_plot, :date_mon)
Plots.plot!(df_to_plot.date_mon, df_to_plot.num_firms,
    title = string("# of firms with at least X days where smile straddle d days"),
    label = ("X = 10, d = 45"), legend = :topleft)
