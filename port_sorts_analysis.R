library("ggplot2")
library("data.table")
library("dplyr")
library("reshape")
library("readr")
setwd("~/Documents/PhD/disaster-risk-revision/")

# Loading returns on portfolios sorted by individual disaster measure:
ret_ind <- read_csv("estimated_data/ind_disaster_sorts/port_sort_ret.csv") %>%
  dplyr::rename(date = X1)
ret_agg <- read_csv("estimated_data/ind_disaster_sorts/port_sort_agg_ret.csv") %>%
  dplyr::rename(date = X1)

bm_ind <- read_csv("estimated_data/ind_disaster_sorts/port_sort_bm.csv") %>%
  dplyr::rename(date = X1)
bm_agg <- read_csv("estimated_data/ind_disaster_sorts/port_sort_agg_bm.csv") %>%
  dplyr::rename(date = X1)

op_ind <- read_csv("estimated_data/ind_disaster_sorts/port_sort_op.csv") %>%
  dplyr::rename(date = X1)
op_agg <- read_csv("estimated_data/ind_disaster_sorts/port_sort_agg_op.csv") %>%
  dplyr::rename(date = X1)

####################################################################
# Comparing averages
####################################################################
ret_ind_means <- ret_ind %>%
  filter(days %in% c(30,60,120,180)) %>%
  select(days, variable, vw_1, vw_2, vw_3, vw_4, vw_5) %>%
  group_by(variable, days) %>%
  summarize_all(mean, na.rm = T)

ret_agg_means <- ret_agg %>%
  select(days, variable, vw_1, vw_2, vw_3, vw_4, vw_5) %>%
  group_by(variable, days) %>%
  summarize_all(mean, na.rm = T)

# Combining them together:
convert_for_plotting <- function(df, port_type_str) {
  # Subsetting columns
  df <- df %>%
    filter(days %in% c(30,60,120,180)) %>%
    filter(variable %in% c("D_clamp", "rn_prob_20", "rn_prob_80"))
  
  # Picking the right columns according to portfolio type specified
  if (port_type_str == "VW") {
    df <- df %>% select(days, variable, vw_1, vw_2, vw_3, vw_4, vw_5)
  } else if (port_type_str == "EW") {
    df <- df %>% select(days, variable, ew_1, ew_2, ew_3, ew_4, ew_5)
  } else {
    stop(paste("No portfolio type", port_type_str))
  }
  
  # Melting the dataframes
  df %>%
    group_by(variable, days) %>%
    mutate_all(function(x) x*12) %>%
    summarize_all(mean, na.rm = T) %>%
    data.table() %>% melt(id.vars = c("days", "variable")) %>%
    dplyr::rename(port = variable.1) %>% # renaming variable for portfolios
    mutate(port = as.numeric(substr(port, 4, 4))) %>%  # Converting portfolio names to numbers
    mutate(port_type = port_type_str) %>% data.table()
}

comp_port_plots <- function(df_ind, df_agg, port_type_str) {
  ret_melt_ew_ind <- convert_for_plotting(df_ind, port_type_str) %>%
    mutate(sorting_var_type = "Individual")
  
  # Renaming portfolios for beta sorts: call the most exposed to disaster as 5 and least as 1
  ret_melt_ew_agg <- convert_for_plotting(df_agg, port_type_str) %>%
    mutate(port = 6 - port) %>%
    mutate(sorting_var_type = "Beta")
  
  ggplot(rbind(ret_melt_ew_agg, ret_melt_ew_ind), aes(x = port, y = value, color = sorting_var_type)) +
    geom_line() +
    facet_grid(variable ~ days)
}

comp_port_plots(ret_ind, ret_agg, "VW")
comp_port_plots(ret_ind, ret_agg, "EW")

comp_port_plots(bm_ind, bm_agg, "VW")
comp_port_plots(bm_ind, bm_agg, "EW")

comp_port_plots(op_ind, op_agg, "VW")
comp_port_plots(op_ind, op_agg, "EW")

# Generating tables:
ret_ind %>%
  select(-c(date, ew_count, vw_count)) %>%
  group_by(variable, days) %>%
  summarize_all(mean, na.rm = T)

####################################################################
# Looking at time series
####################################################################
port_type_str = "EW"
df_to_plot <- ret_agg %>%
  filter(days %in% c(30,60,120,180)) %>%
  filter(variable %in% c("D_clamp", "rn_prob_20", "rn_prob_80"))
  if (port_type_str == "VW") {
    df_to_plot <- df_to_plot %>% select(date, days, variable, vw_1, vw_2, vw_3, vw_4, vw_5)
  } else if (port_type_str == "EW") {
    df_to_plot <- df_to_plot %>% select(date, days, variable, ew_1, ew_2, ew_3, ew_4, ew_5)
  } else {
    stop(paste("No portfolio type", port_type_str))
  }
df_to_plot <- df_to_plot %>%
  data.table() %>% melt(id.vars = c("date", "days", "variable")) %>%
  dplyr::rename(port = variable.1) %>% # renaming variable for portfolios
  mutate(port = as.numeric(substr(port, 4, 4))) %>%  # Converting portfolio names to numbers
  mutate(port_type = port_type_str) %>% 
  arrange(variable, days, port, date) %>%
  group_by(variable, days, port) %>%
  mutate(value = value + 1) %>%
  mutate(position = cumprod(value))

ggplot(df_to_plot %>% filter(port %in% c(1,5)), aes(x = date, y = position, color = factor(port))) +
  geom_line() +
  facet_grid(variable ~ days)
         





ret_ind %>%
  filter(days %in% c(30,60,120,180)) %>%
  mutate(diff = 12*(ew_5 - ew_1)) %>%
  group_by(variable, days) %>%
  summarize(mean = mean(diff))

ret_agg %>%
  mutate(diff = 12*(ew_1 - ew_5)) %>%
  group_by(variable, days) %>%
  summarize(mean = mean(diff))

# Comparing return on HML strategies for individually 
# vs aggregate exposure
ret_agg %>%
  filter(days %in% c(30, 180)) %>%
  select(-c(vw_count, ew_count)) %>%
  select(-c(vw_1, vw_2, vw_3, vw_4, vw_5)) %>%
  mutate(diff = ew_5 - ew_1) %>%
  data.table() %>%
  melt(id.vars = c("date", "variable", "days")) %>%
  dplyr::rename(port = variable.1) %>%
  mutate(port = substr(port, 4, 4)) %>%
  as_tibble() %>%
  mutate(value = value + 1) %>%
  arrange(variable, days, date) %>%
  group_by(variable, days, port) %>%
  mutate(position = cumprod(value)) %>%
  ggplot(aes(x = date, y = position, color = port)) +
  geom_line() +
  facet_grid(variable~days)




ret_ind <- read_csv("estimated_data/ind_disaster_sorts/port_sort_op.csv")
ret_agg <- read_csv("estimated_data/ind_disaster_sorts/port_sort_agg_op.csv")
ret_ind %>% 
  select(days, variable, ew_1, ew_2, ew_3, ew_4, ew_5) %>%
  group_by(variable, days) %>%
  summarize_all(mean, na.rm = T) %>%
  data.table() %>%
  melt(id.vars = c("variable", "days")) %>%
  ggplot(aes(x = variable.1, y = value, color = days)) +
  geom_line() + facet_wrap(~variable)

ret_df %>% 
  select(days, variable, vw_1, vw_2, vw_3, vw_4, vw_5) %>%
  group_by(variable, days) %>%
  summarize_all(mean, na.rm = T) %>%
  data.table() %>%
  melt(id.vars = c("variable", "days")) %>%
  ggplot(aes(x = variable.1, y = value, color = days)) +
  geom_line() + facet_wrap(~variable)

ret_df %>%
  filter(days == -99) %>%
  select(ew_1, ew_2, ew_3, ew_4, ew_5) %>%
  summarize_all(mean, na.rm = T)

ret_df %>%
  filter(days == -99) %>%
  select(ew_1, ew_2, ew_3, ew_4, ew_5) %>%
  summarize_all(mean, na.rm = T)





