library("ggplot2")
library("data.table")
library("dplyr")
library("reshape")
library("readr")
library("stargazer")
setwd("~/Documents/PhD/disaster-risk-revision/")

############################################################
# Comparing loadings on fctors for different maturities:
############################################################
reg_res <- read_csv("estimated_data/disaster_sorts/reg_results_agg.csv")
reg_res_filter <- reg_res %>%
  filter(level %in% c("union_cs"), variable %in% c("D_clamp"), port == "ew_diff")

# Adding standard error bands
reg_res_ci <- reg_res_filter %>%
  mutate(alpha = alpha_se*2, beta_MKT = beta_MKT_se*2, beta_SMB = beta_SMB_se*2,
         beta_HML = beta_HML_se*2, beta_CMA=beta_CMA_se*2, beta_RMW=beta_RMW_se*2) %>%
  select(FF, days, alpha, beta_MKT, beta_SMB, beta_HML, beta_CMA, beta_RMW) %>%
  dplyr::rename(MKT = beta_MKT, HML = beta_HML, SMB = beta_SMB, CMA = beta_CMA, RMW = beta_RMW) %>%
  data.table() %>% melt(id.vars = c("days", "FF")) %>%
  dplyr::rename(ci = value)
  
df_to_plot <-reg_res_filter %>%
  select(FF, days, alpha, beta_MKT, beta_SMB, beta_HML, beta_CMA, beta_RMW) %>%
  dplyr::rename(MKT = beta_MKT, HML = beta_HML, SMB = beta_SMB, CMA = beta_CMA, RMW = beta_RMW) %>%
  data.table() %>% melt(id.vars = c("days", "FF")) %>%
  left_join(reg_res_ci, by = c("days", "FF", "variable"))

ggplot(df_to_plot, aes(x = factor(FF), y = value, fill = factor(days))) +
  geom_bar(position=position_dodge(), stat="identity") +
  geom_errorbar(aes(ymin=value-ci, ymax=value+ci),width=.1, alpha = 0.6, position=position_dodge(.9)) +
  facet_grid(variable ~ ., scales = "free_y") +
  scale_fill_brewer(palette="Paired", name = "Days") + theme_minimal() +
  geom_hline(yintercept = 0, size = 0.2) +
  xlab("Factors included") + ylab("")


# Now looking at market-to-book and op. profitability:
bm <- read_csv("estimated_data/disaster_sorts/port_sort_bm_agg.csv") %>% 
  dplyr::rename(date = X1)
op <- read_csv("estimated_data/disaster_sorts/port_sort_op_agg.csv") %>%
  dplyr::rename(date = X1)

# Averaging:
bm_av <- bm %>%
  group_by(variable, days) %>%
  summarize_all(mean) %>% ungroup()

op_av <- op %>%
  group_by(variable, days) %>%
  summarize_all(mean) %>% ungroup()

# Melting, merging and prepare for plotting:
bm_av_melt <- bm_av %>% 
  filter(variable == "D_clamp") %>%
  select(days, ew_1, ew_2, ew_3, ew_4, ew_5) %>%
  data.table() %>% melt(id.vars = "days") %>%
  mutate(char = "Book-to-Market")

op_av_melt <- op_av %>% 
  filter(variable == "D_clamp") %>%
  select(days, ew_1, ew_2, ew_3, ew_4, ew_5) %>%
  data.table() %>% melt(id.vars = "days") %>%
  mutate(char = "Operating Profitability")

av_melt <- rbind(bm_av_melt,op_av_melt)

# Plotting
ggplot(av_melt, aes(x = factor(days), y = value, fill = variable)) +
  geom_bar(position=position_dodge(), stat="identity") +
  facet_grid(char ~ ., scales = "free_y") +
  scale_fill_brewer(palette="Paired", name = "Portfolio") + theme_minimal()

ggplot(av_melt %>% mutate(variable = as.numeric(substr(variable, 4,4))), 
       aes(x = variable, y = value)) +
  geom_point() + geom_line() +
  facet_grid(char ~ days, scales = "free_y") +
  theme_minimal() + xlab("Portfolio") + ylab("")

# Plotting time series of operating profitability and
# book to market at formation for ew_1 and ew_2:
bm_melt <- bm %>%
  filter(variable == "D_clamp", level == "union_cs", days == -99) %>%
  select(date, days, ew_1, ew_5) %>%
  data.table() %>% melt(id.vars = c("date", "days")) %>%
  mutate(char = "Book-to-Market")

op_melt <- op %>%
  filter(variable == "D_clamp", level == "union_cs", days == -99) %>%
  select(date, days, ew_1, ew_5) %>%
  data.table() %>% melt(id.vars = c("date", "days")) %>%
  mutate(char = "Operating Profitability")

rbind(bm_melt, op_melt) %>%
  ggplot(aes(x = date, y=value, color = variable)) +
  geom_line() + facet_grid(char~., scales = "free_y") +
  xlab("") + ylab("") + theme_minimal() +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y")



############################################################
# Comparing loadings and characteristics for sorts on 
# individual companies
############################################################
reg_res <- read_csv("estimated_data/disaster_sorts/reg_results_ind.csv")
reg_res_filter <- reg_res %>%
  filter(variable %in% c("D_clamp"), port == "ew_diff")

# Adding standard error bands
reg_res_ci <- reg_res_filter %>%
  mutate(alpha = alpha_se*2, beta_MKT = beta_MKT_se*2, beta_SMB = beta_SMB_se*2,
         beta_HML = beta_HML_se*2, beta_CMA=beta_CMA_se*2, beta_RMW=beta_RMW_se*2) %>%
  select(FF, days, alpha, beta_MKT, beta_SMB, beta_HML, beta_CMA, beta_RMW) %>%
  dplyr::rename(MKT = beta_MKT, HML = beta_HML, SMB = beta_SMB, CMA = beta_CMA, RMW = beta_RMW) %>%
  data.table() %>% melt(id.vars = c("days", "FF")) %>%
  dplyr::rename(ci = value)

df_to_plot <-reg_res_filter %>%
  select(FF, days, alpha, beta_MKT, beta_SMB, beta_HML, beta_CMA, beta_RMW) %>%
  dplyr::rename(MKT = beta_MKT, HML = beta_HML, SMB = beta_SMB, CMA = beta_CMA, RMW = beta_RMW) %>%
  data.table() %>% melt(id.vars = c("days", "FF")) %>%
  left_join(reg_res_ci, by = c("days", "FF", "variable"))

ggplot(df_to_plot, aes(x = factor(FF), y = value, fill = factor(days))) +
  geom_bar(position=position_dodge(), stat="identity") +
  geom_errorbar(aes(ymin=value-ci, ymax=value+ci),width=.1, alpha = 0.6, position=position_dodge(.9)) +
  facet_grid(variable ~ ., scales = "free_y") +
  scale_fill_brewer(palette="Paired", name = "Days") + theme_minimal() +
  geom_hline(yintercept = 0, size = 0.2) +
  xlab("Factors included") + ylab("")


# Now looking at market-to-book and op. profitability:
bm <- read_csv("estimated_data/disaster_sorts/port_sort_bm_ind.csv") %>% 
  dplyr::rename(date = X1)
op <- read_csv("estimated_data/disaster_sorts/port_sort_op_ind.csv") %>%
  dplyr::rename(date = X1)

# Averaging:
bm_av <- bm %>%
  group_by(variable, days) %>%
  summarize_all(mean) %>% ungroup()

op_av <- op %>%
  group_by(variable, days) %>%
  summarize_all(mean) %>% ungroup()

# Melting, merging and prepare for plotting:
bm_av_melt <- bm_av %>% 
  filter(variable == "D_clamp") %>%
  select(days, ew_1, ew_2, ew_3, ew_4, ew_5) %>%
  data.table() %>% melt(id.vars = "days") %>%
  mutate(char = "Book-to-Market")

op_av_melt <- op_av %>% 
  filter(variable == "D_clamp") %>%
  select(days, ew_1, ew_2, ew_3, ew_4, ew_5) %>%
  data.table() %>% melt(id.vars = "days") %>%
  mutate(char = "Operating Profitability")

av_melt <- rbind(bm_av_melt,op_av_melt)

# Plotting
ggplot(av_melt, aes(x = factor(days), y = value, fill = variable)) +
  geom_bar(position=position_dodge(), stat="identity") +
  facet_grid(char ~ ., scales = "free_y") +
  scale_fill_brewer(palette="Paired", name = "Portfolio") + theme_minimal()

ggplot(av_melt %>% mutate(variable = as.numeric(substr(variable, 4,4))), 
       aes(x = variable, y = value)) +
  geom_point() + geom_line() +
  facet_grid(char ~ days, scales = "free_y") +
  theme_minimal() + xlab("Portfolio") + ylab("")


# Plotting time series of operating profitability and
# book to market at formation for ew_1 and ew_2:
bm_melt <- bm %>%
  filter(variable == "D_clamp", days == -99) %>%
  select(date, days, ew_1, ew_5) %>%
  data.table() %>% melt(id.vars = c("date", "days")) %>%
  mutate(char = "Book-to-Market")

op_melt <- op %>%
  filter(variable == "D_clamp", days == -99) %>%
  select(date, days, ew_1, ew_5) %>%
  data.table() %>% melt(id.vars = c("date", "days")) %>%
  mutate(char = "Operating Profitability")

rbind(bm_melt, op_melt) %>%
  ggplot(aes(x = date, y=value, color = variable)) +
  geom_line() + facet_grid(char~., scales = "free_y") +
  xlab("") + ylab("") + theme_minimal() +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y")
