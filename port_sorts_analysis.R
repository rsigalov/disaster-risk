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

switch_port_name <- function(x){
  switch(x, 
        "ew_1" = "ew_5", "ew_2" = "ew_4", "ew_3" = "ew_3", "ew_4" = "ew_2", "ew_5" = "ew_1", "ew_diff" = "ew_diff",
        "vw_1" = "vw_5", "vw_2" = "vw_4", "vw_3" = "vw_3", "vw_4" = "vw_2", "vw_5" = "vw_1", "vw_diff" = "vw_diff")
}

reg_res$port <- as.character(sapply(X = reg_res$port, FUN = switch_port_name))

for (port_ in c("ew_diff", "ew_1","vw_diff","vw_1", "vw_5", "ew_5")) {
  reg_res_filter <- reg_res %>%
    filter(variable %in% c("D_clamp"), port == port_)
  
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
  
  # pdf(paste("estimated_data/disaster_sorts/images/ff_reg_agg_", port_,".pdf",sep = ""), height = 8, width = 7)
  plot_tmp <- ggplot(df_to_plot, aes(x = factor(FF), y = value, fill = factor(days))) +
      geom_bar(position=position_dodge(), stat="identity") +
      geom_errorbar(aes(ymin=value-ci, ymax=value+ci),width=.1, alpha = 0.6, position=position_dodge(.9)) +
      facet_grid(variable ~ ., scales = "free_y") +
      scale_fill_brewer(palette="Paired", name = "Days") + theme_minimal() +
      geom_hline(yintercept = 0, size = 0.2) +
      xlab("Factors included") + ylab("") +
    ggtitle(paste("Aggregate, ", port_, sep = ""))
  ggsave(filename = paste("estimated_data/disaster_sorts/images/ff_reg_agg_", port_,".pdf",sep = ""), 
         plot = plot_tmp, device = "pdf", height = 7, width = 7)
  # dev.off()
}

# Now looking at market-to-book and op. profitability:
bm <- read_csv("estimated_data/disaster_sorts/port_sort_bm_agg.csv") %>% 
  dplyr::rename(date = X1)
op <- read_csv("estimated_data/disaster_sorts/port_sort_op_agg.csv") %>%
  dplyr::rename(date = X1)

bm <- bm %>%
  dplyr::rename(ew_1 = ew_5, ew_2 = ew_4, ew_4 = ew_2, ew_5 = ew_1,
                vw_1 = vw_5, vw_2 = vw_4, vw_4 = vw_2, vw_5 = vw_1)
op <- op %>%
  dplyr::rename(ew_1 = ew_5, ew_2 = ew_4, ew_4 = ew_2, ew_5 = ew_1,
                vw_1 = vw_5, vw_2 = vw_4, vw_4 = vw_2, vw_5 = vw_1)


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
# ggplot(av_melt, aes(x = factor(days), y = value, fill = variable)) +
#   geom_bar(position=position_dodge(), stat="identity") +
#   facet_grid(char ~ ., scales = "free_y") +
#   scale_fill_brewer(palette="Paired", name = "Portfolio") + theme_minimal()

pdf("estimated_data/disaster_sorts/images/bm_op_port_agg.pdf", height = 5, width = 9)
  ggplot(av_melt %>% mutate(variable = as.numeric(substr(variable, 4,4))), 
         aes(x = variable, y = value)) +
    geom_point() + geom_line() +
    facet_grid(char ~ days, scales = "free_y") +
    theme_minimal() + xlab("Portfolio") + ylab("") +
    ggtitle("Aggregate")
dev.off()

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

pdf("estimated_data/disaster_sorts/images/bm_op_ts_agg.pdf", height = 5, width = 9)
  rbind(bm_melt, op_melt) %>%
    ggplot(aes(x = date, y=value, color = variable)) +
    geom_line() + facet_grid(char~., scales = "free_y") +
    xlab("") + ylab("") + theme_minimal() +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    ggtitle("Aggregate")
dev.off()



############################################################
# Comparing loadings and characteristics for sorts on 
# individual companies
############################################################
reg_res <- read_csv("estimated_data/disaster_sorts/reg_results_ind.csv")
for (port_ in c("ew_diff", "ew_1","ew_2","ew_3","ew_4", "ew_5")) {
  reg_res_filter <- reg_res %>%
    filter(variable %in% c("D_clamp"), port == port_)
  
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
  
  
  plot_tmp <- ggplot(df_to_plot, aes(x = factor(FF), y = value, fill = factor(days))) +
      geom_bar(position=position_dodge(), stat="identity") +
      geom_errorbar(aes(ymin=value-ci, ymax=value+ci),width=.1, alpha = 0.6, position=position_dodge(.9)) +
      facet_grid(variable ~ ., scales = "free_y") +
      scale_fill_brewer(palette="Paired", name = "Days") + theme_minimal() +
      geom_hline(yintercept = 0, size = 0.2) +
      xlab("Factors included") + ylab("") +
    ggtitle(paste("Individual, ", port_, sep = ""))
  ggsave(filename = paste("estimated_data/disaster_sorts/images/ff_reg_ind_", port_,".pdf",sep = ""), 
         plot = plot_tmp, device = "pdf", height = 7, width = 7)
  # dev.off()
}


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
# ggplot(av_melt, aes(x = factor(days), y = value, fill = variable)) +
#   geom_bar(position=position_dodge(), stat="identity") +
#   facet_grid(char ~ ., scales = "free_y") +
#   scale_fill_brewer(palette="Paired", name = "Portfolio") + theme_minimal()

pdf("estimated_data/disaster_sorts/images/bm_op_port_ind.pdf", height = 5, width = 9)
  ggplot(av_melt %>% mutate(variable = as.numeric(substr(variable, 4,4))), 
         aes(x = variable, y = value)) +
    geom_point() + geom_line() +
    facet_grid(char ~ days, scales = "free_y") +
    theme_minimal() + xlab("Portfolio") + ylab("") +
    ggtitle("Individual")
dev.off()


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

pdf("estimated_data/disaster_sorts/images/bm_op_ts_ind.pdf", height = 5, width = 9)
  rbind(bm_melt, op_melt) %>%
    ggplot(aes(x = date, y=value, color = variable)) +
    geom_line() + facet_grid(char~., scales = "free_y") +
    xlab("") + ylab("") + theme_minimal() +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    ggtitle("Individual")
dev.off()


############################################################
# Looking at industry portfolio composition:
port_const = read_csv("estimated_data/disaster_sorts/port_sort_const_agg.csv")
port_const <- port_const %>% dplyr::rename(D_clamp_30 = "('D_clamp', -99)")

# Calculating the share of firms of each industry for every portfolio
port_ind_agg <- port_const %>%
  group_by(form_date, D_clamp_30) %>%
  mutate(total_cnt = length(permno)) %>%
  ungroup() %>%
  group_by(form_date, D_clamp_30, ff_ind) %>%
  summarize(share = sum(1/total_cnt))

ggplot(port_ind_agg %>% filter(ff_ind == "Money", D_clamp_30 %in% c(1,5), form_date >= "2005-01-01"), 
       aes(x = form_date, y = share, color = factor(D_clamp_30))) +
  geom_line() + geom_point(size = 1) + ggtitle("Share of Finance Companies")

# Industry composition before (Jan-08) and after the crisis (Dec-10):
ggplot(port_ind_agg %>% filter(form_date %in% as.Date(c("2006-12-31", "2010-12-31")), D_clamp_30 %in%c(1,5)), 
       aes(x = form_date, y = share, fill = ff_ind)) +
  geom_bar(stat = "identity") + scale_fill_brewer(palette="Set3") + theme_minimal() +
  facet_grid(.~D_clamp_30) + ggtitle("Share of Number of Companies")

# Calculating the share of market value coming from firms in each 
# for every portfolio
port_ind_agg <- port_const %>%
  group_by(form_date, D_clamp_30) %>%
  mutate(total_mktcap = sum(mktcap)) %>%
  ungroup() %>%
  group_by(form_date, D_clamp_30, ff_ind) %>%
  summarize(share = sum(mktcap/total_mktcap))

ggplot(port_ind_agg %>% filter(ff_ind == "Money", D_clamp_30 %in% c(1,5)),
       aes(x = form_date, y = share, color = factor(D_clamp_30))) +
  geom_line() + geom_point(size = 1) + ggtitle("Share of Market Value")

ggplot(port_ind_agg %>% filter(form_date %in% as.Date(c("2006-12-31", "2010-12-31")), D_clamp_30 %in%c(1,5)),
       aes(x = form_date, y = share, fill = ff_ind)) +
  geom_bar(stat = "identity") + scale_fill_brewer(palette="Set3") + theme_minimal() +
  facet_grid(.~D_clamp_30) + ggtitle("Share of Market Value")


##########

port_ret <- read_csv("estimated_data/disaster_sorts/port_sort_ret_agg.csv") %>%
  dplyr::rename(date = X1)

port_ret %>% 
  filter(variable == "D_clamp", days == -99) %>%
  select(date, ew_1, ew_5) %>% 
  data.table() %>% melt(id.vars = "date") %>%
  group_by(variable) %>%
  mutate(value = cumprod(value+1)) %>% 
  ggplot(aes(x = date, y = value, color = variable)) +
  geom_line()

port_ret %>% 
  filter(variable == "D_clamp", days == 30) %>%
  mutate(ew_diff = ew_1 - ew_5, vw_diff = vw_1 - vw_5) %>%
  select(date, ew_diff) %>% 
  data.table() %>% melt(id.vars = "date") %>%
  group_by(variable) %>%
  mutate(value = cumprod(value+1)) %>% 
  ggplot(aes(x = date, y = value, color = variable)) +
  geom_line()



port_ret %>%
  mutate_at(c("ew_1", "ew_2", "ew_3", "ew_4", "ew_5", "vw_1", "vw_2", "vw_3", "vw_4", "vw_5"), function(x) 12*log(1+x)) %>%
  summarize_all(mean)

port_ret %>%
  filter(variable == "D_clamp", days == -99) %>%
  filter(date <= "2008-01-01") %>%
  summarize_all(mean)

port_ret %>%
  filter(variable == "D_clamp", days == -99) %>%
  summarize_all(mean)



###############
reg_res <- read_csv("estimated_data/disaster_sorts/reg_results_agg_3.csv")

for (port_ in c("ew_diff", "vw_diff")) {
  reg_res_filter <- reg_res %>%
    filter(variable %in% c("D_clamp"), port == port_)
  
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
  
  # pdf(paste("estimated_data/disaster_sorts/images/ff_reg_agg_", port_,".pdf",sep = ""), height = 8, width = 7)
  plot_tmp <- ggplot(df_to_plot, aes(x = factor(FF), y = value, fill = factor(days))) +
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=value-ci, ymax=value+ci),width=.1, alpha = 0.6, position=position_dodge(.9)) +
    facet_grid(variable ~ ., scales = "free_y") +
    scale_fill_brewer(palette="Paired", name = "Days") + theme_minimal() +
    geom_hline(yintercept = 0, size = 0.2) +
    xlab("Factors included") + ylab("") +
    ggtitle(paste("Aggregate, ", port_, sep = ""))
  ggsave(filename = paste("estimated_data/disaster_sorts/images/ff_reg_agg_", port_,"_3.pdf",sep = ""), 
         plot = plot_tmp, device = "pdf", height = 7, width = 7)
  # dev.off()
}


############################################################
# Comparing with Emil's measure
############################################################
emil <- read_csv("estimated_data/disaster_sorts/port_sort_ret_agg_emil.csv") %>%
  dplyr::rename(date = X1)
emil %>%
  filter(variable == "D") %>%
  summarise_all(function(x) ifelse(is.numeric(x), 12*mean(x),NA))


reg_res_emil <- read_csv("estimated_data/disaster_sorts/reg_results_agg_emil.csv")
reg_res_emil %>% filter(variable == "D", port == "ew_diff")
reg_res_emil %>% filter(variable == "D_clamp", days == 180, port == "ew_diff")

reg_res_emil %>% filter(variable == "D", port == "vw_diff")
reg_res_emil %>% filter(variable == "D_clamp", days == 180, port == "vw_diff")

############################################################
# Comparing with SPX disaster measure
############################################################
reg_res <- read_csv("estimated_data/disaster_sorts/reg_results_agg_emil.csv")

switch_port_name <- function(x){
  switch(x, 
         "ew_1" = "ew_5", "ew_2" = "ew_4", "ew_3" = "ew_3", "ew_4" = "ew_2", "ew_5" = "ew_1", "ew_diff" = "ew_diff",
         "vw_1" = "vw_5", "vw_2" = "vw_4", "vw_3" = "vw_3", "vw_4" = "vw_2", "vw_5" = "vw_1", "vw_diff" = "vw_diff")
}

reg_res$port <- as.character(sapply(X = reg_res$port, FUN = switch_port_name))

for (port_ in c("ew_diff", "ew_1","vw_diff","vw_1", "vw_5", "ew_5")) {
  reg_res_filter <- reg_res %>%
    filter(variable %in% c("D"), port == port_)
  
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
  
  # pdf(paste("estimated_data/disaster_sorts/images/ff_reg_agg_", port_,".pdf",sep = ""), height = 8, width = 7)
  plot_tmp <- ggplot(df_to_plot, aes(x = factor(FF), y = value, fill = factor(days))) +
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=value-ci, ymax=value+ci),width=.1, alpha = 0.6, position=position_dodge(.9)) +
    facet_grid(variable ~ ., scales = "free_y") +
    scale_fill_brewer(palette="Paired", name = "Days") + theme_minimal() +
    geom_hline(yintercept = 0, size = 0.2) +
    xlab("Factors included") + ylab("") +
    ggtitle(paste("Emil's disaster measure, ", port_, sep = ""))
  ggsave(filename = paste("estimated_data/disaster_sorts/images/ff_emils_disaster_", port_,".pdf",sep = ""), 
         plot = plot_tmp, device = "pdf", height = 7, width = 7)
  # dev.off()
}


############################################################
# Full spectrum of disaster measures:
############################################################
reg_res <- read_csv("estimated_data/disaster_sorts/reg_results.csv")

# port_list <- c("ew_1", "ew_2", "ew_3", "ew_4", "ew_5", "vw_diff")
port_list <- c("vw_1", "vw_2", "vw_3", "vw_4", "vw_5", "vw_diff")
reg_res %>%
  filter(port %in% port_list, 
         variable == "D_clamp", FF %in% c(0, 1,5), maturity == "level") %>%
  mutate(alpha_ci = alpha_se*2) %>%
  ggplot(aes(x = port, y = alpha, fill = level)) +
  geom_bar(position=position_dodge(), stat="identity") +
  geom_errorbar(aes(ymin = alpha-alpha_ci, ymax = alpha+alpha_ci), width=.2, alpha = 0.6, position=position_dodge(.9)) +
  facet_grid(FF ~.)
  
# port_list <- c("ew_1", "ew_2", "ew_3", "ew_4", "ew_5")
port_list <- c("vw_1", "vw_2", "vw_3", "vw_4", "vw_5")
reg_res %>%
  filter(port %in% port_list, 
         variable == "D_clamp", FF %in% c(0,1,5), maturity == "level") %>%
  mutate(alpha_ci = alpha_se*2) %>%
  mutate(port = as.numeric(substr(port,4,4))) %>%
  ggplot(aes(x = port, y = alpha, color = level)) +
  geom_line() + geom_point() +
  facet_grid(FF~., scales = "free") +
  theme_minimal()


reg_res %>% 
  filter(port == "ew_diff", FF == 0) %>%
  select(level, variable, maturity,alpha, alpha_se) %>% View()

reg_res %>% 
  filter(port == "ew_diff", FF == 1) %>%
  select(level, variable, maturity,alpha, alpha_se) %>% View()

df_to_plot <- reg_res %>%
  filter(port == "ew_diff", maturity == "level", variable == "D_clamp") %>%
  select(level, variable,FF, maturity, alpha, alpha_se) %>%
  mutate(alpha_ci = alpha_se*2)

ggplot(df_to_plot, aes(x = factor(FF), y = alpha, fill = level)) +
  geom_bar(position=position_dodge(), stat="identity") +
  geom_errorbar(aes(ymin = alpha-alpha_ci, ymax = alpha+alpha_ci), width=.2, alpha = 0.6, position=position_dodge(.9))








