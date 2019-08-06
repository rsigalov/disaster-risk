library("ggplot2")
library("data.table")
library("dplyr")
library("reshape")
library("readr")
library("corrplot")
setwd("~/Documents/PhD/disaster-risk-revision/")

# Looking at one level factor regressions:
df_reg <- read_csv("estimated_data/term_structure/reg_results_pc_unbalance.csv")

pdf("estimated_data/term_structure/dist_r2_unbalanced.pdf", width = 6.5, height = 3.5)
  df_to_plot <- df_reg %>% filter(N >= 12*10, m %in% c(30, 180))
  df_to_plot$m <- factor(df_to_plot$m, levels = c(30, 60, 90, 120, 150, 180),
                         labels = c("30 days", "60 days", "90 days", "120 days", "150 days", "180 days"))
  
  df_to_plot %>%
    ggplot(aes(x = R2, fill = factor(pc_num))) +
    geom_histogram(alpha = 0.5, position = "identity", bins = 10) +
    facet_wrap(~m, scales = "free") +
    xlab("") + ylab("") +
    theme_minimal() +
    theme(legend.position = "bottom") +
    labs(fill = "PCs included")
dev.off()

####################################################
# Comparing between PCs among each other 

# Loading monthly PCs:
pc_file_list <- c("pc_unbalanced", "pc_balanced", "pc_sp_mon")
pc_type_list <- c("Ind(U)", "Ind(B)", "SPX")

for (i in 1:length(pc_file_list)) {
  if (i == 1) {
    df_pc <- read_csv(paste("estimated_data/term_structure/", pc_file_list[i], ".csv", sep = "")) %>%
      mutate(type = pc_type_list[i])
  } else {
    to_append <- read_csv(paste("estimated_data/term_structure/", pc_file_list[i], ".csv", sep = "")) %>%
      mutate(type = pc_type_list[i])
    df_pc <- rbind(df_pc, to_append)
  }
}

# Loading daily PCs:
pc_file_list <- c("pc_sp_daily", "pc_unbalanced_daily")
pc_type_list <- c("SPX", "Ind(U)")

for (i in 1:length(pc_file_list)) {
  if (i == 1) {
    df_pc_daily <- read_csv(paste("estimated_data/term_structure/", pc_file_list[i], ".csv", sep = "")) %>%
      mutate(type = pc_type_list[i])
  } else {
    to_append <- read_csv(paste("estimated_data/term_structure/", pc_file_list[i], ".csv", sep = "")) %>%
      mutate(type = pc_type_list[i])
    df_pc_daily <- rbind(df_pc_daily, to_append)
  }
}

# Normalizing variables:
norm_func <- function(x) x/sd(x, na.rm = T)
df_pc <- df_pc %>%
  group_by(type) %>%
  mutate_at(c("PC1", "PC2", "PC3"), norm_func)

df_pc_daily <- df_pc_daily %>% 
  mutate(PC2 = replace(PC2, PC2 >= 10000, NA)) %>%
  as.data.frame()

df_pc_daily <- df_pc_daily %>%
  group_by(type) %>%
  mutate_at(c("PC1", "PC2", "PC3"), norm_func)

# Plotting variables:
df_pc %>%
  data.table() %>%
  melt(id.vars = c("date_mon", "type")) %>%
  ggplot(aes(x = date_mon,y=value, color = variable)) +
    geom_line() + 
    facet_wrap(~type)

df_pc_daily %>%
  data.table() %>%
  melt(id.vars = c("date", "type")) %>%
  ggplot(aes(x = date,y=value, color = variable)) +
  geom_line() + 
  facet_wrap(~type)

pdf("estimated_data/term_structure/pc_mon_comparison.pdf", width = 5, height = 6.5)
  df_pc %>%
    data.table() %>%
    melt(id.vars = c("date_mon", "type")) %>%
    ggplot(aes(x = date_mon, y = value, color = type)) +
    geom_line(alpha = 0.9, size = 0.5) + 
    facet_grid(variable~., scales = "free_y") +
    labs(color = "") + xlab("") + ylab("") +
    theme_minimal() +
    theme(legend.position = "bottom")
dev.off()
  
pdf("estimated_data/term_structure/pc_daily_comparison.pdf", width = 5, height = 6.5)
  df_pc_daily %>%
    data.table() %>%
    melt(id.vars = c("date", "type")) %>%
    filter(abs(value) <= 25) %>%
    ggplot(aes(x = date, y = value, color = type)) +
    geom_line(alpha = 0.8) + 
    facet_grid(variable~., scales = "free_y") +
    labs(color = "") + xlab("") + ylab("") +
    theme_minimal() +
    theme(legend.position = "bottom")
dev.off()

# Calculating correlation in levels:
pdf("estimated_data/term_structure/pc_corr_level_mon.pdf", width = 5.5, height = 5.5)
  df_pc %>%
    data.table() %>%
    melt(id.vars = c("date_mon", "type")) %>%
    cast(date_mon ~ type + variable) %>%
    cor(use = "pairwise.complete.obs") %>%
    corrplot(method="number", tl.col = "black", cl.pos = "n")
dev.off()

pdf("estimated_data/term_structure/pc_corr_level_daily.pdf", width = 4, height = 4)
  df_pc_daily %>%
    data.table() %>%
    melt(id.vars = c("date", "type")) %>%
    cast(date ~ type + variable) %>%
    cor(use = "pairwise.complete.obs") %>%
    corrplot(method="number", tl.col = "black", cl.pos = "n")
dev.off()

# Calculating correlations in differences:
pdf("estimated_data/term_structure/pc_corr_diff_mon.pdf", width = 5.5, height = 5.5)
  df_pc %>%
    data.table() %>%
    melt(id.vars = c("date_mon", "type")) %>%
    group_by(type, variable) %>%
    mutate(value_diff = value - lag(value)) %>%
    select(-c("value")) %>%
    cast(date_mon ~ type + variable) %>%
    cor(use = "pairwise.complete.obs") %>%
    corrplot(method="number", tl.col = "black", cl.pos = "n")
dev.off()

pdf("estimated_data/term_structure/pc_corr_diff_daily.pdf", width = 4, height = 4)
  df_pc_daily %>%
    data.table() %>%
    melt(id.vars = c("date", "type")) %>%
    group_by(type, variable) %>%
    mutate(value_diff = value - lag(value)) %>%
    select(-c("value")) %>%
    cast(date ~ type + variable) %>%
    cor(use = "pairwise.complete.obs") %>%
    corrplot(method="number", tl.col = "black", cl.pos = "n")
dev.off()

####################################################
# Comparing daily "PCs" 

