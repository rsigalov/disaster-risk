library("ggplot2")
library("data.table")
library("dplyr")
library("reshape")
setwd("~/Documents/PhD/disaster-risk-revision/")

# Loading data on disaster series with and without extrapolation:
df_extr <- data.table(read.csv("estimated_data/disaster-risk-series/combined_no_extr.csv"))
df_extr$type <- "new"

df_no_extr <- data.table(read.csv("estimated_data/disaster-risk-series/combined_disaster_df.csv"))
df_no_extr$type <- "old"

df <- rbind(df_extr, df_no_extr)
df$date <- as.Date(df$date, format = "%Y-%m-%d")

# 1. Comparing D_clamp for all days between extrapolation and no extrapolation
pdf(file = "images/compare_disaster_series_new/compare_old_new_1.pdf", width = 10, height = 3)
  df_to_plot <- 
    df[(var == "D_clamp") & (level == "ind") & (agg_type == "mean_all")]

  ggplot(df_to_plot, aes(x = date, y = value, color = type)) +
    geom_line(alpha = 0.8) +
    facet_grid(var~days)
dev.off()

# 2. Comparing rn_prob between extrapolation and no extrapolation
pdf(file = "images/compare_disaster_series_new/compare_old_new_2.pdf", width = 10, height = 6)
  df_to_plot <- 
    df[(var %in% c("rn_prob_20mon", "rn_prob_40mon")) & (level == "ind") & (agg_type == "mean_all")]
  
  ggplot(df_to_plot, aes(x = date, y = value, color = type)) +
    geom_line(alpha = 0.8) +
    facet_grid(var~days, scales = "free")
dev.off()

# 3. Comparing (new) series without extrapolation among each other (plotting different
# types on different graphs and combining the days on the same graph)
pdf(file = "images/compare_disaster_series_new/compare_new_term_structure.pdf", width = 10, height = 6)
  df_to_plot <-
    df[(type == "new") & (level == "ind") & (agg_type == "mean_all") &
         (var %in% c("D_clamp", "rn_prob_20", "rn_prob_40", "rn_prob_60", "rn_prob_80",
                     "rn_prob_20mon", "rn_prob_40mon", "rn_prob_sigma", "rn_prob_2sigma"))]
  
  ggplot(df_to_plot, aes(x = date, y = value, color = factor(days))) +
    geom_line(alpha = 0.8) +
    facet_wrap(~var, scales = "free")
dev.off()

# 4. Showing different types of series on the same graph:
pdf(file = "images/compare_disaster_series_new/compare_new_cross_section.pdf", width = 10, height = 6)
  df_to_plot <-
    df[(type == "new") & (level == "ind") & (agg_type == "mean_all") &
         (var %in% c("D_clamp", "rn_prob_20", "rn_prob_80", "rn_prob_20mon", "rn_prob_2sigma"))]
  
  # Dividing each series by standard deviation to make them comparable:
  df_sum <- df_to_plot[, sd := sd(value), by = c("days", "var")]
  df_sum$value <- df_sum$value/df_sum$sd
  
  ggplot(df_sum, aes(x = date, y = value, color = var)) +
    geom_line(alpha = 0.8) +
    facet_wrap(~days, scales = "free")
dev.off()

# 5. Comparing aggregate measures, doint the term structure
df <- as_tibble(read.csv("estimated_data/disaster-risk-series/combined_disaster_df.csv"))
df$date <- as.Date(df$date, format = "%Y-%m-%d")

df_to_plot <- df %>%
  filter(level == "sp_500") %>%
  filter(var %in% c("D_clamp", "rn_prob_sigma", "rn_prob_2sigma", "rn_prob_5", "rn_prob_20")) %>%
  filter(days %in% c(40, 100, 180))

pdf(file = "images/compare_disaster_series_new/compare_agg_term_structure.pdf", width = 10, height = 6)
  ggplot(df_to_plot, aes(x = date, y = value, color = factor(days))) +
    geom_line(alpha = 0.8) +
    facet_wrap(~var, scales = "free")
dev.off()


# 6. Comparing aggregate measures, cross section:
df_to_plot <- df %>%
  filter(level == "sp_500") %>%
  filter(var %in% c("D_clamp", "rn_prob_sigma", "rn_prob_2sigma", "rn_prob_5", "rn_prob_20")) %>%
  filter(days %in% c(40, 100, 180)) %>%
  group_by(var, days) %>%
  mutate(std = sd(value, na.rm = T)) %>%
  mutate(value_norm = value/std)

pdf(file = "images/compare_disaster_series_new/compare_agg_cross_section.pdf", width = 10, height = 3)
  ggplot(df_to_plot, aes(x = date, y = value_norm, color = factor(var))) +
    geom_line(alpha = 0.8) +
    facet_wrap(~days, scales = "free")
dev.off()

# Additional plots:
df_to_plot <- df %>%
  filter(((level == "ind") & (days == 120)) | ((level == "sp_500_OM") & (days == 100))) %>%
  filter(var == "D_clamp")

ggplot(df_to_plot, aes(x = date, y = value, color = factor(level))) +
  geom_line(alpha = 0.8)

cor(cast(df_to_plot, date ~ level + days), use = "complete.obs")
diff(cast(df_to_plot, date ~ level + days) %>% select(sp_500_CME_40, sp_500_CME_100))


diff(cast(df_to_plot, date ~ level + days)$sp_500_CME_40)

