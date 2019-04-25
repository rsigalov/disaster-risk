library('ggplot2')
library('data.table')

setwd("~/Documents/PhD/disaster-risk-revision/")


df <- data.table(read.csv("simulations_results_2.csv"))

################################################
# Calculating theoretical value of D:
# 1. Values of lambdas from the data frame:
lambda_disaster_base <- 0.05
lambda_disaster_list <- unique(df[var_to_vary == "Disaster Intensity"]$lambda_level)*lambda_disaster_base

lambda_small_base <- 20
lambda_small_list <- unique(df[var_to_vary == "Small Intensity"]$lambda_level)*lambda_small_base

# 2. Other values from the simulation:
mu_small <- -0.005
sigma_small <- 0.05
mu_disaster <- -0.8
sigma_disaster <- 0.1

# 3. Calculating value of D for varying disaster intensity:
D <- function(lambda, mu, sigma) {
  2 * (1 + mu + 0.5 * (mu^2 + sigma^2) - exp(mu+0.5*sigma^2)) * lambda
}

D_vary_small_small <- sapply(lambda_small_list, FUN = function(x) D(x, mu_small, sigma_small))
D_vary_small_disaster <- D(lambda_disaster_base, mu_disaster, sigma_disaster)
D_vary_small <- D_vary_small_small + D_vary_small_disaster

D_vary_disaster_small <- D(lambda_small_base, mu_small, sigma_small)
D_vary_disaster_disaster <- sapply(lambda_disaster_list, function(x) D(x, mu_disaster, sigma_disaster))
D_vary_disaster <- D_vary_disaster_small + D_vary_disaster_disaster

# 4. Creating extra data set with theoretical values
measure_list <- unique(df$measure)
min_strike_list <- unique(df$min_strike)
df_theory <- data.frame(measure = c(NA), min_strike = c(NA), var_to_vary = c(NA),
                        lambda_level = c(NA), D = c(NA))
for (measure in measure_list) {
  for (min_strike in min_strike_list) {
    df_theory_to_append <- data.frame(measure = rep(measure, 10), 
                                      min_strike = rep(min_strike, 10), 
                                      var_to_vary = rep("Small Intensity (Theoretical)", 10),
                                      lambda_level = lambda_small_list/lambda_small_base, 
                                      D = D_vary_small)
    df_theory <- rbind(df_theory, df_theory_to_append)
    
    df_theory_to_append <- data.frame(measure = rep(measure, 10), 
                                      min_strike = rep(min_strike, 10), 
                                      var_to_vary = rep("Disaster Intensity (Theoretical)", 10),
                                      lambda_level = lambda_disaster_list/lambda_disaster_base, 
                                      D = D_vary_disaster)
    df_theory <- rbind(df_theory, df_theory_to_append)
  }
}
df_theory <- data.table(df_theory)
df_theory <- df_theory[2:.N]

df <- rbind(df, df_theory)



pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_25.pdf", width = 10, height = 6)
  ggplot(df[min_strike == 0.25], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
    geom_point(alpha = 0.8) + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_wrap(~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") +
    ggtitle("Minimum strike at 0.25*spot") + theme_minimal()
dev.off()

pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_40.pdf", width = 10, height = 6)
  ggplot(df[min_strike == 0.4], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
    geom_point() + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_wrap(~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") +
    ggtitle("Minimum strike at 0.40*spot") + theme_minimal()
dev.off()

pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_60.pdf", width = 10, height = 6)
  ggplot(df[min_strike == 0.6], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
    geom_point() + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_wrap(~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") +
    ggtitle("Minimum strike at 0.60*spot") + theme_minimal()
dev.off()

pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_80.pdf", width = 10, height = 6)
  ggplot(df[min_strike == 0.8], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
    geom_point() + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_wrap(~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") +
    ggtitle("Minimum strike at 0.80*spot") + theme_minimal()
dev.off()
  


############################################
# Repeating the same for different parameters:
df2 <- data.table(read.csv("simulations_results_alt.csv"))

lambda_small_base <- 1.57
lambda_small_list <- unique(df2[var_to_vary == "Small Intensity"]$lambda_level)*lambda_small_base

# 2. Other values from the simulation:
mu_small <- -0.05
sigma_small <- 0.05

D_vary_small_small <- sapply(lambda_small_list, FUN = function(x) D(x, mu_small, sigma_small))
D_vary_small_disaster <- D(lambda_disaster_base, mu_disaster, sigma_disaster)
D_vary_small <- D_vary_small_small + D_vary_small_disaster

D_vary_disaster_small <- D(lambda_small_base, mu_small, sigma_small)
D_vary_disaster_disaster <- sapply(lambda_disaster_list, function(x) D(x, mu_disaster, sigma_disaster))
D_vary_disaster <- D_vary_disaster_small + D_vary_disaster_disaster

# 4. Creating extra data set with theoretical values
measure_list <- unique(df$measure)
min_strike_list <- unique(df$min_strike)
df_theory <- data.frame(measure = c(NA), min_strike = c(NA), var_to_vary = c(NA),
                        lambda_level = c(NA), D = c(NA))
for (measure in measure_list) {
  for (min_strike in min_strike_list) {
    df_theory_to_append <- data.frame(measure = rep(measure, 10), 
                                      min_strike = rep(min_strike, 10), 
                                      var_to_vary = rep("Small Intensity (Theoretical)", 10),
                                      lambda_level = lambda_small_list/lambda_small_base, 
                                      D = D_vary_small)
    df_theory <- rbind(df_theory, df_theory_to_append)
    
    df_theory_to_append <- data.frame(measure = rep(measure, 10), 
                                      min_strike = rep(min_strike, 10), 
                                      var_to_vary = rep("Disaster Intensity (Theoretical)", 10),
                                      lambda_level = lambda_disaster_list/lambda_disaster_base, 
                                      D = D_vary_disaster)
    df_theory <- rbind(df_theory, df_theory_to_append)
  }
}
df_theory <- data.table(df_theory)
df_theory <- df_theory[2:.N]

df2 <- rbind(df2, df_theory)

pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_25_alt.pdf", width = 10, height = 6)
ggplot(df2[min_strike == 0.25], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
  geom_point() + geom_smooth(se = F, method = "lm", size = 0.25) +
  facet_wrap(~measure, scales = "free") +
  xlab("Intensity Relative to Baseline Level") +
  ggtitle("Minimum strike at 0.25*spot") + theme_minimal()
dev.off()

pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_40_alt.pdf", width = 10, height = 6)
ggplot(df2[min_strike == 0.4], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
  geom_point() + geom_smooth(se = F, method = "lm", size = 0.25) +
  facet_wrap(~measure, scales = "free") +
  xlab("Intensity Relative to Baseline Level") +
  ggtitle("Minimum strike at 0.40*spot") + theme_minimal()
dev.off()

pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_60_alt.pdf", width = 10, height = 6)
ggplot(df2[min_strike == 0.6], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
  geom_point() + geom_smooth(se = F, method = "lm", size = 0.25) +
  facet_wrap(~measure, scales = "free") +
  xlab("Intensity Relative to Baseline Level") +
  ggtitle("Minimum strike at 0.60*spot") + theme_minimal()
dev.off()

pdf(file = "write-up-files/Simulations/images/D_comp_min_strike_80_alt.pdf", width = 10, height = 6)
ggplot(df2[min_strike == 0.8], aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
  geom_point() + geom_smooth(se = F, method = "lm", size = 0.25) +
  facet_wrap(~measure, scales = "free") +
  xlab("Intensity Relative to Baseline Level") +
  ggtitle("Minimum strike at 0.80*spot") + theme_minimal()
dev.off()



################################################
# Doing graphs for probability:

df3 <- data.table(read.csv("simulation_results_prob_3.csv"))

pdf(file = "write-up-files/Simulations/images/rn_probability_of_60_perc_drop.pdf", width = 6, height = 4)
ggplot(df3[min_strike %in% c(0.25, 0.8)], aes(x = lambda_level, y = prob, color = var_to_vary, shape = var_to_vary)) +
  geom_point(size = 2.5) + geom_smooth(se = F, method = "lm", size = 0.25) +
  facet_grid(min_strike~measure, scales = "free") +
  xlab("Intensity Relative to Baseline Level") +
  theme_minimal() + ylab("Risk-Neutral Probability of 60% Drop")
dev.off()

################################################
# Plotting Distribution for PCA: correlation and explained variance
df_sim <- data.table(read.csv("simulations_pca.csv"))

pdf(file = "write-up-files/Simulations/images/pca_simulation_corr.pdf", width = 8, height = 5)
  ggplot(df_sim, aes(x = corr, fill = factor(var_e_i))) +
    geom_histogram(alpha = 0.75, position="identity") +
    facet_grid(rho_i ~ rho_d) +
    scale_fill_discrete(name = "std(e)") + xlab("Correlation") + ylab("")
dev.off()

pdf(file = "write-up-files/Simulations/images/pca_simulation_exp_var.pdf", width = 8, height = 5)
  ggplot(df_sim, aes(x = exp_var, fill = factor(var_e_i))) +
    geom_histogram(alpha = 0.75, position="identity") +
    facet_grid(rho_i ~ rho_d) +
    scale_fill_discrete(name = "std(e)") + xlab("Explained Variance") + ylab("")
dev.off()

################################################
# Plotting individual series for D and PC1 for each parameter set
df_sim <- data.table(read.csv("simulations_pca_ind.csv"))
df_sim <- melt(df_sim,id.vars = c("rho_d", "rho_i", "var_e_i", "obs"))

pdf(file = "write-up-files/Simulations/images/pca_and_d_comp.pdf", width = 8, height = 5)
ggplot(df_sim[var_e_i == 2], aes(x = obs, y = value, color = variable)) +
  geom_line(alpha = 0.75) +
  facet_grid(rho_i ~ rho_d) + xlab("time period") +
  ylab("")
dev.off()

################################################
# Plotting summary statistics of PCA simualtions
# using the correct calculation of PCA
df_sim <- data.table(read.csv("simulation_pca_summary_stats.csv"))
df_sim2 <- data.table(read.csv("simulation_pca_summary_stats_2.csv"))
df_sim <- rbind(df_sim, df_sim2)

df_sim[est == "MeanD"]$est <- "meanD"

pdf("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/pca_sims_corr_sum_stats_1.pdf", width = 8, height = 6)
ggplot(df_sim[rho_i == 0.05], aes(x = num_firms, y = mean_est, ymin = quant_5, ymax = quant_95, color = factor(est))) +
  geom_hline(aes(yintercept=0.8), alpha = 0.5) + geom_point() + geom_errorbar() + 
  facet_grid(std_e_i ~ rho_d) + theme(legend.position = "bottom") + scale_color_discrete(name = "Estimator") +
  ylab("Correlation Between Estimator and Factor") + xlab("Number of Firms")
dev.off()

pdf("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/pca_sims_corr_sum_stats_2.pdf", width = 8, height = 6)
ggplot(df_sim[rho_i == 0.5], aes(x = num_firms, y = mean_est, ymin = quant_5, ymax = quant_95, color = factor(est))) +
  geom_hline(aes(yintercept=0.8), alpha = 0.5) + geom_point() + geom_errorbar() +
  facet_grid(std_e_i ~ rho_d) + theme(legend.position = "bottom") + scale_color_discrete(name = "Estimator") +
  ylab("Correlation Between Estimator and Factor") + xlab("Number of Firms")
dev.off()

pdf("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/pca_sims_corr_sum_stats_3.pdf", width = 8, height = 6)
ggplot(df_sim[rho_i == 0.9], aes(x = num_firms, y = mean_est, ymin = quant_5, ymax = quant_95, color = factor(est))) +
  geom_hline(aes(yintercept=0.8), alpha = 0.5) + geom_point() + geom_errorbar() +
  facet_grid(std_e_i ~ rho_d) + theme(legend.position = "bottom") + scale_color_discrete(name = "Estimator") +
  ylab("Correlation Between Estimator and Factor") + xlab("Number of Firms")
dev.off()

# Doing one table with very persistent disaster process (rho = 0.9):
pdf("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/pca_sims_corr_sum_stats_short.pdf", width = 8, height = 6)
  ggplot(df_sim[(rho_d == 0.9) & (rho_i)], 
         aes(x = num_firms, y = mean_est, ymin = quant_5, ymax = quant_95, color = factor(est))) +
    geom_hline(aes(yintercept=0.8), alpha = 0.5) + geom_point() + geom_errorbar() +
    facet_grid(std_e_i ~ rho_i, labeller = label_both) + 
    theme(legend.position = "bottom") + scale_color_discrete(name = "Estimator") +
    ylab("Correlation Between Estimator and Factor") + xlab("Number of Firms")
dev.off()

############################################################
# Comparing different cases of small jump distribution:
df_prob_case_1 <- data.table(read.csv("simulation_results_prob_case_1.csv"))
df_prob_case_2 <- data.table(read.csv("simulation_results_prob_case_2.csv"))
df_prob_case_3 <- data.table(read.csv("simulation_results_prob_case_3.csv"))

pdf(file = "/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/rn_prob_case_1.pdf", width = 6, height = 4)
  ggplot(df_prob_case_1[min_strike %in% c(0.25, 0.8)], aes(x = lambda_level, y = prob, color = var_to_vary, shape = var_to_vary)) +
    geom_point(size = 2.5) + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_grid(min_strike~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") +
    theme_minimal() + ylab("Risk-Neutral Probability of 50% Drop")
dev.off()

pdf(file = "/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/rn_prob_case_2.pdf", width = 6, height = 4)
  ggplot(df_prob_case_2[min_strike %in% c(0.25, 0.8)], aes(x = lambda_level, y = prob, color = var_to_vary, shape = var_to_vary)) +
    geom_point(size = 2.5) + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_grid(min_strike~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") +
    theme_minimal() + ylab("Risk-Neutral Probability of 50% Drop")
dev.off()

pdf(file = "/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/rn_prob_case_3.pdf", width = 6, height = 4)
  ggplot(df_prob_case_3[min_strike %in% c(0.25, 0.8)], aes(x = lambda_level, y = prob, color = var_to_vary, shape = var_to_vary)) +
    geom_point(size = 2.5) + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_grid(min_strike~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") +
    theme_minimal() + ylab("Risk-Neutral Probability of 50% Drop")
dev.off()

############################################################
# Comparing different cases of small jump distribution for
# comparison of D estimates to the data:

# Case I: Small Jump distribution log(r) ~ N(-0.05, 0.05)
#         Disaster Jump Distribution log(r) ~ N(-0.8, 0.1)
# Case I: Small Jump distribution log(r) ~ N(-0.05, 0.05)
#         Disaster Jump Distribution log(r) ~ N(-0.8, 0.1)
# Case I: Small Jump distribution log(r) ~ N(-0.05, 0.05)
#         Disaster Jump Distribution log(r) ~ N(-0.8, 0.1)
df <- data.table(read.csv("simulations_results_case_3.csv"))

# 1. Values of lambdas from the data frame:
lambda_disaster_base <- 0.05
lambda_disaster_list <- unique(df[var_to_vary == "Disaster Intensity"]$lambda_level) * lambda_disaster_base

lambda_small_base <- 20
lambda_small_list <- unique(df[var_to_vary == "Small Intensity"]$lambda_level) * lambda_small_base

# 2. Other values from the simulation:
mu_small <- -0.005
sigma_small <- 0.01
mu_disaster <- -0.8
sigma_disaster <- 0.1

# 3. Calculating value of D for varying disaster intensity:
D <- function(lambda, mu, sigma) {
  2 * (1 + mu + 0.5 * (mu^2 + sigma^2) - exp(mu+0.5*sigma^2)) * lambda
}

D_vary_small_small <- sapply(lambda_small_list, FUN = function(x) D(x, mu_small, sigma_small))
D_vary_small_disaster <- D(lambda_disaster_base, mu_disaster, sigma_disaster)
D_vary_small <- D_vary_small_small + D_vary_small_disaster

D_vary_disaster_small <- D(lambda_small_base, mu_small, sigma_small)
D_vary_disaster_disaster <- sapply(lambda_disaster_list, function(x) D(x, mu_disaster, sigma_disaster))
D_vary_disaster <- D_vary_disaster_small + D_vary_disaster_disaster

# 4. Creating extra data set with theoretical values
measure_list <- unique(df$measure)
min_strike_list <- unique(df$min_strike)
df_theory <- data.frame(measure = c(NA), min_strike = c(NA), var_to_vary = c(NA),
                        lambda_level = c(NA), D = c(NA))
for (measure in measure_list) {
  for (min_strike in min_strike_list) {
    df_theory_to_append <- data.frame(measure = rep(measure, 10), 
                                      min_strike = rep(min_strike, 10), 
                                      var_to_vary = rep("Small Intensity (Theoretical)", 10),
                                      lambda_level = lambda_small_list/lambda_small_base, 
                                      D = D_vary_small)
    df_theory <- rbind(df_theory, df_theory_to_append)
    
    df_theory_to_append <- data.frame(measure = rep(measure, 10), 
                                      min_strike = rep(min_strike, 10), 
                                      var_to_vary = rep("Disaster Intensity (Theoretical)", 10),
                                      lambda_level = lambda_disaster_list/lambda_disaster_base, 
                                      D = D_vary_disaster)
    df_theory <- rbind(df_theory, df_theory_to_append)
  }
}
df_theory <- data.table(df_theory)
df_theory <- df_theory[2:.N]

df <- rbind(df, df_theory)

pdf("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/D_comp_short_case_3.pdf", width = 7, height = 4)
  ggplot(df[(min_strike %in% c(0.25, 0.8)) & (measure %in% c("D", "D clamp", "D in sample"))], 
         aes(x = lambda_level, y = D, color = var_to_vary, shape = var_to_vary)) +
    geom_point(alpha = 0.8) + geom_smooth(se = F, method = "lm", size = 0.25) +
    facet_grid(min_strike~measure, scales = "free") +
    xlab("Intensity Relative to Baseline Level") + theme_minimal()
dev.off()







