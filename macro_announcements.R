library("dplyr")
library("ggplot2")
library("readr")
library("lfe")
library("stargazer")


macro_react <- read_csv("data/macro_announcements/macro_ann_reaction.csv")
macro_react <- macro_react %>%
    group_by(ann_type, window) %>%
    mutate(surprise = surprise/sd(surprise, na.rm=T))
macro_react <- macro_react %>%
    mutate(diff_ind = after - before, diff_spx = after_spx - before_spx)

# Loading aggregated disaster measure to normalize the reaction
disaster <- read_csv("data/disaster_risk_measures/disaster_risk_measures.csv")
sdD_ind <- sd(log(filter(disaster, level=="Ind", maturity=="level", variable=="D_clamp",agg_freq=="date_mon")$value))
sdD_spx <- sd(log(filter(disaster, level=="SPX", maturity=="level", variable=="D_clamp",agg_freq=="date_mon")$value))

df_list <- list(); i <- 1
for (ann_type_ in c("nonfarm", "pmi", "cpi", "fomc")) {
    df_to_reg <- macro_react %>%
        filter(ann_type == ann_type_, window == 1) %>%
        mutate(
            log_diff_ind = (log(after) - log(before))/sdD_ind, 
            log_diff_spx = (log(after_spx) - log(before_spx))/sdD_spx) %>%
        filter(log_diff_ind < 0.5, log_diff_spx < 0.5) 
    sum_ind <- df_to_reg %>%
        felm(data = ., formula = log_diff_ind ~ surprise) %>%
        summary()
    sum_spx <- df_to_reg %>%
        felm(data = ., formula = log_diff_spx ~ surprise) %>%
        summary()

    df_to_append <- data.frame(sum_ind$coefficients)
    df_to_append$ann_type <- ann_type_
    df_to_append$var <- rownames(df_to_append)
    df_to_append$type <- "Ind"
    df_list[[i]] <- df_to_append; i <- i + 1

    df_to_append <- data.frame(sum_spx$coefficients)
    df_to_append$ann_type <- ann_type_
    df_to_append$var <- rownames(df_to_append)
    df_to_append$type <- "SPX"
    df_list[[i]] <- df_to_append; i <- i + 1
}

coefs <- bind_rows(df_list) %>% tibble()

coefs %>%
    mutate(low_conf = Estimate - 1.96*Std..Error, upp_conf = Estimate + 1.96*Std..Error) %>%
    ggplot(aes(x = ann_type, fill = type, y = Estimate, ymin = low_conf, ymax = upp_conf)) +
    geom_bar(stat = "identity", position="dodge") + geom_errorbar(position=position_dodge(.9), width=.2)+
    geom_hline(yintercept = 0.0, linetype="dashed") + theme_minimal() + 
    facet_grid(var ~ ., scales="free_y")



macro_react %>%
    filter(ann_type == "nonfarm", window == 1) %>%
    mutate(log_diff_spx = (log(after) - log(before))/sdD_ind) %>%
    filter(log_diff_spx < 0.5) %>%
    ggplot(aes(x = surprise, y = log_diff_spx)) + geom_point() + geom_smooth(method="lm")

macro_react %>%
    mutate(
        log_diff_ind = (log(after) - log(before))/sdD_ind, 
        log_diff_spx = (log(after_spx) - log(before_spx))/sdD_spx) %>%
    filter(window == 7, abs(log_diff_ind) < 0.5, abs(log_diff_spx) < 0.5) %>%
    ggplot(aes(x = log_diff_ind, y = log_diff_spx, color=ann_type)) + geom_point() +
    geom_smooth(method="lm")
