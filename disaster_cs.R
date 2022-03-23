library("dplyr")
library("ggplot2")
library("readr")
library("lfe")
library("stargazer")

variable <- "D_clamp"
# variable <- "IV_clamp"
# variable <- "V_clamp"

variable_list <- c("D_clamp", "IV_clamp", "V_clamp")

################################################################
# Loading datasets
# 1. monthly average of individual disaster measures
df_list <- list(); i <- 1
for (variable in variable_list) {
    df_to_append <- read_csv(paste("data/sorting_variables/monthly_average_", variable,".csv", sep=""))
    df_to_append$variable <- variable
    df_list[[i]] <- df_to_append
    i <- i + 1
}
ind_disaster <- bind_rows(df_list)

# 2. aggregate disaster measure (for comparison)
disaster <- read_csv("data/disaster_risk_measures/disaster_risk_measures.csv")

# 3. balance sheet characteristics
crsp_ccm <- read_csv("data/crsp_ccm_3.csv") %>% select(-`...1`, -`Unnamed: 0`)
crsp_ccm <- crsp_ccm %>% select(
    permno, date = date_eom, hsiccd, mktcap = permco_mktcap, bm, op, at_growth) %>%
    mutate(log_mktcap = log(mktcap))

################################################################
# Merging ind averages to balance sheet info
ind_disaster <- ind_disaster %>% left_join(crsp_ccm, on = c("permno", "date"))

################################################################
# Plotting the distribution of each measure
ind_disaster %>%
    filter(value > 0.0001, value < 2.0) %>%
    ggplot(aes(x = value, fill = variable)) + 
    geom_histogram() + scale_x_log10() + facet_grid(variable ~ .)

################################################################
# Estimating cross-sectional regressions. Where the LHS is a 
# normalized log measure (D, V or IV)
reg1 <- ind_disaster %>%
    filter(variable == "D_clamp", value > 0.0) %>%
    mutate(value = value/sd(value)) %>%
    group_by(date, hsiccd) %>% mutate(date_X_industry = cur_group_id()) %>% ungroup() %>%
    felm(data = ., formula = I(log(value)) ~ log_mktcap + bm + op + at_growth|date_X_industry|0|date)

reg2 <- ind_disaster %>%
    filter(variable == "V_clamp", value > 0.0) %>%
    mutate(value = value/sd(value)) %>%
    group_by(date, hsiccd) %>% mutate(date_X_industry = cur_group_id()) %>% ungroup() %>%
    felm(data = ., formula = I(log(value)) ~ log_mktcap + bm + op + at_growth|date_X_industry|0|date)

reg3 <- ind_disaster %>%
    filter(variable == "IV_clamp", value > 0.0) %>%
    mutate(value = value/sd(value)) %>%
    group_by(date, hsiccd) %>% mutate(date_X_industry = cur_group_id()) %>% ungroup() %>%
    felm(data = ., formula = I(log(value)) ~ log_mktcap + bm + op + at_growth|date_X_industry|0|date)

stargazer(
    reg1, reg2, reg3, type="text", df=F,
    column.labels = c("D", "V", "IV"), dep.var.labels = c())

################################################################
# Estimating cross sectional regression for each month

sdD <- sd(log(filter(ind_disaster, variable == "D_clamp", value > 0.0)$value))
sdV <- sd(log(filter(ind_disaster, variable == "V_clamp", value > 0.0)$value))
sdIV <- sd(log(filter(ind_disaster, variable == "IV_clamp", value > 0.0)$value))

date_list <- sort(unique(ind_disaster$date))
coef_list <- list(); i <- 1
for (date_ in date_list) {
    print(paste(i,"out of",length(date_list)))
    reg1 <- ind_disaster %>%
        filter(variable == "D_clamp", value > 0.0, date == date_) %>%
        # mutate(value = value/sd(value)) %>%
        mutate(lhs = log(value)/sdD) %>%
        felm(data = ., formula = lhs ~ log_mktcap + bm + op + at_growth|0|0|0)

    reg2 <- ind_disaster %>%
        filter(variable == "V_clamp", value > 0.0, date == date_) %>%
        # mutate(value = value/sd(value)) %>%
        mutate(lhs = log(value)/sdV) %>%
        felm(data = ., formula = lhs ~ log_mktcap + bm + op + at_growth|0|0|0)

    reg3 <- ind_disaster %>%
        filter(variable == "IV_clamp", value > 0.0, date == date_) %>%
        # mutate(value = value/sd(value)) %>%
        mutate(lhs = log(value)/sdIV) %>%
        felm(data = ., formula = lhs ~ log_mktcap + bm + op + at_growth|0|0|0)
    
    # Formatting
    sum1 <- summary(reg1)
    sum2 <- summary(reg2)
    sum3 <- summary(reg3)

    df1 <- data.frame(sum1$coefficients)
    df2 <- data.frame(sum2$coefficients)
    df3 <- data.frame(sum3$coefficients)

    df1$rhs <- row.names(sum1$coefficients)
    df2$rhs <- row.names(sum2$coefficients)
    df3$rhs <- row.names(sum3$coefficients)

    df1$date <- date_
    df2$date <- date_
    df3$date <- date_

    df1$variable <- "D"
    df2$variable <- "V"
    df3$variable <- "IV"

    coef_list[[i]] <- bind_rows(df1, df2, df3); i <- i + 1
}

coefs <- bind_rows(coef_list) %>% as_tibble()
coefs$date <- as.Date(coefs$date, origin="1970-01-01")
colnames(coefs) <- c("Estimate", "Std Error", "t-value", "p-value", "rhs", "date", "variable")

# Plotting dynamic coefficients
coefs %>%
    mutate(low_conf = Estimate - 1.96*`Std Error`, upp_conf = Estimate + 1.96*`Std Error`) %>%
    ggplot(aes(x = date, y = Estimate, ymin = low_conf, ymax = upp_conf, color=variable, fill=variable)) + 
    geom_line() + geom_ribbon(alpha = 0.25) + facet_grid(rhs~., scales="free_y") +
    theme_minimal() + geom_hline(yintercept = 0.0, linetype="dashed")

