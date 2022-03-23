# Forming a long short portfolio of companies the most minus the
# least exposed to the disaster, eiher by the individual disaster
# measure or based on rolling beta w.r.t. aggregate disaster.
library("dplyr")
library("ggplot2")
library("tidyverse")

####################################################################
# Loading data

port_list <- list()

# 1. Using individual measure to sort companies
port_const <- read_csv("data/disaster_sorts/port_sort_const_ind_D_clamp.csv")
port_const <- port_const %>%
    select(permno, form_date, port = `('D_clamp', -99)`)
port_list[[1]] <- port_const

# 2. Using beta w.r.t. aggregate measure based on individual averaged JTIX
port_const <- read_csv("data/disaster_sorts/port_beta_sort_const.csv")
port_list[[2]] <- port_const %>%
    filter(variable == "D_clamp", maturity == "level", level == "Ind") %>%
    select(permno, form_date, port)

# 3. Using beta w.r.t. aggregate measure based on SPX JTIX
port_list[[3]] <- port_const %>%
    filter(variable == "D_clamp", maturity == "level", level == "SPX") %>%
    select(permno, form_date, port)

crsp_ccm <- read_csv("data/crsp_ccm_3.csv")
crsp_ccm <- crsp_ccm %>%
    select(permno, date = date_eom, mktcap = permco_mktcap, bm, op, at_growth, ret)

####################################################################
# Getting portfolio at the end of 2007, fixing them and following 
# the performance throughout 2008 and 2009

dynamic_df_list <- list()
dist_df_list <- list()
type_vec <- c("Individual JTIX", "Beta on Average Ind. JTIX", "Beta on SPX JTIX")

for (i in 1:3) {
    form_date_ <- as.Date("2007-12-31")
    port_form_date <- port_list[[i]] %>% 
        filter(form_date == form_date_, !is.na(port))  %>%
        select(-form_date)
    crsp_ccm_base <- crsp_ccm %>% 
        filter(date == form_date_) %>%
        mutate(cumret_base = 1+ret) %>%
        select(-date, permno, bm_base=bm, 
            op_base=op, at_growth_base=at_growth, cumret_base, -ret)

    dynamic <- crsp_ccm %>%
        filter(date >= form_date_, date <= "2009-12-31") %>%
        left_join(crsp_ccm_base, by="permno") %>%
        arrange(permno, date) %>%
        group_by(permno) %>%
        mutate(cumret=cumprod(1+ret)) %>%
        ungroup()
    dynamic <- dynamic %>%
        mutate(bm_diff = bm - bm_base, op_diff = op - op_base, 
            at_growth_diff = at_growth - at_growth_base, cumret=cumret/cumret_base)  %>%
        select(permno, date, bm_diff, op_diff, at_growth_diff, cumret)
    dynamic <- dynamic %>%
        left_join(port_form_date, by="permno")
    dynamic <- dynamic %>%
        filter(!is.na(port))

    # Plotting dynamic response
    dynamic_to_plot <- dynamic %>%
        filter(abs(op_diff) < 1) %>%
        # filter(abs(op_diff) < 1, abs(bm_diff) < 1, abs(at_growth_diff) < 1, abs(cumret) < 1) %>%
        group_by(date, port) %>%
        summarise(
            bm_diff = mean(bm_diff, na.rm=T), bm_diff_med = median(bm_diff, na.rm = T), bm_diff_5 = quantile(bm_diff, 0.05, na.rm=T),
            op_diff = mean(op_diff, na.rm=T), op_diff_med = median(op_diff, na.rm = T),
            at_growth_diff = mean(at_growth_diff, na.rm=T), at_growth_diff_med = median(at_growth_diff, na.rm = T),
            cumret = mean(cumret, na.rm=T), cumret_med = mean(cumret, na.rm=T)) %>%
        mutate(cumret = cumret - 1, cumret_med = cumret_med - 1) %>%
        ungroup() %>%
        # pivot_longer(cols=c(bm_diff_med, op_diff_med, at_growth_diff_med, cumret_med), names_to="variable") %>%
        select(date, port, bm_diff, op_diff, at_growth_diff, cumret) %>%
        pivot_longer(cols=c(bm_diff, op_diff, at_growth_diff, cumret), names_to="variable")

    # Plotting distributions at mid 2009
    dist_to_plot <- dynamic %>%
        filter(abs(op_diff) < 1) %>%
        filter(date == "2009-05-31", port %in% c(1,5)) %>%
        mutate(cumret = cumret-1) %>%
        pivot_longer(cols = c(bm_diff, at_growth_diff, op_diff, cumret)) %>%
        filter(abs(value) < 2.0) 

    # Writing into lists
    dynamic_to_plot$type <- type_vec[i]
    dist_to_plot$type <- type_vec[i]
    dist_df_list[[i]] <- dist_to_plot
    dynamic_df_list[[i]] <- dynamic_to_plot
}

# Reversing the portfolios for beta sorts (so that a higher
# portfolio number corresponds to more disaster risky)
replace_vec <- c("1 (least)", "2", "3", "4", "5 (most)")
names(replace_vec) <- c("1", "2", "3", "4", "5")

dynamic_df_list[[1]] <- dynamic_df_list[[1]] %>%
    mutate(port = replace_vec[as.character(port)])
dynamic_df_list[[2]] <- dynamic_df_list[[2]] %>%
    mutate(port = 6-port) %>%
    mutate(port = replace_vec[as.character(port)])
dynamic_df_list[[3]] <- dynamic_df_list[[3]] %>%
    mutate(port = 6-port) %>%
    mutate(port = replace_vec[as.character(port)])

# Plotting dynamics of fixed portfolios
bind_rows(dynamic_df_list) %>%
    mutate(date = as.Date(date)) %>%
    ggplot(aes(x = date, y = value, color = factor(port))) +
    geom_line(size = 0.75) +
    facet_grid(variable ~ type, scales="free") +
    theme_minimal() + geom_hline(yintercept=0.0, linetype="dashed")

ggsave("SS_figures/fixed_port_gfc.pdf", width=10, height=6)

# Plotting 5-1 portfolio for different sorts
bind_rows(dynamic_df_list) %>%
    filter(port %in% c("1 (least)", "5 (most)")) %>%
    pivot_wider(id_cols = c(date, type, variable), names_from = port, 
        values_from = c(value)) %>%
    mutate(`5 minus 1` = `5 (most)` - `1 (least)`) %>%
    ggplot(aes(x = date, y = `5 minus 1`, color = type)) +
    geom_line() +
    facet_grid(variable ~ ., scales = "free_y") +
    theme_minimal() + geom_hline(yintercept=0.0, linetype="dashed")

ggsave("SS_figures/fixed_port_gfc_5_minus_1.pdf", width=8, height=6)









# ggsave("SS_figures/fixed_port_gfc.pdf")


dist_to_plot %>%
    ggplot(aes(x = value, fill = factor(port))) + 
    geom_histogram(position="identity", alpha = 0.5) + 
    theme_minimal() +
    facet_grid(name ~ ., scales="free_y") +
    labs(title = "Distribution: May 2009 vs. Dec 2007")



# ggsave("SS_figures/fixed_port_gfc_dist.pdf")


################################################################
# Repeating for portfolios sorted on beta w.r.t. the aggregate
# disaster measures



