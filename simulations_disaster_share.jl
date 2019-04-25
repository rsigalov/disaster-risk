


################################################################################
# Calculating share of disaster D in overall D for different parameters
# values
################################################################################

lambda_disaster = 0.05
mean_disaster = -0.8
sigma_disaster = 0.1
lambda_small = 20
mean_small = -0.05
sigma_small = 0.01

# Function to calculate the slope of D (that multiplies lambda):
function slope(mu, sigma)
    return 2 * (1 + mu + 0.5*(mu^2 + sigma^2) - exp(mu + 0.5sigma^2))
end

# Calculating theoretical value of D for baseline scenarios:
function share_calc(lambda_disaster, mean_disaster, sigma_disaster,
                    lambda_small, mean_small, sigma_small)
    # D from disaster
    D_disaster = slope(mean_disaster, sigma_disaster) * lambda_disaster
    # D from small jumps
    D_small = slope(mean_small, sigma_small) * lambda_small
    # Overall D
    D_full = D_disaster + D_small

    return D_disaster/D_full
end

mean_small_list = LinRange(-0.1, 0.0, 51)
share_list_1 = map(x -> share_calc(0.05, -0.8, 0.1, 20, x, 0.05), mean_small_list)
share_list_2 = map(x -> share_calc(0.05, -0.8, 0.1, 20, x, 0.025), mean_small_list)
share_list_3 = map(x -> share_calc(0.05, -0.8, 0.1, 20, x, 0.01), mean_small_list)
case_1_point = share_calc(0.05, -0.8, 0.1, 20, -0.05, 0.05)
case_2_point = share_calc(0.05, -0.8, 0.1, 20, -0.005, 0.05)
case_3_point = share_calc(0.05, -0.8, 0.1, 20, -0.005, 0.01)

Plots.plot(mean_small_list, share_list_1,label = "sigma = 0.05",
    legend = :bottomright,
    xaxis = "mu_small", yaxis = "disaster D/(disaster D + small D)")
Plots.plot!(mean_small_list, share_list_2,label = "sigma = 0.025")
Plots.plot!(mean_small_list, share_list_3,label = "sigma = 0.01")
Plots.scatter!([-0.05],[case_1_point], label = "Case I")
Plots.scatter!([-0.005],[case_2_point], label = "Case II")
Plots.scatter!([-0.005],[case_3_point], label = "Case III")
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/disaster_D_share.pdf")
