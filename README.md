# disaster-risk-revision

## Scripts used to generate D

1. load_zcb.py -- python code for downloading ZCB rates from OptionMetrics required to calculate option prices

2. loading_data.py -- python code for downloading option price data (includes filters that don't require distributions such as dividends) and distribution data (e.g. dividends). Uses python WRDS-SQL package that allows to write SQL scripts to WRDS database. Uses load_option_list.sql file

3. load_option_list.sql -- SQL script to load option price data from WRDS server

4. funcs.jl -- file with functions for the main Julia algorithm to fit SVI and estimate D for individual options data

5. fit_smiles.jl -- Julia file optimized for parallelization to fit SVI into implied volatilities. Outputs 4 SVI parameters (we set rho = 0). Note that this file is written for individual options as they have discrete dividends that need to be taken care off when calculating forward prices that enters Black-Scholes formula. To fit SVI for indices need to use another script

6. fit_smiles_index.jl -- Julia file to fit SVI for indices *!!! eed to check this one!!!*

6. est_parameters.jl -- Julia script to estimate D from fitted SVI curves

7. est_parameters_short.jl -- Julia script to estimate only a subset of D measures

8. OptionSmile.py -- old Python script to fit SVI. It was too slow so we switched to Julia

## Various analysis files that Roman needs to take care of...

1. analysis.py, calculate_D.py, compare_Ds.py, compare_clamp.py -- various files to load estimated data and calculate D + compare them and with Emil's measure of D

2. final_generating_scripts.py, generate_scripts.py, track_progress.py -- python files to generate shell scripts to estimate Ds

3. large_sigma_share.py -- share of observations with sigmas > 1

## Scripts with simulation results

1. simulations.jl -- main simulation file to assess accuracy and sensitivity of D to parameters when the process involves two jump processes

2. simulations_upper_limits.jl -- simulation to assess sensitivity of estimate to upper limit of integration

3. simulations_PCA.py, simulations_PCA.jl, simulations_PCA_parallel.jl -- files to simulations to assess performance of PCA in identifying the common component when noise is not iid

4. simulation_plots.R -- R code to plot simulation results


## Consider removing

analysis.py
