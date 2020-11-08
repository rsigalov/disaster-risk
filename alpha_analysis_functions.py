########################################################
# This file contains functions for analysis of alpha on
# portfolios. It requires files with portfolio returns
# in ../Results for a given number of window of Beta
# and number of portfolios.
#
#
########################################################


from __future__ import division
from __future__ import print_function

import csv
import numpy as np
from numpy import dot, mat, asarray, mean, size, shape, hstack, ones, ceil, \
zeros, arange
from numpy.linalg import inv, lstsq
from scipy.stats import chi2
from scipy.stats import f as F_dist
import pandas as pd
import seaborn as sns
from pandas.tseries.offsets import *

from matplotlib import pyplot as plt
import matplotlib
import EAPv2 as EAP
import wrds

import pprint
import warnings; warnings.simplefilter('ignore') # remove warning printing
import statsmodels.formula.api as smf # library with OLS regression

# Loading Fama French and Momentum factors from WRDS website as opposed to
# Returns a file with monthly factors:
def get_ff_factors():
    db = wrds.Connection()

    ff_query = """
    select
        date, mktrf, smb, hml, umd, rf
    from factors_monthly
    """
    factors_df = db.raw_sql(ff_query.replace('\n', ' '))
    factors_df['date'] = pd.to_datetime(factors_df['date']) + MonthEnd(0)
    factors_df = factors_df.dropna() # all factors are available from 1927-01-01 through 2018-05-01
    
    return factors_df

# Function to estimate alphas, betas (and covariance matrix) for a given set of parameters:
def factor_model(beta_length, num_buckets, factor_list,
                 sort_types = ['AAA', 'AA', 'A', 'BAA', 'HY'],
                 start_date = '1920-01-01', end_date = '2050-01-01'):
    
    # Uploading data (Portfolio returns, FF factors and Momentum factor)
    # TO DO: make downloading this stuff from the web directly
    Pdf = pd.read_csv('../Results/bucket_returns_' + str(beta_length) + 'yr_' + str(num_buckets) + 'buckets.csv')
    
    # Dealing with dates:
    Pdf['DATE_AHEAD'] = pd.to_datetime(Pdf['DATE_AHEAD'])

    factors_df = get_ff_factors()

    output_for_all_sort_types = {} # to store (nested) dictionaries of output
    
    for i_sort_type, sort_type in enumerate(sort_types):

        # Dealing with data periods
        Pdf_sort = Pdf[Pdf['sort_beta_rating'] == sort_type]
        Pdf_sort.drop('sort_beta_rating', inplace=True, axis = 1)
        
        # Merging factors and portfolio returns to make sure that there are
        # no missing observations
        Pdf_sort = pd.merge(Pdf_sort, factors_df, left_on = 'DATE_AHEAD', right_on = 'date', how = 'inner')

        # Subtracting risk-free rate to estimate factor model on excess returns
        # of portfolios
        for i_bucket in range(num_buckets):
            Pdf_sort['bucket_' + str(i_bucket + 1) + '_return'] = Pdf_sort['bucket_' + str(i_bucket + 1) + '_return'] - Pdf_sort['rf']/100
            
        # Slicing the right time interval:
        Pdf_sort = Pdf_sort[(Pdf_sort['DATE_AHEAD'] >= start_date) & 
                            (Pdf_sort['DATE_AHEAD'] <= end_date)]

        # Creating matrix with portfolios (with dimensions NxT where
        # N is the number of portfolios and T is number of periods)
        P = np.array(Pdf_sort[['bucket_' + str(x + 1) + '_return' for x in range(num_buckets)]])
        P = P.T
        
        # Getting the right factors from factor list:
        F = np.array(Pdf_sort[factor_list])
        F = F.T
        
        # Estimating the model:
        est_results = EAP.time_series(P, F)
        
        output_for_all_sort_types[sort_type] = est_results
        
    return output_for_all_sort_types

# This function takes as input output of function factor_model() and
# plots alphas with error bars
def plot_alphas(factor_model_output):
    sort_types = list(factor_model_output.keys())
    n_sort_types = len(sort_types)
    
    fig, axes = plt.subplots(nrows = n_sort_types // 2 + 2, ncols = 2,  figsize = (11, 5 * (n_sort_types // 2 + 1)))
    axes_list = [item for sublist in axes for item in sublist] 
    
    for sort_type in sort_types:
        alpha = factor_model_output[sort_type]['alpha']
        T = factor_model_output[sort_type]['T']
        alpha_se = np.sqrt(np.diag(factor_model_output[sort_type]['VCV']))[0:len(alpha)]
        x_port_values = [x + 1 for x in range(len(alpha))]

        ax = axes_list.pop(0)
        ax.errorbar(x = x_port_values, y = alpha, yerr = list(alpha_se),
                    fmt="--o", linewidth=3, elinewidth=0.5, ecolor='k',
                    capsize=5, capthick=0.5, markersize = 5)

        ax.set_title('Sorted on ' + sort_type + ' beta')    
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    for ax in axes_list:
        ax.remove()

    plt.show()


# This function takes as input output of function factor_mode() and
# tests whether alphas for all portfolios are equal (a1 = ... = aN)
def test_hyp_all_equal(factor_model_output):
    sort_types = list(factor_model_output.keys())
    output = {}
    
    for sort_type in sort_types:
        output_sort_type = factor_model_output[sort_type]
        
        vcv = output_sort_type['VCV']
        N = len(output_sort_type['alpha'])
        vcv_a = vcv[0:N,0:N]
        a_hat = output_sort_type['alpha']

        # Constructing constraint matrix R (N-1 constraints and N alphas):
        R = np.hstack((np.ones((N-1, 1)), (-1) * np.eye(N-1)))
        # test statistics
        bread = R.dot(a_hat)
        meat = R.dot(vcv_a).dot(R.T)
        test_stat = (bread.T).dot(inv(meat)).dot(bread)
        p_value =  1 - chi2.cdf(test_stat, N-1)
        
        # writing results into return dictionary:
        output_sort_type = {}
        output_sort_type['chi_stat'] = float(test_stat)
        output_sort_type['p_value'] = float(p_value)
        output[sort_type] = output_sort_type
    
    return output  

# This function takes as input output of function factor_mode() and
# tests whether alphas for portfolios 1 and N are equal (a1 = aN)
def test_hyp_first_last_equal(factor_model_output):
    sort_types = list(factor_model_output.keys())
    output = {}
    
    for sort_type in sort_types:
        output_sort_type = factor_model_output[sort_type]
        
        vcv = output_sort_type['VCV']
        N = len(output_sort_type['alpha'])
        vcv_a = vcv[0:N,0:N]
        a_hat = output_sort_type['alpha']

        # testing for equality of first and the last of portfolios:
        R_first_last = np.zeros((1,N))
        R_first_last[0,0] = 1
        R_first_last[0,N-1] = -1
        # test statistics
        bread = R_first_last.dot(a_hat)
        meat = R_first_last.dot(vcv_a).dot(R_first_last.T)
        test_stat = (bread.T).dot(inv(meat)).dot(bread)
        p_value =  1 - chi2.cdf(test_stat, 1)
        
        # writing results into return dictionary:
        output_sort_type = {}
        output_sort_type['chi_stat'] = float(test_stat)
        output_sort_type['p_value'] = float(p_value)
        output[sort_type] = output_sort_type
    
    return output   

# This function takes as input output of function factor_model() estimated
# with a only one factor and plots the coefficient on this factor for
# different portfolios with error bars
def plot_capm_betas(factor_model_output, path_to_save = None):
    
    sort_types = list(factor_model_output.keys())
    n_sort_types = len(sort_types)
    
    fig, axes = plt.subplots(nrows = n_sort_types // 2 + 2, ncols = 2,  figsize = (11, 4 * (n_sort_types // 2 + 1)))
    axes_list = [item for sublist in axes for item in sublist] 
    
    for sort_type in sort_types:
        alpha = factor_model_output[sort_type]['beta']
        T = factor_model_output[sort_type]['T']
        alpha_se = np.sqrt(np.diag(factor_model_output[sort_type]['VCV']))[len(alpha):]
        x_port_values = [x + 1 for x in range(len(alpha))]

        ax = axes_list.pop(0)
        ax.errorbar(x = x_port_values, y = alpha, yerr = list(alpha_se),
                    fmt="--o", linewidth=3, elinewidth=0.5, ecolor='k',
                    capsize=5, capthick=0.5, markersize = 5)

        ax.set_title('Sorted on ' + sort_type + ' beta')    
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    for ax in axes_list:
        ax.remove()

    if path_to_save is Null:
    	plt.show()
    else:
    	plt.savefig(path_to_save, bbox_inches='tight', format = 'pdf')

# Function from 
#   https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
# that sets prettier colors:
def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

# colormap = cmap_map(lambda x: 0.5 * x + 0.5, matplotlib.cm.Oranges)
def plot_heat_map_list(matrix_list, xlabel_list, ylabel_list, matrix_label_list, path_to_save = None):
    num_matrices = len(matrix_list)

    if num_matrices == 1:
        i_corr_mat = 0
        x_tick_num = len(xlabel_list)
        y_tick_num = len(ylabel_list)

        ax = plt.subplot()

        colormap = cmap_map(lambda x: 0.5 * x + 0.5, matplotlib.cm.Oranges)

        im = ax.imshow(np.array(matrix_list[i_corr_mat]), cmap = colormap)

        ax.set_xticks(np.arange(x_tick_num))
        ax.set_yticks(np.arange(y_tick_num))
        ax.set_yticklabels(xlabel_list)
        ax.set_xticklabels(ylabel_list)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(x_tick_num):
            for j in range(y_tick_num):
                text = ax.text(j, i, round(matrix_list[i_corr_mat][i, j] * 100,1),
                               ha="center", va="center", color="black", weight = 'bold')

        ax.set_title(matrix_label_list[i_corr_mat])

    else:

        fig, axes = plt.subplots(nrows = num_matrices // 2 + 1, ncols = 2,  figsize = (7, 3.5 * (num_matrices // 2 + 1)))
        axes_list = [item for sublist in axes for item in sublist] 

        colormap = cmap_map(lambda x: 0.5 * x + 0.5, matplotlib.cm.Oranges)

        for i_corr_mat in range(num_matrices):
            ax = axes_list.pop(0)
            im = ax.imshow(np.array(matrix_list[i_corr_mat]), cmap = colormap)

            ax.set_xticks(np.arange(x_tick_num))
            ax.set_yticks(np.arange(y_tick_num))
            ax.set_yticklabels(xlabel_list)
            ax.set_xticklabels(ylabel_list)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(x_tick_num):
                for j in range(y_tick_num):
                    text = ax.text(j, i, round(matrix_list[i_corr_mat][i, j] * 100,1),
                                   ha="center", va="center", color="black", weight = 'bold')

            ax.set_title(matrix_label_list[i_corr_mat])

        for ax in axes_list:
            ax.remove()

    plt.tight_layout()
    if path_to_save is None:
    	plt.show()
    else:
    	plt.savefig(path_to_save, bbox_inches='tight', format = 'pdf')


