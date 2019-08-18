import numpy as np
from numpy import dot, mat, asarray, mean, size, shape, hstack, ones, ceil, \
zeros, arange
from numpy.linalg import inv, lstsq
from scipy.stats import chi2
from scipy.stats import f as F_dist
import pandas as pd
import copy

# For plotting functions
import matplotlib.pyplot as plt
import matplotlib

from pandas.tseries.offsets import * # For MonthEnd(0)
import wrds # Downloading Fama-French and Momentum factors
import alpha_analysis_functions as aaf # Function to estimate factor models 
                                       # plot alphas and betas for portfolios
                                       # and plot matrices


class EAP(object):

    def __init__(self, p, db = None, supress_printing = False, estimate_drawdowns = True, 
    	constituents = None, hml = True, zc = []):

        self.N = len(p.columns)
        self.portfolio = p
        self.portfolio.index = pd.to_datetime(self.portfolio.index)
        self.names = p.columns

        self.constituents = constituents

        # Calculating transition matrix if constituents are provided:
        # if constituents is not None:
        #     self.transition_matrix = self.compute_transition_matrix()

        if not supress_printing:
            print('Downloading Risk Factors')
        self.famafrench_factors =  self.get_FamaFrench_factors(db)

        self.TS_results = {}
        self.TS_summary = None
        self.Ann_Info_Ratio = None
        if not supress_printing:
            print('Computing Basic Statistics')
        self.raw_stats = self.summary_stats_port_df()

        if estimate_drawdowns:
	        if not supress_printing:
	            print('Estimatig maximum drawdowns for portfolios')
	        self.estimate_drawdowns()

        if not supress_printing:
                print('Estimatig maximum drawdown based on HWM')
        self.drawdown_hwm = self.estimate_drawdowns_hwm()

        if not supress_printing:
            print('Estimating CAPM')
        self.capm_stats = self.factor_model(model = 'capm')

        if not supress_printing:
            print('Estimating 3 Factor Model')
        self.ff_stats = self.factor_model(model = '3f')

        if not supress_printing:
            print('Estimating 4 Factor Model')
        self.ffmom_stats = self.factor_model(model = '4f')

    @property
    def excess_returns(self):

        xs = pd.merge(self.portfolio, self.famafrench_factors[['rf']], on = 'date', how = 'inner')

        for col in self.portfolio.columns:
            xs[col] = xs[col] - xs['rf']

        xs = xs.drop('rf', axis = 1)

        return xs

    def get_FamaFrench_factors(self, db = None):
        if db is None:
            db = wrds.Connection()

        ff_query = """
        select
            date, mktrf, smb, hml, umd, rf
        from factors_monthly
        """
        factors_df = db.raw_sql(ff_query.replace('\n', ' '))

        factors_df['date'] = pd.to_datetime(factors_df['date']) + MonthEnd(0)
        factors_df = factors_df.dropna()
        factors_df = factors_df.set_index('date')

        return factors_df

    def summary_stats_port_df(self):
        stats_df = pd.DataFrame(list(self.portfolio.apply(summary_stats_port_ts, axis = 0)))
        stats_df.index = self.names
        return stats_df

    # This function finds a local minimum and maximum of portfolios' cumulative profit. A
    # point in time t is a local minimum (maximum) if it is at least a great as all point
    # in time t-12, t-11,..., t+12. Local maximum followed by a local minimum is defined
    # as a drawdown if the following local minimum is less than preceding local maximum
    def estimate_drawdowns(self):
        drawdown_df_list = [] # list of dataframes to store in instance
        max_drawdown_list = [] # to add to raw_stats
        max_drawdown_start_list = [] # to add to raw_stats
        max_drawdown_end_list = [] # to add to raw_stats

        for name in self.names:
            port_index = (self.portfolio[name] + 1).cumprod()
            drawdown_start = []
            drawdown_end = []

            for t in range(12, port_index.shape[0] - 12):
                if port_index[t] >= max(port_index[(t - 12):(t + 12)]):
                    drawdown_start.append(t)

                if port_index[t] <= min(port_index[(t - 12):(t + 12)]):
                    drawdown_end.append(t)

            drawdown_end_att = []
            for t in drawdown_start:
                after_end_list = [x for x in drawdown_end if x > t]
                if len(after_end_list) > 0:
                    if (port_index[t] > port_index[after_end_list[0]]): # making sure that value at the start of draw down is higher than at the end
                        drawdown_end_att.append(after_end_list[0])
                    else:
                        drawdown_end_att.append(None)
                else:
                    drawdown_end_att.append(None)

            # Combining everyting into data frame:
            drawdown_df = pd.DataFrame(columns = ['start_date','start_value', 'end_date', 'end_value'])
            start_date_list = []
            end_date_list = []
            start_value_list = []
            end_value_list = []
            for start_i, start_t in enumerate(drawdown_start):
                if drawdown_end_att[start_i] is not None:
                    start_date_list.append(port_index.index[start_t])
                    end_t = drawdown_end_att[start_i]
                    end_date_list.append(port_index.index[end_t])
                    start_value_list.append(port_index[start_t])
                    end_value_list.append(port_index[end_t])


                    drawdown_df = pd.DataFrame({'start_date': start_date_list,
                                                'start_value': start_value_list,
                                                'end_date': end_date_list,
                                                'end_value': end_value_list})

            drawdown_df['drawdown'] = drawdown_df['end_value']/drawdown_df['start_value'] - 1

            drawdown_df_list.append(drawdown_df)

            # Info about max drawdowns
            max_ind = drawdown_df['drawdown'] == min(drawdown_df['drawdown'])
            max_row = drawdown_df.loc[max_ind]
            
            max_drawdown_list.append(list(max_row['drawdown'])[0])
            max_drawdown_start_list.append(list(max_row['start_date'])[0])
            max_drawdown_end_list.append(list(max_row['end_date'])[0])

        # Adding info about max drawdown to raw statistics:
        self.raw_stats['max_down'] = max_drawdown_list
        self.raw_stats['max_down_s'] = max_drawdown_start_list
        self.raw_stats['max_down_e'] = max_drawdown_end_list

        self.drawdowns = drawdown_df_list

    def estimate_drawdowns_hwm(self):
        # Function to calculate drawdown time series using High Water Mark (HWM)
        position = (self.portfolio + 1).cumprod()
        drawdown_hwm = pd.DataFrame(columns = self.names)

        for port_num in range(self.N):
            port_pos = position.iloc[:, port_num]
            port_drawdown_list = []
            for t in range(port_pos.shape[0]):
                hwm = max(port_pos[0:t+1])
                port_drawdown_list.append((port_pos[t] - hwm)/hwm)
                
            drawdown_hwm[self.names[port_num]] = port_drawdown_list
            
        drawdown_hwm.index = self.portfolio.index

        return drawdown_hwm
    
    # Calculate alphas with respect to specified factors:
    def factor_model(self, model):
        if model == 'capm':
            factors = ['mktrf']
        elif model == '3f':
            factors = ['mktrf', 'smb', 'hml']
        elif model == '4f':
            factors = ['mktrf', 'smb', 'hml', 'umd']
        else:
            print('!!!! Model ' + model + ' is not supported.')
            print('!!!! Doing 4 Factors instead')
            factors = ['mktrf', 'smb', 'hml', 'umd']
            
        p_ff_merged = pd.merge(self.excess_returns, self.famafrench_factors[factors], how = 'inner', on = 'date')
        P = np.array(p_ff_merged.iloc[:, [x for x in range(self.N)]]).T
        F = np.array(p_ff_merged.iloc[:, [x + self.N for x in range(len(factors))]]).T
        
        return time_series(P, F)
    
    def plot_alphas(self, model = '4f', ax = None):
        obj_to_plot = self.factor_stats_for_model(model)
            
        alpha = obj_to_plot['alpha']
        alpha_se = np.sqrt(np.diag(obj_to_plot['VCV']))[0:self.N]
        
        return plot_values_with_error_bars(alpha, alpha_se, ax)
    
    def plot_capm_betas(self, ax):
        
        beta = self.capm_stats['beta']
        beta_se = np.sqrt(np.diag(self.capm_stats['VCV']))[self.N:(self.N*2)]
        
        return plot_values_with_error_bars(beta, beta_se, ax)
    
    # Three methods for testing alphas
    def test_all_zero(self, model = '4f'):
        stats = self.factor_stats_for_model(model)
        N = self.N
        
        R = np.eye(N)
        vcv_a = stats['VCV'][0:N, 0:N]
        a_hat = stats['alpha']
        
        return test(R, vcv_a, a_hat)
    
    def test_all_equal(self, model = '4f'):
        stats = self.factor_stats_for_model(model)
        N = self.N
        
        R = np.hstack((np.ones((N-1, 1)), (-1) * np.eye(N-1)))
        vcv_a = stats['VCV'][0:N, 0:N]
        a_hat = stats['alpha']
        
        return test(R, vcv_a, a_hat)
    
    def test_first_last_equal(self, model = '4f'):
        stats = self.factor_stats_for_model(model)
        N = self.N
        
        # testing for equality of first and the last of portfolios:
        R = np.zeros((1,N))
        R[0,0] = 1
        R[0,N-1] = -1
        vcv_a = stats['VCV'][0:N, 0:N]
        a_hat = stats['alpha']
        
        return test(R, vcv_a, a_hat)
        
    def factor_stats_for_model(self, model):
        if model == 'capm':
            return self.capm_stats
        elif model == '3f':
            return self.ff_stats
        elif model == '4f':
            return self.ffmom_stats
        else:
            print('!!!! Model ' + model + ' is not supported.')
            print('!!!! Doing 4 Factors instead')
            return self.ffmom_stats

    # This function plots
    def plot_portfolio_performance(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,5))

        # Plotting recessions:
        rec_start_end = pd.read_csv('../Input_Data/recessions_start_end.csv')
        rec_start_end['start'] = pd.to_datetime(rec_start_end['start'])
        rec_start_end['end'] = pd.to_datetime(rec_start_end['end'])
        rec_start_end_short = rec_start_end[rec_start_end['end'] >= min(self.portfolio.index)]

        for i_row in range(rec_start_end_short.shape[0]):
            ax.axvspan(rec_start_end_short.iloc[i_row]['start'], 
                       rec_start_end_short.iloc[i_row]['end'],
                       alpha=0.2, color='grey')

        # Plotting portfolio cumulative return:
        linewidth_list = [2] + [1] * (self.N - 2) + [2]
        linestyle_list = ['-'] + ['--'] * (self.N - 2) + ['-']
        for i in range(self.N):
            ax.semilogy(self.portfolio.index, (self.portfolio.iloc[:, [i]] + 1).cumprod(), 
                linewidth = linewidth_list[i], linestyle = linestyle_list[i])

        # Adding cumulative return on the market:
        ff_short = self.famafrench_factors[self.famafrench_factors.index.isin(self.portfolio.index)]
        ax.semilogy(ff_short.index, (ff_short['mktrf'] + ff_short['rf'] + 1).cumprod(), linestyle = ':')

        ax.legend(list(self.portfolio.columns) + ['market'])

        return ax

    def plot_drawdown_hwm_ts(self, plot_all = False, ax = None): # by default plot drawdow for first and 
                                                         # last portfolios. Alternatively, plot all
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,5))

        for port_num in range(self.N) if plot_all else [0, self.N - 1]:
            plt.plot(self.drawdown_hwm.index, self.drawdown_hwm.iloc[:, port_num], 
                     alpha=0.5, label = 'Portfolio ' + str(port_num + 1))
            
        plt.legend(loc='bottom right')

        return ax

    def plot_drawdown_hwm_hist(self, plot_all = False, ax = None):

        if ax is None:
            fig, ax = plt.subplots(figsize = (8,5))

        # Plotting distribution of drawdowns
        min_drawdown = min(self.drawdown_hwm.min()) if plot_all else min(self.drawdown_hwm.iloc[:, [0, self.N - 1]].min())
        bins = np.linspace(min_drawdown, 0, 50)

        # Excluding observations with zero drawdown since there will be just a huge peak there
        for port_num in range(self.N) if plot_all else [0, self.N - 1]:
            plt.hist(self.drawdown_hwm.iloc[:, port_num][self.drawdown_hwm.iloc[:, port_num] < 0], bins, 
                alpha=0.5, label = 'Portfolio ' + str(port_num + 1))
            
        plt.legend(loc='upper left')

        return ax


def plot_coef_list(portfolio_list, title_list, plot_type, model = '4f'):
    portfolio_num = len(portfolio_list)
    fig = plt.figure()
    
    fig_height = 3*(portfolio_num//2 + 1)
    fig_width = 4*2
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
    # Trying to plot multiple plots on the same figure:
    for i in range(portfolio_num):
        ax = fig.add_subplot(portfolio_num//2 + 1, 2, i + 1)
        ax.set_title(title_list[i])
        if plot_type == 'alpha':
            portfolio_list[i].plot_alphas(model, ax)
        else:
            portfolio_list[i].plot_capm_betas(ax)

    
    return fig

def plot_performance_list(portfolio_list, title_list):
    portfolio_num = len(portfolio_list)
    fig = plt.figure()
    
    fig_height = 5 * (portfolio_num//2 + 1)
    fig_width = 8 * 2
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
    # Trying to plot multiple plots on the same figure:
    for i in range(portfolio_num):
        ax = fig.add_subplot(portfolio_num//2 + 1, 2, i + 1)
        ax.set_title(title_list[i])
        portfolio_list[i].plot_portfolio_performance(ax)
    
    return fig

def plot_drawdow_hwm_ts_list(portfolio_list, title_list, plot_type = 'hist', plot_all = False):
    portfolio_num = len(portfolio_list)
    fig = plt.figure()
    
    fig_height = 5 * (portfolio_num//2 + 1)
    fig_width = 8 * 2
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
    # Trying to plot multiple plots on the same figure:
    for i in range(portfolio_num):
        ax = fig.add_subplot(portfolio_num//2 + 1, 2, i + 1)
        ax.set_title(title_list[i])
        if plot_type == 'hist':
            portfolio_list[i].plot_drawdown_hwm_hist(plot_all = plot_all, ax = ax)
        elif plot_type == 'ts':
            portfolio_list[i].plot_drawdown_hwm_ts(plot_all = plot_all, ax = ax)
        else:
            print('!!! Plot type '+ str(plot_type) + ' is not supported')
            print('!!! Plotting histogram instead')
            portfolio_list[i].plot_drawdown_hwm_hist(plot_all = plot_all, ax = ax)
    
    return fig
    

def summary_stats_port_ts(port_ts):
    stats_output = {}

    # Basic summary statistics:
    stats_output['T'] = port_ts.shape[0]
    stats_output['mean'] = np.mean(port_ts)
    stats_output['std'] = np.std(port_ts)
    stats_output['median'] = np.median(port_ts)
    stats_output['min'] = min(port_ts)
    stats_output['date_min'] = pd.to_datetime(port_ts[port_ts == min(port_ts)].index).strftime('%Y-%m-%d')[0]
    stats_output['max'] = max(port_ts)
    stats_output['date_max'] = pd.to_datetime(port_ts[port_ts == max(port_ts)].index).strftime('%Y-%m-%d')[0]

    return stats_output


def test(R, vcv_a, a_hat):
    bread = R.dot(a_hat)
    meat = R.dot(vcv_a).dot(R.T)
    test_stat = (bread.T).dot(inv(meat)).dot(bread)
    p_value =  1 - chi2.cdf(test_stat, R.shape[0])

    # writing results into return dictionary:
    output = {}
    output['chi_stat'] = float(test_stat)
    output['p_value'] = float(p_value)

    return output
              

def plot_values_with_error_bars(values, errors, ax = None):
    return_all = False
    if ax is None:
        return_all = True
        fig, ax = plt.subplots()
    
    x_port_values = [x + 1 for x in range(len(values))]
    ax.errorbar(x = x_port_values, y = values, yerr = list(errors),
                fmt="--o", linewidth=3, elinewidth=0.5, ecolor='k',
                capsize=5, capthick=0.5, markersize = 5)
    
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    return ax
    
def time_series(P, F):

    ### Read in inputs
    N,T_P = shape(P)
    N=int(N)
    T_P=int(T_P)

    K,T_F = shape(F)
    K=int(K)
    T_F=int(T_F)

    if T_P != T_F:
        raise Exception('Time-series of portfolios must be same length as time-series of factors')
    else:
        T=T_P

    Lambda = np.mean(F,1)
    if K>1:
        Sigma_f = np.cov(F)
        t_Lambda =  Lambda/np.sqrt(np.diag(Sigma_f)/T)
    elif K==1:
        Sigma_f = np.var(F)
        t_Lambda = Lambda/np.sqrt(Sigma_f/T)

    
    Lambda = np.array([Lambda]).T

    ### Stack the portfolios to run stacked regression
    stacked_Y = np.ndarray.flatten(P)
    F_aug = np.hstack((np.ones((T,1)),F.T)) # Transpose the factors, and add a column of ones for the regression constant
    stacked_X = np.kron(np.eye(N),F_aug)

    stacked_b,stacked_vcv,stacked_s2,stacked_R2,stacked_R2bar,stacked_e = olsnw(stacked_Y,stacked_X,constant=False)


    ### Put coefficients and errors into their own vectors
    alpha = np.empty((N,1))
    beta=np.empty((N,K))


    Epsilon = np.empty((N,T))
    for p in range(0,N):
        alpha[p,:]=stacked_b[p*(K+1)]
        beta[p,:]=np.ndarray.flatten(stacked_b[p*(K+1)+1:(p+1)*(K+1)])

        Epsilon[p,:]=np.ndarray.flatten(stacked_e[p*T:(p+1)*T])

    #### Calculate standard errors ####
    Ktilde=K+1
    moments = np.empty((N*(K+1),T))


    for t in range(0,T):
        temp1=np.reshape(Epsilon[:,t],(N,1))
        temp2=np.reshape(np.kron(Epsilon[:,t],F[:,t]),(N*K,1))
        moments[:,t]=np.reshape(np.vstack((temp1,temp2)),(N*(K+1),))

    #Demean moments (as per Cochrane suggestion)
    mean_moments = np.reshape(np.mean(moments,1),(N*Ktilde,1))
    mean_moments_mat = np.kron(np.ones((1,N*Ktilde)),mean_moments)
    moments = moments - mean_moments

    #Construct D matrix
    mean_F = np.array([np.mean(F,1)]).T
    FFp=F.dot(F.T)/T
    D1=np.hstack((np.eye(N),np.kron(np.eye(N),mean_F.T)))

    D=copy.deepcopy(D1)

    for i in range(0,N):
        io = iota(N, i+1)
        D2_a = np.kron(io, mean_F)
        D2_b = np.kron(io, FFp)
        D2=np.hstack((D2_a, D2_b))
        D=np.vstack((D, D2))

    #Construct S matrix
    lags = int(ceil(1.2 * float(T)**(1.0/3)))
    w = 1 - arange(0,lags+1)/(lags+1)
    
    Gamma = zeros((lags+1,N*Ktilde,N*Ktilde))
    for lag in range(lags+1):
        Gamma[lag] = moments[:,lag:].dot(moments[:,:T-lag].T)

    Gamma = Gamma/T
    S = Gamma[0].copy()
    for i in range(1,lags+1):
        S = S + w[i] * (Gamma[i] + Gamma[i].T)
    

    VCV = inv(D).dot(S).dot(inv(D).T)/T 
    alpha_vcv = VCV[0:N,0:N]
    beta_vcv = VCV[N:,N:]

    alpha_se = np.sqrt(np.diag(alpha_vcv))
    beta_se = np.sqrt(np.diag(beta_vcv))

    t_alpha = np.empty((N,1))
    t_beta = np.empty((N,K))

    for p in range(0,N):
        t_alpha[p,:] = alpha[p,:] / alpha_se[p]
        t_beta[p,:] = beta[p,:] / beta_se[p*K:(p+1)*K]


    # Joint test that all alpha are zero.
    Chi_stat = float((alpha[0:N-1].T).dot(inv(alpha_vcv[0:N-1, 0:N-1])).dot(alpha[0:N-1]))
    p_value =  1 - chi2.cdf(Chi_stat,N-1)

    ### Calculate R2 for each time-series regression

    R2_all=np.empty((N,1))
    R2bar_all=np.empty((N,1))
    info_ratio = np.empty((N,1))

    for p in range(0,N):
        Y = P[p,:]
        X=np.transpose(copy.copy(F))

        b,vcv,s2,R2,R2bar,e = olsnw(Y,X)
        R2_all[p,:]=R2
        R2bar_all[p,:]=R2bar

        info_ratio[p, :] = alpha[p] / np.std(e)


    #Output dictionary
    outputdict={}
    outputdict['alpha']=alpha
    outputdict['t_alpha']=t_alpha
    outputdict['lambda']=Lambda
    outputdict['t_lambda']=t_Lambda
    outputdict['Chi_stat']=Chi_stat
    outputdict['p_value']=p_value
    outputdict['R2_all']=R2_all
    outputdict['R2bar_all']=R2bar_all
    outputdict['beta']=beta
    outputdict['t_beta']=t_beta
    outputdict['info_ratio'] = info_ratio
    outputdict['VCV'] = VCV
    outputdict['T'] = T

    return outputdict
              
def olsnw(y, X, constant=True, lags=None):
    T = y.size
    if size(X, 0) != T:
        X = X.T
        T,K = shape(X)
    if constant:
        X = copy.copy(X)
        X = hstack((ones((T,1)),X))
        K = size(X,1)

    K=size(X,1)
    if lags==None:
        lags = int(ceil(1.2 * float(T)**(1.0/3)))
    # Parameter estimates and errors
    out = lstsq(X,y)
    b = out[0]
    e = np.reshape(y - dot(X,b),(T,1))
    
    # Covariance of errors    
    gamma = zeros((lags+1))
    for lag in range(lags+1):
        gamma[lag] = e[:T-lag].T.dot(e[lag:]) / T
    w = 1 - arange(0,lags+1)/(lags+1)       
    w[0] = 0.5

    s2 = dot(gamma,2*w)
    
    # Covariance of parameters
    Xe = mat(zeros(shape(X)))
    for i in range(T):
        Xe[i] = X[i] * float(e[i])
    Gamma = zeros((lags+1,K,K))
    for lag in range(lags+1):
        Gamma[lag] = Xe[lag:].T*Xe[:T-lag]

    Gamma = Gamma/T
    S = Gamma[0].copy()
    for i in range(1,lags+1):
        S = S + w[i]*(Gamma[i] + Gamma[i].T)
    XpX = dot(X.T,X)/T
    XpXi = inv(XpX)
    vcv = mat(XpXi)*S*mat(XpXi)/T
    vcv = asarray(vcv)
    # R2, centered or uncentered

    if constant:
        R2 = e.T.dot(e)/( (y-mean(y)).T.dot(y-mean(y))) 
    else:
        R2 = e.T.dot(e)/((y.T).dot(y))

    R2bar = 1-R2*(T-1)/(T-K)
    R2 = 1 - R2
    return b,vcv,s2,R2,R2bar,e

def iota(N,i):
    if i>N:
        raise Exception('Index can not be longer than vector')
    temp = np.zeros((1,N))
    temp[0,i-1]=1
    return temp

# variable plot_type can be either 'alpha' or 'beta'. In case
# of plot_type = 'alpha' need to also specify model. In case of
# plot_type = 'beta' will plot CAPM betas: