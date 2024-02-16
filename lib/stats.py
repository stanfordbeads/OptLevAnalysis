import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2,norm
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import jit
from funcs import *

# ************************************************************************ #
# This file contains the functions used for statistical analysis of
# AggregateData objects.
# ************************************************************************ #

def fit_alpha_all_files(agg_dict,file_indices=None,sensor='qpd',use_ml=False,ml_veto=False,num_cores=1):
    '''
    Find the best fit alpha to a dataset. Most of this should eventually be moved to
    stats.py so that different likelihood functions can be called from here.
    '''

    if file_indices is None:
        file_indices = np.array(range(agg_dict['times'].shape[0]))

    print('Computing the likelihood functions for the specified files...')
    likelihood_coeffs = Parallel(n_jobs=num_cores)(delayed(fit_alpha_for_file)\
                                                   (agg_dict,ind,sensor,use_ml,ml_veto)\
                                                    for i,ind in enumerate(tqdm(file_indices)))

    return np.array(likelihood_coeffs)


def fit_alpha_for_file(agg_dict,file_index,sensor='qpd',use_ml=False,ml_veto=False):
    '''
    Find the best fit alpha for a single file and return it. This function will be
    called in parallel so that all files can be processed to produce a limit.
    '''

    lambdas = agg_dict['template_params'][file_index]
    yuk_ffts = agg_dict['template_ffts'][file_index]
    bead_ffts = agg_dict[sensor+'_ffts'][file_index]
    bead_sb_ffts = agg_dict[sensor+'_sb_ffts'][file_index]
    num_sb = int(bead_sb_ffts.shape[1]/bead_ffts.shape[1])
    good_inds = agg_dict['good_inds']
    mot_likes = agg_dict['mot_likes'][file_index]
    likelihood_coeffs = np.zeros((len(good_inds),3,len(lambdas),4),dtype=np.float128)
    
    for harm,_ in enumerate(good_inds):
        for axis in [0,1,2]:

            sb_fft_array = bead_sb_ffts[axis,harm*num_sb:(harm+1)*num_sb]
            data = bead_ffts[axis,harm]
            if use_ml and axis!=2:
                if not ml_veto:
                    ml = mot_likes[axis,harm]
                    noise = np.mean(np.abs(sb_fft_array)**2)
                    data = np.sqrt(ml*max(0,np.abs(data)**2-noise) + noise)*np.exp(1j*np.angle(data))
            data_real = np.real(data)
            data_imag = np.imag(data)
            var = (1./(2.*num_sb))*np.sum(np.real(sb_fft_array)**2+np.imag(sb_fft_array)**2)

            for lambda_ind,_ in enumerate(lambdas):
                if use_ml and axis!=2:
                    if ml_veto and mot_likes[axis,harm]<0.5:
                        likelihood_coeffs[harm,axis,lambda_ind,:] = np.array((1e-64,0,0,0))
                        continue
                
                model_real = np.real(yuk_ffts[lambda_ind,axis,harm])
                model_imag = np.imag(yuk_ffts[lambda_ind,axis,harm])

                # coefficients for negative log-likelihood quadratic function
                NLL_a = (model_real**2 + model_imag**2)/(2.*var)
                NLL_b = -(data_real*model_real + data_imag*model_imag)/var
                NLL_c = (data_real**2 + data_imag**2)/(2.*var)

                # maximum likelihood estimate for alpha
                alpha_mle = -NLL_b/(2.*NLL_a)

                likelihood_coeffs[harm,axis,lambda_ind,:] = np.array((NLL_a,NLL_b,NLL_c,alpha_mle))

    return likelihood_coeffs


def combine_likelihoods_over_dim(likelihood_coeffs,which='file'):
    '''
    Combine likelihoods along a given dimensions. Likelihoods can be combined over
    files, axes, or harmonics. Corresponding arguments are 'file', 'axis', and 'harm'.
    Returns an array with dimension reduced by 1.
    '''

    # choose the axis to sum over based on the input argument
    axis = 999
    if which=='file':
        axis = len(likelihood_coeffs.shape)-5
    elif which=='harm':
        axis = len(likelihood_coeffs.shape)-4
    elif which=='axis':
        axis = len(likelihood_coeffs.shape)-3
    if axis<0:
        axis = 0

    # check to ensure that the input arguments make sense
    if axis==999:
        raise Exception('Error: argument not recognized. Allowed arguments are '+\
                        '"file", "axis", and "harm".')
    elif len(likelihood_coeffs.shape)+axis<0:
        raise Exception('Error: input likelihood coefficients are too low-dimensional!')

    # for quadratic log-likelihoods, just add coefficients and calculate
    # minimum analytically
    coeffs = np.sum(likelihood_coeffs,axis=axis)
    coeffs[...,-1] = -coeffs[...,1]/(2.*coeffs[...,0])
    return coeffs


def group_likelihoods_by_test(likelihood_coeffs,axis=None):
    '''
    Returns an array of likelihood functions, each of which will independently be used
    to calculate a test statistic. A limit can then be calculated using the sum of the
    resulting test statistics. This is designed based on the 2021 PRD.
    '''

    # final array should have shape (harmonic, lambda, NLL coefficients)
    if axis==None:
        likelihood_coeffs = likelihood_coeffs.reshape((int(likelihood_coeffs.shape[0]*likelihood_coeffs.shape[1]),\
                                                       likelihood_coeffs.shape[2],likelihood_coeffs.shape[3]))
    else:
        likelihood_coeffs = likelihood_coeffs[:,axis,:,:]
    
    return likelihood_coeffs


def test_stat_func(alpha,likelihood_coeffs,discovery=False):
    '''
    Returns the test statistic as a function of alpha given a single set of
    likelihood coefficients. Works for positive or negative alpha depending on
    the sign of the input.
    '''

    # only set a limit on one sign of alpha at a time
    if not all((np.sign(alpha)-np.sign(alpha)[0])==0):
        raise Exception('Error: alphas must all be the same sign!')

    # if alpha is not the same sign as alpha-hat, return zeros
    alpha_hat = likelihood_coeffs[-1]
    if np.sign(alpha[0])!=np.sign(alpha_hat):
        return np.zeros_like(alpha)
    
    # definition of the test statistic
    test_stat = 2.*(quadratic(alpha,*likelihood_coeffs[:-1]) - quadratic(alpha_hat,*likelihood_coeffs[:-1]))

    # if setting a limit, set the test stat to 0 for alpha less than the MLE
    if not discovery:
        test_stat[np.abs(alpha)<np.abs(alpha_hat)] = 0

    return test_stat


def get_limit_analytic(likelihood_coeffs,confidence_level=0.95,alpha_sign=1):
    '''
    For test statistics that are half-quadratics, a limit can be found analytically.
    This will give incorrect results for test statistics calculated by summing
    multiple half-quadratic test statistics with different locations of the minimum.
    '''
    sign = np.float128(alpha_sign)
    con_val = chi2(1).ppf(confidence_level)*0.5

    a,b,_,alpha_hat = likelihood_coeffs
    
    # test statistic is twice the NLL
    a *= 2.
    b *= 2.
    c = -a*alpha_hat**2 - b*alpha_hat - con_val

    # only use likelihoods with alpha hat of the correct sign for the limit
    if alpha_hat*sign<0:
        return 0
    
    # get alpha that solves the half-quadratic equation
    alpha = (-b+sign*np.sqrt(b**2.-4.*a*c))/(2.*a)

    return alpha


def get_limit_from_likelihoods(likelihood_coeffs,confidence_level=0.95,alpha_sign=1):
    '''
    Get an upper limit on alpha at a single lambda given the coefficients
    of the quadratic negative log likleihood function. If multiple sets of likelihood
    coefficients are given, a test statistic is computed for each independently and the
    limit is computed from the sum of test statistics.
    '''
    # critical value of the test statistic ASSUMING WILKS'S THEOREM HOLDS
    con_val = chi2(1).ppf(confidence_level)*0.5

    # can either pass an array of likelihood functions, in which case test statistics will be
    # computed for each and summed, or only a single likelihood function
    if len(likelihood_coeffs.shape)==1:
        likelihood_coeffs = likelihood_coeffs[np.newaxis,:]

    # only valid for cases where alpha_hat is the same sign as alpha_sign
    likelihood_coeffs = likelihood_coeffs[likelihood_coeffs[...,-1]*alpha_sign > 0]
    if len(likelihood_coeffs)==0:
        return np.nan

    # function to be minimized
    limit_func = lambda alpha : np.sum([test_stat_func(alpha,likelihood_coeffs[i,:]) \
                                        for i in range(likelihood_coeffs.shape[0])],axis=0) - con_val

    # to make minimization easier we can approximate the solution with a few
    # iterative searches through log-spaced values of alpha
    alpha_low = 1e0
    alpha_high = 1e15
    sign = np.float128(alpha_sign)
    for i in range(3):
        alphas = sign*np.logspace(np.log10(alpha_low),np.log10(alpha_high),10000)
        test_stats = limit_func(alphas)
        test_stats[test_stats<0] = -1e12
        alpha_ind = np.argmin(np.abs(test_stats))
        alpha_low = sign*alphas[max(alpha_ind - 1,0)]
        alpha_high = sign*alphas[min(alpha_ind + 1,len(alphas)-1)]
        alpha_guess = alphas[alpha_ind]

    # set the bounds so the minimizer does not wander into the region where the
    # test stat is always zero
    alpha_hats = likelihood_coeffs[:,-1]
    min_alpha_hat = alpha_sign*np.amin(alpha_sign*alpha_hats[alpha_hats*alpha_sign>0])
    bounds = [None,None]
    if min_alpha_hat<0:
        bounds[1] = min_alpha_hat
    else:
        bounds[0] = min_alpha_hat

    # now minimize it properly
    result = minimize(limit_func,alpha_guess,bounds=(bounds,),tol=1e-9)

    # if the function value at the fit is too large, return nan 
    if np.abs(result.fun) > 0.5*con_val:
        return np.nan

    return result.x[0]


def get_discovery_from_likelihoods(likelihood_coeffs,z_discovery=3):
    '''
    Report a discovery at some alpha and lambda. When reporting a discovery, likelihood functions
    should be combined and a single test statistic reported for all axes/harmonics, otherwise a
    false discovery will be made if individual harmonics show background-like measurements.
    '''
    # critical value of the test statistic ASSUMING WILKS'S THEOREM HOLDS
    confidence_level = norm.cdf(z_discovery)
    con_val = chi2(1).ppf(confidence_level)

    # can either pass an array of likelihood functions, in which case they will be combined,
    # or just a single likelihood function. Either way a single discovery will be reported.
    while len(likelihood_coeffs.shape)>1:
        likelihood_coeffs = np.sum(likelihood_coeffs,axis=0)
        likelihood_coeffs[...,-1] = -likelihood_coeffs[...,1]/(2.*likelihood_coeffs[...,0])

    # test statistic for the discovery is the likelihood ratio for alpha=0
    test_stat = test_stat_func(np.array([0.]),likelihood_coeffs,discovery=True)[0]

    # if discovery threshold is crossed, a discovery has been made. Report both the alpha MLE
    # and the significance of the discovery
    if test_stat > con_val:
        signif = norm.ppf(test_stat)
        return [likelihood_coeffs[-1],signif]
    else:
        # no discovery, return nan for the discovered alpha and the significance
        return [np.nan,np.nan]


def get_alpha_vs_lambda(likelihood_coeffs,lambdas,discovery=False,\
                        z_discovery=3,confidence_level=0.95):
    '''
    Loop through all lambda values and compute the limit or discovery for each value.
    '''

    if not discovery:
        pos_limit = []
        neg_limit = []
        for i in range(len(lambdas)):
            pos_limit.append(get_limit_from_likelihoods(likelihood_coeffs[...,i,:],\
                                                        confidence_level=confidence_level,\
                                                        alpha_sign=1))
            neg_limit.append(get_limit_from_likelihoods(likelihood_coeffs[...,i,:],\
                                                        confidence_level=confidence_level,\
                                                        alpha_sign=-1))
        pos_limit = np.array(pos_limit)
        neg_limit = -np.array(neg_limit)
        return pos_limit,neg_limit
            
    if discovery:
        disc = []
        for i in range(len(lambdas)):
            disc.append(get_discovery_from_likelihoods(likelihood_coeffs[:,i,:],\
                                                            z_discovery=z_discovery)[0])
        disc = np.array(disc)
        pos_disc = np.ones(len(disc))*np.nan
        neg_disc = np.ones(len(disc))*np.nan
        pos_disc[disc>0] = disc[disc>0]
        neg_disc[disc<0] = np.abs(disc[disc<0])
        return pos_disc,neg_disc


def get_test_stat_dist(likelihood_coeffs,alpha,lamb_ind,num_samples=100,files_per_sample=100,\
                       num_cores=1,mode='prd'):
    '''
    Returns an array of test statistics computed with synthetic datasets. Intended to be used
    to examine the asymptotic behavior of the test statistic for accurate limit setting.
    '''

    sub_inds = np.random.randint(0,files_per_sample,(num_samples,likelihood_coeffs.shape[0]))

    print('Computing the test statistic for {} random subsets of {} files...'\
          .format(num_samples,files_per_sample))
    
    test_stats = Parallel(n_jobs=num_cores)(delayed(compute_test_stat)\
                                                    (likelihood_coeffs[sub_inds[i,...]],np.array([alpha]),lamb_ind,mode)\
                                                     for i in tqdm(range(sub_inds.shape[0])))
    
    return np.array(test_stats)[:,0]


def compute_test_stat(likelihood_coeffs,alpha,lamb_ind,mode='prd'):
    '''
    Compute the test stastistic for a single batch of files for an array of alpha values.
    This uses the test statistic construction from the 2021 PRD.
    '''

    likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='file')
    if mode=='sum':
        likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='axis')
        likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='harm')
        test_stat = test_stat_func(alpha,likelihood_coeffs[lamb_ind,:])
    elif mode=='prd':
        likelihood_coeffs = group_likelihoods_by_test(likelihood_coeffs)
        test_stat = np.sum([test_stat_func(alpha,likelihood_coeffs[j,lamb_ind,:]) \
                            for j in range(likelihood_coeffs.shape[0])],axis=0)
    else:
        raise Exception('Error: mode input not recognized!')
    return test_stat


def get_alpha_scale(likelihood_coeffs,lambdas):
    '''
    Compute the RMS alpha for the dataset over all harmonics. Used as a comparison
    point for background levels between datasets.
    '''
    if len(likelihood_coeffs.shape)>4:
        likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='file')
    ten_um_ind = np.argmin(np.abs(lambdas - 1e-5))
    alpha_hats = likelihood_coeffs[:,:,ten_um_ind,-1]
    median = np.median(np.log10(np.abs(alpha_hats[~np.isnan(alpha_hats)])))
    lower_1s = np.quantile(np.log10(np.abs(alpha_hats[~np.isnan(alpha_hats)])),0.16)
    upper_1s = np.quantile(np.log10(np.abs(alpha_hats[~np.isnan(alpha_hats)])),0.84)

    return 10**np.array((median,lower_1s,upper_1s))


def reshape_nll_args(x,data_shape,alpha,single_beta=False,num_gammas=1,delta_means=[0.1,0.1],\
                     spline=False,axes=['x','y']):
    '''
    Read the arguments from the dictionary and reshape them into the correctly-shaped
    numpy arrays.
    '''

    num_datasets = data_shape[0]

    # choose which axes to include
    first_axis = 0
    second_axis = 2
    if 'x' not in axes:
        first_axis = 1
    if 'y' not in axes:
        second_axis = 1

    # for unconditional mle alpha is the first element and others are offset by 1
    offset = 0
    if alpha is None:
        alpha = x[0]
        offset = 1

    # get number of betas
    num_betas = 1
    if not single_beta:
        num_betas = data_shape[2]

    # extract beta and gamma aarrays from the input vector, constructing the full
    # complex numbers from the real and imaginary parts for the betas
    betas_split = x[offset:offset+2*num_betas]
    beta = betas_split[0::2] + 1j*betas_split[1::2]
    gammas = x[offset+2*num_betas:]

    # reshape array of betas if more than one
    if num_betas>1:
        beta = beta[np.newaxis,np.newaxis,:]

    # reshape array of gammas
    gammas = gammas.reshape(-1,len(axes))

    if spline:
        times = np.arange(0,num_datasets,np.ceil(num_datasets/gammas.shape[0]))
        cs = CubicSpline(times,gammas)
        gammas = cs(np.arange(num_datasets))
        gammas = gammas[...,np.newaxis]
    else:
        gammas = gammas.repeat(np.ceil(num_datasets/gammas.shape[0]),axis=0)[:num_datasets,:,np.newaxis]
    
    # reshape array of deltas
    deltas = np.array(delta_means)[np.newaxis,first_axis:second_axis,np.newaxis]

    return alpha,beta,gammas,deltas


def nll_with_background(x,agg_dict,file_inds=None,alpha=None,lamb=1e-5,single_beta=False,\
                        num_gammas=1,delta_means=[0.1,0.1],spline=False,axes=['x','y']):
    '''
    Implementation of the negative log-likelihood function including both the signal
    model and an in-situ background measurement on some additional data stream.
    '''

    if file_inds is None:
        file_inds = np.array(range(agg_dict['times'].shape[0]))

    # choose which axes to include
    first_axis = 0
    second_axis = 2
    if 'x' not in axes:
        first_axis = 1
    if 'y' not in axes:
        second_axis = 1

    # work with a single value of lambda
    lambdas = agg_dict['template_params'][0]
    lambda_ind = np.argmin(np.abs(lambdas-lamb))
    
    # extract the data for these datasets
    yuk_ffts = agg_dict['template_ffts'][file_inds,lambda_ind,...]
    qpd_ffts = agg_dict['qpd_ffts'][file_inds]
    qpd_sb_ffts = agg_dict['qpd_sb_ffts'][file_inds]
    num_sb = int(qpd_sb_ffts.shape[2]/qpd_ffts.shape[2])

    # get the background measurements
    background = qpd_ffts[:,3,np.newaxis,:]
    background_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,3,:]
    background_var = (1./(2.*num_sb))*np.sum(np.real(background_sb_ffts)**2+np.imag(background_sb_ffts)**2,axis=-1)

    # get the measurements for each file/axis/harm
    data = qpd_ffts[:,first_axis:second_axis,:]
    data_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,first_axis:second_axis,...]
    data_var = (1./(2.*num_sb))*np.sum(np.real(data_sb_ffts)**2+np.imag(data_sb_ffts)**2,axis=-1)

    # get the templates for each file/axis/harm
    signal = yuk_ffts[:,first_axis:second_axis,:]

    # all parameter arrays should be put in the shape (files,axes,harmonics)
    alpha,beta,gammas,deltas = reshape_nll_args(x,data.shape,alpha,single_beta,num_gammas,delta_means,spline,axes)

    # numerators of the Gaussian terms for signal and backround
    num_background = background[:,0,:] - beta[:,0,:]*background[:,0,:] - alpha*np.sum(deltas*signal,axis=1)
    num_signal = data - alpha*signal - beta*gammas*background

    # the background channel NLL is defined for only a single background axis, so has shape (files,harmonics)
    nll_background = np.real(num_background)**2/(2*background_var) \
                   + np.imag(num_background)**2/(2*background_var)

    # the signal channel NLL has a shape before summing of (files,axes,harmonics)
    nll_signal = np.real(num_signal)**2/(2*data_var) \
               + np.imag(num_signal)**2/(2*data_var)
    
    return np.sum(nll_background) + np.sum(nll_signal)


def nll_tester(x,agg_dict,file_inds=None,alpha=None,lamb=1e-5,single_beta=False,\
               num_gammas=1,delta_means=[0.1,0.1],spline=False,axes=['x','y']):
    '''
    Check dimensions of all terms that go into the likelihood function to ensure the
    data is being unpacked correctly.
    '''

    if file_inds is None:
        file_inds = np.array(range(agg_dict['times'].shape[0]))

    # choose which axes to include
    first_axis = 0
    second_axis = 2
    if 'x' not in axes:
        first_axis = 1
    if 'y' not in axes:
        second_axis = 1

    # work with a single value of lambda
    lambdas = agg_dict['template_params'][0]
    lambda_ind = np.argmin(np.abs(lambdas-lamb))
    
    # extract the data for these datasets
    yuk_ffts = agg_dict['template_ffts'][file_inds,lambda_ind,...]
    qpd_ffts = agg_dict['qpd_ffts'][file_inds]
    qpd_sb_ffts = agg_dict['qpd_sb_ffts'][file_inds]
    num_sb = int(qpd_sb_ffts.shape[2]/qpd_ffts.shape[2])

    # get the background measurements
    background = qpd_ffts[:,3,np.newaxis,:]
    background_real = np.real(background)
    background_imag = np.imag(background)
    background_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,3,:]
    background_var = (1./(2.*num_sb))*np.sum(np.real(background_sb_ffts)**2+np.imag(background_sb_ffts)**2,axis=-1)

    # get the measurements for each file/axis/harm
    data = qpd_ffts[:,first_axis:second_axis,:]
    data_real = np.real(data)
    data_imag = np.imag(data)
    data_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,first_axis:second_axis,...]
    data_var = (1./(2.*num_sb))*np.sum(np.real(data_sb_ffts)**2+np.imag(data_sb_ffts)**2,axis=-1)

    # get the templates for each file/axis/harm
    signal_real = np.real(yuk_ffts[:,first_axis:second_axis,:])
    signal_imag = np.imag(yuk_ffts[:,first_axis:second_axis,:])

    # all parameter arrays should be put in the shape (files,axes,harmonics)
    alpha,beta,gammas,deltas = reshape_nll_args(x,data.shape,alpha,single_beta,num_gammas,delta_means,spline,axes)

    print('Shapes of likelihood parameters and data:')
    print('-----------------------------------------')
    print('Alpha:\t\t',alpha.shape)
    print('Beta:\t\t',beta.shape)
    print('Gamma:\t\t',gammas.shape)
    print('Delta:\t\t',deltas.shape)
    print('Signal:\t\t',data_real.shape)
    print('Signal var:\t',data_var.shape)
    print('Background:\t', background_real.shape)
    print('Background var:\t',background_var.shape)
    print('\nValues of likelihood parameters and data:')
    print('-----------------------------------------')
    print('Alpha:\t\t',alpha)
    print('Beta:\t\t',beta)
    print('Gamma:\t\t',gammas[0])
    print('Delta:\t\t',deltas)
    print('Signal:\t\t',data_real[0])
    print('Signal var:\t',data_var[0])
    print('Background:\t', background_real[0])
    print('Background var:\t',background_var[0])


def unconditional_mle(agg_dict,file_inds=None,lamb=1e-5,single_beta=False,\
                      num_gammas=1,delta_means=[0.1,0.1],spline=False,axes=['x','y']):
    '''
    Maximize the likelihood function over both the signal and background
    parameters. Returns the unconditional maximum likelihood estimates for
    alpha and beta.
    '''
    num_betas = 1
    if single_beta==False:
        num_betas = (np.shape(agg_dict['qpd_ffts'])[-1])
    beta_bounds = ((-10,10) for i in range(2*num_betas))
    gamma_bounds = ((-1e3,1e3) for i in range(len(axes)*num_gammas))
    alpha_bound = 1e10*np.exp(2e-5/lamb)
    result = minimize(nll_with_background,[1e9]+[1,0]*num_betas+[1]*len(axes)*num_gammas,\
                      args=(agg_dict,file_inds,None,lamb,single_beta,num_gammas,\
                            delta_means,spline,axes),\
                      bounds=((-alpha_bound,alpha_bound),*beta_bounds,*gamma_bounds),\
                      options={'maxfev':1000000,'xatol':1e-9},method='Nelder-Mead')
    if result.success==False:
        print('Minimization failed!')
    return result
    

def conditional_mle(agg_dict,file_inds=None,alpha=1e8,lamb=1e-5,single_beta=False,\
                    num_gammas=1,delta_means=[0.1,0.1],spline=False,axes=['x','y']):
    '''
    Maximize the likelihood function over both the background only.
    Returns the conditional maximum likelihood estimate for beta.
    '''
    num_betas = 1
    if single_beta==False:
        num_betas = (np.shape(agg_dict['qpd_ffts'])[-1])
    beta_bounds = ((-10,10) for i in range(2*num_betas))
    gamma_bounds = ((-1e3,1e3) for i in range(len(axes)*num_gammas))
    result = minimize(nll_with_background,[1,0]*num_betas+[1]*len(axes)*num_gammas,\
                      args=(agg_dict,file_inds,alpha,lamb,single_beta,num_gammas,\
                            delta_means,spline,axes),\
                      bounds=(*beta_bounds,*gamma_bounds),\
                      options={'maxfev':100000,'xatol':1e-9},method='Nelder-Mead')
    if result.success==False:
        print('Minimization failed!')
    return result


def q_with_background(alpha,uncond_nll,agg_dict,file_inds=None,\
                      lamb=1e-5,**kwargs):
    '''
    Function to compute the profile likelihood ratio test statistic for an exclusion
    for a given value of alpha.
    '''
    cond_res = conditional_mle(agg_dict,file_inds,alpha=alpha,lamb=lamb,**kwargs)
    cond_nll = cond_res.fun
    q_alpha = 2.*(cond_nll - uncond_nll)

    print('Computed q for lambda={:.2e}...'.format(lamb))
    
    return q_alpha


def z_discovery(q_disc):
    '''
    Return a z-score for a discovery.
    '''

    return 2.*norm.ppf(chi2(1).cdf(q_disc))


def plr_limit_func(alpha,alpha_hat,uncond_nll,agg_dict,file_inds=None,\
                   lamb=1e-5,cl=0.95,**kwargs):
    '''
    This function's root occurs at the alpha value corresponding to an exclusion at
    the specified confidence level.
    '''
    con_val = chi2(1).ppf(cl)*0.5
    val = q_with_background(alpha,alpha_hat,uncond_nll,agg_dict,file_inds=file_inds,\
                            lamb=lamb,**kwargs) - con_val
    
    return val


def plr_critical_value(alpha_hat,uncond_nll,agg_dict,file_inds=None,\
                       lamb=1e-5,cl=0.95,tol=1e-1,max_iters=3,**kwargs):
    '''
    Custom root-finding function to get the value of alpha for a given confidence level.
    '''
    alpha_low = alpha_hat
    alpha_high = 10*alpha_hat
    func_val = 2*tol
    iter = 0
    while np.abs(func_val)>tol:
        alpha = (alpha_low + alpha_high)/2.
        func_val = plr_limit_func(alpha,alpha_hat,uncond_nll,agg_dict,file_inds,\
                                  lamb=lamb,cl=cl,**kwargs)
        if func_val<0:
            alpha_low = alpha
        else:
            alpha_high = alpha
        iter += 1
        if iter>max_iters:
            break

    return alpha


def get_limit_with_background(agg_dict,file_inds=None,lamb=1e-5,cl=0.95,tol=1e-1,\
                              max_iters=3,**kwargs):
    '''
    Find the value of alpha that is greater than 95 percent of the test statistic
    distribution under the null hypothesis.
    '''
    uncond_res = unconditional_mle(agg_dict,file_inds=file_inds,lamb=lamb,**kwargs)
    alpha_hat = uncond_res.x[0]
    uncond_nll = uncond_res.fun
    limit = plr_critical_value(alpha_hat,uncond_nll,agg_dict,file_inds,\
                               lamb=lamb,cl=cl,tol=tol,max_iters=max_iters,**kwargs)

    return limit


def get_alpha_vs_lambda_background(agg_dict,file_inds=None,cl=0.95,num_points=5,**kwargs):
    '''
    Get the 95% CL limit on alpha for the range of lambda values in agg_dict.
    '''
    lambdas = agg_dict['template_params'][0]
    print('Computing the limit at each lambda in parallel...')
    limit = Parallel(n_jobs=len(lambdas))(delayed(get_limit_from_q_parabola)\
                                          (agg_dict,file_inds,lamb,cl,num_points,\
                                           False,**kwargs)\
                                          for lamb in tqdm(lambdas))
    
    return np.array(limit)


def q_vs_alpha(alpha_hat,uncond_nll,agg_dict,file_inds=None,lamb=1e-5,num_points=5,**kwargs):
    '''
    Plot the test statistic as a function of alpha.
    '''

    alphas = np.sort(np.append(np.linspace(-alpha_hat,5*alpha_hat,num_points),alpha_hat))
    q_vals = Parallel(n_jobs=len(alphas))(delayed(q_with_background)\
                                        (alpha,uncond_nll,agg_dict,\
                                        file_inds,lamb,**kwargs)\
                                        for alpha in alphas)
    
    # the test statistic should have a minimum at (alpha_hat,0). If it dips negative it's due
    # to errors in the fitting, so set it back to zero
    q_vals = np.array(q_vals) - np.amin(q_vals)
    
    return alphas,q_vals


def get_limit_from_q_parabola(agg_dict,file_inds=None,lamb=1e-5,cl=0.95,num_points=20,\
                              return_q0=True,**kwargs):
    '''
    The minimization is noisy and is susceptible to fluctuations if parameters are not
    chosen carefully. This function instead fits a parabola to the alpha curve and
    determines the limit from the parabola.
    '''

    print('Running job for lambda={:.2e}...'.format(lamb))

    # threshold test statistic for a given confidence level, from Wilks' theorem
    con_val = chi2(1).ppf(cl)*0.5

    # compute the test statistic for a range of alphas
    uncond_res = unconditional_mle(agg_dict,file_inds=file_inds,lamb=lamb,**kwargs)
    alpha_hat = uncond_res.x[0]
    uncond_nll = uncond_res.fun
    alphas,q_vals = q_vs_alpha(alpha_hat,uncond_nll,agg_dict,file_inds,lamb,\
                               num_points,**kwargs)
    
    print('Got q-alpha curve for lambda={:.2e}...'.format(lamb))

    # fit a parabola with a minimum at (alpha_hat,0) which has one free parameter
    # q = a * (alpha - alpha_hat)^2
    # analytic solution to the least-squares fitting
    a_hat = np.sum(q_vals*(alphas-alpha_hat)**2)/np.sum((alphas-alpha_hat)**4)

    # value at which the parabola crosses the CL threshold
    alpha_lim = alpha_hat + np.sign(alpha_hat)*np.sqrt(con_val/a_hat)

    # value of the test statistic at alpha=0
    q_zero = a_hat*alpha_hat**2

    if return_q0:
        return alpha_lim,q_zero
    else:
        return alpha_lim
