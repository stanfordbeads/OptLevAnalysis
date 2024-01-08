import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2,norm
from joblib import Parallel, delayed
from tqdm import tqdm
from funcs import *

# ************************************************************************ #
# This file contains the functions used for statistical analysis of
# AggregateData objects.
# ************************************************************************ #

def fit_alpha_all_files(agg_dict,file_indices=None,sensor='qpd',use_ml=False,num_cores=1):
    '''
    Find the best fit alpha to a dataset. Most of this should eventually be moved to
    stats.py so that different likelihood functions can be called from here.
    '''

    if file_indices is None:
        file_indices = np.array(range(agg_dict['times'].shape[0]))

    print('Computing the likelihood functions for the specified files...')
    likelihood_coeffs = Parallel(n_jobs=num_cores)(delayed(fit_alpha_for_file)\
                                                   (agg_dict,ind,sensor,use_ml)\
                                                    for i,ind in enumerate(tqdm(file_indices)))

    return np.array(likelihood_coeffs)


def fit_alpha_for_file(agg_dict,file_index,sensor='qpd',use_ml=False):
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
    mot_likes = agg_dict['mot_likes'][file_index][:,good_inds]
    likelihood_coeffs = np.zeros((len(good_inds),3,len(lambdas),4),dtype=np.float128)
    
    for harm,_ in enumerate(good_inds):
        for axis in [0,1,2]:

            sb_fft_array = bead_sb_ffts[axis,harm*num_sb:(harm+1)*num_sb]
            data = bead_ffts[axis,harm]
            if use_ml and axis!=2:
                ml = mot_likes[axis,harm]
                noise = np.mean(np.abs(sb_fft_array)**2)
                data = np.sqrt(ml*max(0,np.abs(data)**2-noise) + noise)*np.exp(1j*np.angle(data))
            data_real = np.real(data)
            data_imag = np.imag(data)
            var = (1./(2.*num_sb))*np.sum(np.real(sb_fft_array)**2+np.imag(sb_fft_array)**2)

            for lambda_ind,_ in enumerate(lambdas):
                
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
    # first approx of true test stat asymptotic behavior
    con_val = 10.

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
    # ***** CHECK IF FACTOR 2 IS REQUIRED HERE *****
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
            pos_limit.append(get_limit_from_likelihoods(likelihood_coeffs[:,i,:],\
                                                        confidence_level=confidence_level,\
                                                        alpha_sign=1))
            neg_limit.append(get_limit_from_likelihoods(likelihood_coeffs[:,i,:],\
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
