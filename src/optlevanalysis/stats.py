import numpy as np
from scipy.optimize import minimize
from iminuit import Minuit
from scipy.stats import chi2, norm
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import warnings
from optlevanalysis.funcs import *

# ************************************************************************ #
# This file contains the functions used for statistical analysis of
# AggregateData objects.
# ************************************************************************ #

def fit_alpha_all_files(agg_dict,file_indices=None,sensor='qpd',use_sidebands=False,num_cores=1):
    """Finds the best fit alpha over multiple files.

    :param agg_dict: The dictionary from the `AggregateData` object containing the data for multiple files.
    :type agg_dict: dict
    :param file_indices: Indices of the files within the dictionary to include in the fit.
    :type file_indices: array-like or None
    :param sensor: Which of the position sensors to use (either qpd or xypd).
    :type sensor: str
    :param use_sidebands: Whether or not to estimate the noise from the sidebands. If False, noise-only files will be used.
    :type use_sidebands: bool
    :param num_cores: Number of cores to use when parallelizing the fitting.
    :type num_cores: int
    :return: An array containing the coefficients of the quadratic negative log-likelihood function.
    :rtype: numpy.ndarray
    """

    if file_indices is None:
        file_indices = np.array(range(agg_dict['times'].shape[0]))

    # compute limits for only the shaking files
    shaking_indices = file_indices[~agg_dict['is_noise'][file_indices]]

    if not use_sidebands:
        noise_indices = file_indices[agg_dict['is_noise'][file_indices]]
        next_true_positions = np.searchsorted(noise_indices, shaking_indices)
        valid_mask = next_true_positions < len(noise_indices)
        shaking_indices = shaking_indices[valid_mask]
        noise_indices = noise_indices[next_true_positions[valid_mask]]
    else:
        noise_indices = [None]*len(file_indices)

    print('Computing the signal-only likelihood functions for the specified files...')
    likelihood_coeffs = Parallel(n_jobs=num_cores)(delayed(fit_alpha_for_file)\
                                                   (agg_dict,ind,use_sidebands,noise_indices[i],sensor)\
                                                    for i,ind in enumerate(tqdm(shaking_indices)))

    return np.array(likelihood_coeffs)


def fit_alpha_for_file(agg_dict,file_index,use_sidebands=False,noise_index=None,sensor='qpd'):
    """Finds the best fit alpha for a single file.

    Since the likelihood function for a single file is a Gaussian, the negative log-likelihood is parabolic, and can be fully
    described by the coefficients a, b, and c, where NLL=a*alpha^2 + b*alpha + c. These coefficients and the MLE for alpha,
    called alpha_hat (which is equal to -b/2/a), are put into the last dimension of an array as (a, b, c, alpha_hat). The
    complete array has dimensions (file, harmonic, axis, lambda, NLL coefficient).

    :param agg_dict: The dictionary from the `AggregateData` object containing the data.
    :type agg_dict: dict
    :param file_index: Index of the file containing the data to be fitted.
    :type file_index: int
    :param use_sidebands: Whether or not to estimate the noise from the sidebands. If False, the specified noise-only file will be used.
    :type use_sidebands: bool
    :param noise_index: The index of the file containing the noise-only data to be used to estimate the uncertainty.
    :type noise_index: int or None
    :param sensor: Which of the position sensors to use (either qpd or xypd).
    :type sensor: str
    :return: An array containing the coefficients of the quadratic negative log-likelihood function.
    :rtype: numpy.ndarray
    """

    if not use_sidebands and noise_index is None:
        raise Exception('Error: if not using sidebands, a noise file index must be provided!')

    lambdas = agg_dict['template_params'][file_index]
    yuk_ffts = agg_dict['template_ffts'][file_index]
    bead_ffts = agg_dict[sensor+'_ffts'][file_index]
    if use_sidebands:
        bead_sb_ffts = agg_dict[sensor+'_sb_ffts'][file_index]
        num_sb = int(bead_sb_ffts.shape[1]/bead_ffts.shape[1])
    else:
        noise_ffts = agg_dict[sensor+'_ffts'][noise_index]
    good_inds = agg_dict['good_inds']
    likelihood_coeffs = np.zeros((len(good_inds),3,len(lambdas),4),dtype=np.float128)
    
    for harm,_ in enumerate(good_inds):
        for axis in [0,1,2]:
            data = bead_ffts[axis,harm]
            if use_sidebands:
                sb_fft_array = bead_sb_ffts[axis,harm*num_sb:(harm+1)*num_sb]
                var = (1./(2.*num_sb))*np.sum(np.real(sb_fft_array)**2+np.imag(sb_fft_array)**2)
            else:
                var = np.abs(noise_ffts[axis,harm])**2
            data_real = np.real(data)
            data_imag = np.imag(data)

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


def combine_likelihoods_over_dim(likelihood_coeffs, which='file'):
    """Combines the negative log-likelihood functions over a given dimension by
    summing the NLL coefficients and recalculating the best fit alpha.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param which: The dimension along which to combine the likelihood functions. Can be file, axis, harm, or all.
    :type which: str
    :return: The new array of NLL coefficients with the dimensionality reduced.
    :rtype: numpy.ndarray
    """

    # choose the axis to sum over based on the input argument
    axis = 999
    if which=='file':
        axis = len(likelihood_coeffs.shape)-5
    elif which=='harm':
        axis = len(likelihood_coeffs.shape)-4
    elif which=='axis':
        axis = len(likelihood_coeffs.shape)-3
    elif which=='all':
        axis = tuple(i for i in range(len(likelihood_coeffs.shape)-2))
    if isinstance(axis, int):
        if axis < 0:
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
    """Returns an array of likelihood functions, each of which will independently be used
    to calculate a test statistic. A limit can then be calculated using the sum of the
    resulting test statistics. This is designed based on the 2021 PRD.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param axis: The axis for which the test statistics should be computed.
    :type axis: int
    :return: The new array of NLL coefficients with the proper shape for individual test statistics to be computed.
    :rtype: numpy.ndarray
    """

    # final array should have shape (harmonic, lambda, NLL coefficients)
    if axis==None:
        likelihood_coeffs = likelihood_coeffs.reshape((int(likelihood_coeffs.shape[0]*likelihood_coeffs.shape[1]),\
                                                       likelihood_coeffs.shape[2],likelihood_coeffs.shape[3]))
    else:
        likelihood_coeffs = likelihood_coeffs[:,axis,:,:]
    
    return likelihood_coeffs


def group_likelihoods_by_alpha_sign(likelihood_coeffs):
    """
    Split the NLL coefficients for different harmonics up by the sign of the best fit alpha,
    then combine each of them over the selected harmonics.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :return: Tuple of the two new arrays of NLL coefficients for positive and negative alpha, respectively.
    :rtype: tuple of numpy.ndarray
    """

    # input shape should be (harmonics, lambdas, NLL coefficients)
    if len(likelihood_coeffs.shape)!=3:
        print('Error: wrong shape! Ensure the likelihood coefficients have been properly combined first.')

    # sum the likelihood functions over harmonics, including only those with the correct sign
    pos_inds = (likelihood_coeffs[:,:,-1]>=0)[...,np.newaxis].repeat(4,axis=-1)
    neg_inds = (likelihood_coeffs[:,:,-1]<0)[...,np.newaxis].repeat(4,axis=-1)
    coeffs_pos = np.sum(likelihood_coeffs*pos_inds,axis=0)
    coeffs_pos[...,-1] = -coeffs_pos[...,1]/(2.*coeffs_pos[...,0])
    coeffs_neg = np.sum(likelihood_coeffs*neg_inds,axis=0)
    coeffs_neg[...,-1] = -coeffs_neg[...,1]/(2.*coeffs_neg[...,0])

    return coeffs_pos, coeffs_neg


def group_likelihood_by_strongest_n_harms(likelihood_coeffs,agg_dict,num_harms):
    """
    Returns an array of likelihood coefficients with the dimension along the harmonics reduced
    to num_harms, where the remaining harmonics are the ones with the greatest signal power.
    This does the same thing as group_likelihoods_by_parameter with the which='signal' argument,
    except it may give a different set of harmonics for each lambda value.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param agg_dict: The dictionary from the `AggregateData` object containing the data.
    :type agg_dict: dict
    :param num_harms: The number of harmonics to use
    :type num_harms: int
    :return: A new array of NLL coefficients
    :rtype: numpy.ndarray
    """

    if len(likelihood_coeffs.shape)<3:
        print('Error: the likelihood coefficients array has the wrong shape!')
        return

    # compute the signal power at each of the harmonics, adding axes in quadrature
    signal_strength = np.sqrt(np.mean(np.sum(np.abs(agg_dict['template_ffts'])**2,axis=-2),axis=0))

    # create a new array for the likelihood coeffs including only the strongest harmonics
    shape = np.array(np.shape(likelihood_coeffs))
    harm_ax = int(len(shape)>4)
    shape[harm_ax] = num_harms
    likelihood_coeffs_new = np.zeros(shape)
    
    # loop through the lambdas and for each, keep only the strongest n harmonics
    harm_inds = []
    for i in range(signal_strength.shape[0]):
        harm_inds = np.argsort(signal_strength[i,:])[-num_harms:]
        likelihood_coeffs_new[...,i,:] = np.take(likelihood_coeffs[...,i,:],harm_inds,axis=harm_ax)

    return likelihood_coeffs_new


def _fit_all_harm_combinations(likelihood_coeffs, num_harms=6, template_ffts=None, \
                               num_cores=1):
    """Do the fits over all combinations of some number of harmonics and return the chi-squared and
    fraction of signal power contained in that combination.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param num_harms: Number of harmonics to include in each combination.
    :type num_harms: int
    :param template_ffts: Array of FFTs of the signal templates, used to calculate signal power fractions.
    :type template_ffts: numpy.ndarray or None
    :param num_cores: Number of cores to use for parallel processing.
    :type num_cores: int
    :return: Tuple containing arrays of chi-squared values, alpha values, signal power fractions, and harmonic combinations.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    if len(likelihood_coeffs.shape)!=3:
        raise Exception('Error: NLL coefficients must be reduced to the form (harmonic, lambda, parameter)!')

    harm_combs = np.array(list(itertools.combinations(np.arange(likelihood_coeffs.shape[0]),num_harms)))
    
    def process_combination(harm_comb):
        these_coeffs = combine_likelihoods_over_dim(likelihood_coeffs[harm_comb,...], which='harm')
        these_coeffs[...,-1] = -these_coeffs[...,1]/(2.*these_coeffs[...,0])
        alpha = these_coeffs[:,-1]
        chi2 = np.array([quadratic(these_coeffs[j,-1], *these_coeffs[j,:3]) \
                         for j in range(likelihood_coeffs.shape[-2])])
        
        if template_ffts is not None:
            signal_frac = np.array([np.sqrt(np.sum(np.abs(template_ffts[:,j,:,harm_comb])**2)/\
                                            np.sum(np.abs(template_ffts[:,j,...])**2)) \
                                    for j in range(likelihood_coeffs.shape[-2])])
        else:
            signal_frac = np.zeros_like(chi2)
            
        return chi2, alpha, signal_frac
    
    print('Checking fits for all combinations of {} harmonics...'.format(num_harms))
    results = Parallel(n_jobs=num_cores)(delayed(process_combination)(harm_comb) \
                                         for harm_comb in tqdm(harm_combs))
    
    chi2_dist = np.array([r[0] for r in results])
    alpha_dist = np.array([r[1] for r in results])
    signal_frac = np.array([r[2] for r in results])
        
    return chi2_dist, alpha_dist, signal_frac, harm_combs


def _get_grouped_harms(param_dist, alpha_dist, harm_combs):
    """Given the distributions of chi-squared, alpha-hat, and signal power for all combinations
    of some number of harmonics, returns the harmonic indices corresponding to the combination
    that appears most signal-like for both positive and negative alpha.

    :param param_dist: Distribution of parameters (chi-squared or signal power) for each harmonic combination.
    :type param_dist: numpy.ndarray
    :param alpha_dist: Distribution of alpha values for each harmonic combination.
    :type alpha_dist: numpy.ndarray
    :param harm_combs: Array containing all possible combinations of harmonics.
    :type harm_combs: numpy.ndarray
    :return: Tuple containing arrays of harmonic indices for positive and negative alpha.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)
    """
    param_pos_alpha = np.copy(param_dist)
    negative_cols = np.any(alpha_dist<0, axis=1)
    positive_cols = np.any(alpha_dist>0, axis=1)
    param_pos_alpha[negative_cols, :] = 1e18
    param_neg_alpha = np.copy(param_dist)
    param_neg_alpha[positive_cols, :] = 1e18
    pos_comb_inds = np.argmin(param_pos_alpha, axis=0)
    neg_comb_inds = np.argmin(param_neg_alpha, axis=0)
    pos_inds = harm_combs[pos_comb_inds]
    neg_inds = harm_combs[neg_comb_inds]

    # make sure we're using the same set of harmonics across all lambdas
    rows, counts = np.unique(neg_inds, axis=0, return_counts=True)
    neg_inds = np.tile(rows[np.argmax(counts)], (neg_inds.shape[0], 1))
    rows, counts = np.unique(pos_inds, axis=0, return_counts=True)
    pos_inds = np.tile(rows[np.argmax(counts)], (pos_inds.shape[0], 1))

    return pos_inds, neg_inds


def group_likelihoods_by_parameter(likelihood_coeffs, num_harms=6, which='chi2', \
                                  template_ffts=None, num_cores=1):
    """Reduce the likelihoods array to only contain the harmonics that look most signal-like
    in both positive and negative alpha. The harmonics are selected based on either minimizing
    chi-squared or maximizing signal power.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param num_harms: Number of harmonics to include in each combination.
    :type num_harms: int
    :param which: Parameter to use for selecting harmonics ('chi2' or 'signal').
    :type which: str
    :param template_ffts: Array of FFTs of the signal templates, required if which='signal'.
    :type template_ffts: numpy.ndarray or None
    :param num_cores: Number of cores to use for parallel processing.
    :type num_cores: int
    :return: Tuple containing arrays of NLL coefficients for positive and negative alpha harmonics.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)
    """
    outputs = _fit_all_harm_combinations(likelihood_coeffs, num_harms, template_ffts, num_cores)
    chi2_dist, alpha_dist, signal_frac, harm_combs = outputs
    if which=='chi2':
        pos_inds, neg_inds = _get_grouped_harms(chi2_dist, alpha_dist, harm_combs)
    elif which=='signal':
        if template_ffts is None:
            raise Exception('Error: signal templates not provided!')
        # the combination which minimized the parameters will be used, so pass -signal_frac
        pos_inds, neg_inds = _get_grouped_harms(-signal_frac, alpha_dist, harm_combs)
    pos_harm_idx = pos_inds.T[:, None]
    neg_harm_idx = neg_inds.T[:, None]
    lambda_idx = np.arange(likelihood_coeffs.shape[1])[None, :]

    pos_harm_coeffs = combine_likelihoods_over_dim(likelihood_coeffs[pos_harm_idx, lambda_idx], which='all')
    neg_harm_coeffs = combine_likelihoods_over_dim(likelihood_coeffs[neg_harm_idx, lambda_idx], which='all')

    return pos_harm_coeffs, neg_harm_coeffs


def add_systematic_uncertainty(likelihood_coeffs, sigma_sys):
    """Broaden the negative log-likelihood parabolas to account for systematic uncertainties
    by modifying the coefficients of the quadratic function. The systematic uncertainty is 
    applied as a fractional uncertainty on alpha.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param sigma_sys: The fractional systematic uncertainty to apply.
    :type sigma_sys: float
    :return: Array of modified NLL coefficients with systematic uncertainties included.
    :rtype: numpy.ndarray
    """
    var_sys = (sigma_sys*likelihood_coeffs[...,-1])**2
    a_new = (likelihood_coeffs[...,0]**-1 + var_sys)**-1
    b_new = likelihood_coeffs[...,1]*a_new/likelihood_coeffs[...,0]
    c_new = likelihood_coeffs[...,2]*a_new/likelihood_coeffs[...,0]
    likelihood_coeffs_new = np.array((a_new, b_new, c_new, likelihood_coeffs[...,-1]))
    likelihood_coeffs_new = np.moveaxis(likelihood_coeffs_new, 0, -1)

    return likelihood_coeffs_new


def test_stat_func(alpha, likelihood_coeffs, discovery=False):
    """Returns the test statistic as a function of alpha given a single set of
    likelihood coefficients. Works for positive or negative alpha depending on
    the sign of the input.

    :param alpha: The alpha value(s) at which to compute the test statistic.
    :type alpha: numpy.ndarray or float
    :param likelihood_coeffs: Array of coefficients describing the negative log-likelihood function.
    :type likelihood_coeffs: numpy.ndarray
    :param discovery: If True, compute discovery test statistic. If False, compute limit test statistic.
    :type discovery: bool
    :return: Array of test statistic values corresponding to the input alpha values.
    :rtype: numpy.ndarray
    
    :raises Exception: If input alpha values have different signs.
    """

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


def get_limit_analytic(likelihood_coeffs, confidence_level=0.95, alpha_sign=1):
    """For test statistics that are half-quadratics, a limit can be found analytically.
    This will give incorrect results for test statistics calculated by summing
    multiple half-quadratic test statistics with different locations of the minimum.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param confidence_level: The desired confidence level for the limit (between 0 and 1).
    :type confidence_level: float
    :param alpha_sign: The sign of alpha for which to compute the limit (+1 or -1).
    :type alpha_sign: int
    :return: The analytic limit on alpha at the specified confidence level.
    :rtype: float

    :note: This method is only valid for single half-quadratic test statistics.
    """
    sign = np.float128(alpha_sign)
    con_val = norm.ppf(confidence_level)**2

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


def get_limit_from_likelihoods(likelihood_coeffs, confidence_level=0.95, alpha_sign=1):
    """Get an upper limit on alpha at a single lambda given the coefficients
    of the quadratic negative log likelihood function. If multiple sets of likelihood
    coefficients are given, a test statistic is computed for each independently and the
    limit is computed from the sum of test statistics.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param confidence_level: The desired confidence level for the limit (between 0 and 1).
    :type confidence_level: float
    :param alpha_sign: The sign of alpha for which to compute the limit (+1 or -1).
    :type alpha_sign: int
    :return: The upper limit on alpha at the specified confidence level.
    :rtype: float or numpy.nan

    :note: Returns numpy.nan if no valid limit can be found for the given likelihood coefficients.
    """
    # critical value of the test statistic ASSUMING WILKS'S THEOREM HOLDS
    con_val = norm.ppf(confidence_level)**2

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


def get_discovery_from_likelihoods(likelihood_coeffs, z_discovery=3):
    """Report a discovery at some alpha and lambda. When reporting a discovery, likelihood functions
    should be combined and a single test statistic reported for all axes/harmonics, otherwise a
    false discovery will be made if individual harmonics show background-like measurements.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param z_discovery: The number of sigma required for a discovery claim.
    :type z_discovery: float
    :return: List containing [alpha_MLE, significance]. Both values are nan if no discovery is made.
    :rtype: list of float or numpy.nan

    :note: Likelihood functions should be combined before calling this function to avoid false discoveries.
    """
    # critical value of the test statistic ASSUMING WILKS'S THEOREM HOLDS
    confidence_level = norm.cdf(z_discovery)
    con_val = norm.ppf(confidence_level)**2

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


def get_alpha_vs_lambda(likelihood_coeffs, lambdas, discovery=False,\
                        z_discovery=3, confidence_level=0.95, analytic=False):
    """Loop through all lambda values and compute the limit or discovery for each value.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param lambdas: Array of lambda values to evaluate.
    :type lambdas: numpy.ndarray
    :param discovery: If True, compute discovery values. If False, compute limits.
    :type discovery: bool
    :param z_discovery: The number of sigma required for a discovery claim (used if discovery=True).
    :type z_discovery: float
    :param confidence_level: The desired confidence level for limits (used if discovery=False).
    :type confidence_level: float
    :param analytic: If True, use analytic limit calculation. If False, use numerical method.
    :type analytic: bool
    :return: Tuple containing arrays of positive and negative limits/discoveries for each lambda value.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)
    """
    limit_func = get_limit_analytic if analytic else get_limit_from_likelihoods

    if not discovery:
        pos_limit = []
        neg_limit = []
        for i in range(len(lambdas)):
            pos_limit.append(limit_func(likelihood_coeffs[...,i,:],\
                                        confidence_level=confidence_level,\
                                        alpha_sign=1))
            neg_limit.append(limit_func(likelihood_coeffs[...,i,:],\
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
    

def get_abs_alpha_vs_lambda(likelihood_coeffs,lambdas,discovery=False,\
                            z_discovery=3,confidence_level=0.95):
    """Set a limit on the absolute value of alpha by computing independent limits based on
    combining harmonics of the same sign, then taking the largest of the limits on each sign.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param lambdas: Array of lambda values to evaluate.
    :type lambdas: numpy.ndarray
    :param discovery: If True, compute discovery values. If False, compute limits.
    :type discovery: bool
    :param z_discovery: The number of sigma required for a discovery claim (used if discovery=True).
    :type z_discovery: float
    :param confidence_level: The desired confidence level for limits (used if discovery=False).
    :type confidence_level: float
    :return: Array of limits/discoveries on the absolute value of alpha for each lambda value.
    :rtype: numpy.ndarray
    """

    # likelihood coefficients should have already be combined over everything but harmonics,
    # and should therefore have shape (harmonics, lambdas, NLL coefficients)
    coeffs_pos,coeffs_neg = group_likelihoods_by_alpha_sign(likelihood_coeffs)
    lim_pos,_ = get_alpha_vs_lambda(coeffs_pos,lambdas,discovery=discovery,\
                                    z_discovery=z_discovery,confidence_level=confidence_level)
    _,lim_neg = get_alpha_vs_lambda(coeffs_neg,lambdas,discovery=discovery,\
                                    z_discovery=z_discovery,confidence_level=confidence_level)
    
    limit = np.amax(np.vstack((np.abs(lim_pos),np.abs(lim_neg))),axis=0)

    return limit


def get_test_stat_dist(likelihood_coeffs,alpha,lamb_ind,num_samples=100,files_per_sample=100,\
                       num_cores=1,mode='prd'):
    """Returns an array of test statistics computed with synthetic datasets. Intended to be used
    to examine the asymptotic behavior of the test statistic for accurate limit setting.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param alpha: The alpha value at which to compute the test statistics.
    :type alpha: float
    :param lamb_ind: Index of the lambda value to use.
    :type lamb_ind: int
    :param num_samples: Number of synthetic datasets to generate.
    :type num_samples: int
    :param files_per_sample: Number of files to include in each synthetic dataset.
    :type files_per_sample: int
    :param num_cores: Number of cores to use for parallel processing.
    :type num_cores: int
    :param mode: Test statistic computation mode ('prd' or 'sum').
    :type mode: str
    :return: Array of test statistic values for each synthetic dataset.
    :rtype: numpy.ndarray
    """

    sub_inds = np.random.randint(0,files_per_sample,(num_samples,likelihood_coeffs.shape[0]))

    print('Computing the test statistic for {} random subsets of {} files...'\
          .format(num_samples,files_per_sample))
    
    test_stats = Parallel(n_jobs=num_cores)(delayed(compute_test_stat)\
                                                    (likelihood_coeffs[sub_inds[i,...]],np.array([alpha]),lamb_ind,mode)\
                                                     for i in tqdm(range(sub_inds.shape[0])))
    
    return np.array(test_stats)[:,0]


def compute_test_stat(likelihood_coeffs,alpha,lamb_ind,mode='prd'):
    """Compute the test statistic for a single batch of files for an array of alpha values.
    This uses the test statistic construction from the 2021 PRD.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param alpha: The alpha value(s) at which to compute the test statistic.
    :type alpha: numpy.ndarray
    :param lamb_ind: Index of the lambda value to use.
    :type lamb_ind: int
    :param mode: Test statistic computation mode ('prd' for PRD method or 'sum' for simple sum).
    :type mode: str
    :return: Array of test statistic values for the given alpha values.
    :rtype: numpy.ndarray
    
    :raises Exception: If mode input is not recognized.
    """

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
    """Compute the RMS alpha for the dataset over all harmonics. Used as a comparison
    point for background levels between datasets.

    :param likelihood_coeffs: The array of negative log-likelihood coefficients.
    :type likelihood_coeffs: numpy.ndarray
    :param lambdas: Array of lambda values to evaluate.
    :type lambdas: numpy.ndarray
    :return: Array containing [median, lower_1sigma, upper_1sigma] of log10 of absolute alpha values.
    :rtype: numpy.ndarray
    """
    if len(likelihood_coeffs.shape)>4:
        likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='file')
    ten_um_ind = np.argmin(np.abs(lambdas - 1e-5))
    alpha_hats = likelihood_coeffs[:,:,ten_um_ind,-1]
    median = np.median(np.log10(np.abs(alpha_hats[~np.isnan(alpha_hats)])))
    lower_1s = np.quantile(np.log10(np.abs(alpha_hats[~np.isnan(alpha_hats)])),0.16)
    upper_1s = np.quantile(np.log10(np.abs(alpha_hats[~np.isnan(alpha_hats)])),0.84)

    return 10**np.array((median,lower_1s,upper_1s))


# **************************************************************************** #
# Everything above here is for setting limits with signal-only likelihoods
# functions. Everything below here is for setting limits with the likelihood
# function which includes the in-situ measurement of scattered-light backgrounds.
# As of November 2024, this is an experimental method which has not been sufficiently
# well-validated to justify use in physics analysis.
# **************************************************************************** #


def reshape_nll_args(x,data_shape,num_gammas=1,delta_means=[0.1,0.1],\
                     spline=False,axes=['x','y'],harms=[]):
    """Reads the arguments from the dictionary and reshapes them into the correctly-shaped
    numpy arrays.
    """

    num_datasets = data_shape[0]

    # choose which axes to include
    first_axis = 0
    second_axis = 2
    if 'x' not in axes:
        first_axis = 1
    if 'y' not in axes:
        second_axis = 1

    # unpack the parameters    
    alpha = x[0]
    deltas = x[1:3]
    gammas = x[3:-4]
    taus = np.array(x[-4::2]) + np.array(1j*x[-3::2])

    # reshape array of gammas
    gammas = gammas.reshape(-1,len(axes),data_shape[2])

    if spline:
        times = np.arange(0,num_datasets,np.ceil(num_datasets/gammas.shape[0]))
        cs = CubicSpline(times,gammas)
        gammas = cs(np.arange(num_datasets))
    else:
        gammas = gammas.repeat(np.ceil(num_datasets/gammas.shape[0]),axis=0)[:num_datasets,:]
    
    # reshape array of deltas
    deltas = np.array(deltas)[np.newaxis,first_axis:second_axis,np.newaxis]

    # reshape array of taus
    taus = np.array(taus)[np.newaxis,:,np.newaxis]

    return alpha,gammas,deltas,taus


def nll_with_background(agg_dict,file_inds=None,lamb=1e-5,num_gammas=1,delta_means=[0.1,0.1],\
                        phi_sigma=10.,spline=False,axes=['x','y'],harms=[],signal_only=False):
    """Implementation of the negative log-likelihood function including both the signal
    model and an in-situ background measurement on some additional data stream. Extracts
    the data from the agg_dict and reshapes it. Returns a function that takes as an argument
    only the 1-dimensional vector of parameters for the NLL to be minimized over.
    """

    if file_inds is None:
        file_inds = np.array(range(agg_dict['times'].shape[0]))

    # choose which axes to include
    first_axis = 0
    second_axis = 2
    if 'x' not in axes:
        first_axis = 1
    if 'y' not in axes:
        second_axis = 1

    # choose which harmonics to include
    if harms==[]:
        harm_inds = np.array(range(len(agg_dict['good_inds'])))
    else:
        harms_full = agg_dict['freqs'][agg_dict['good_inds']]
        harm_inds = [np.argmin(np.abs(3*h - harms_full)) for h in harms]

    # work with a single value of lambda
    lambdas = agg_dict['template_params'][0]
    lambda_ind = np.argmin(np.abs(lambdas-lamb))
    
    # extract the data for these datasets
    yuk_ffts = agg_dict['template_ffts'][file_inds,lambda_ind,...]
    qpd_ffts = agg_dict['qpd_ffts'][file_inds]
    qpd_sb_ffts = agg_dict['qpd_sb_ffts'][file_inds]
    num_sb = int(qpd_sb_ffts.shape[2]/qpd_ffts.shape[2])

    # get the background measurements
    background = qpd_ffts[:,np.newaxis,3,harm_inds]
    background_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,3,harm_inds]
    background_var = (1./(2.*num_sb))*np.sum(np.real(background_sb_ffts)**2+np.imag(background_sb_ffts)**2,axis=-1)

    # get the measurements for each file/axis/harm
    data = qpd_ffts[:,first_axis:second_axis,harm_inds]
    data_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)\
                   [:,first_axis:second_axis,harm_inds]
    data_var = (1./(2.*num_sb))*np.sum(np.real(data_sb_ffts)**2+np.imag(data_sb_ffts)**2,axis=-1)

    # get the templates for each file/axis/harm
    signal = yuk_ffts[:,first_axis:second_axis,harm_inds]

    # get the angle between the x and y axes
    psi = agg_dict['axis_angles'][file_inds,np.newaxis]

    # convert the input argument into radians
    phi_sigma = phi_sigma*np.pi/180.

    def nll_func(x):
        """Function to be minimized.
        """

        # all parameter arrays should be put in the shape (files,axes,harmonics)
        alpha,gammas,deltas,taus = reshape_nll_args(x,data.shape,num_gammas,delta_means,spline,axes,harms)
        
        # Gaussians term for the real and imaginary parts of the signal
        num_signal = data - alpha*taus*signal - gammas*(background - alpha*np.sum(deltas*taus*signal,axis=1)[:,np.newaxis,:])
        den_signal = 2*(data_var + background_var[:,np.newaxis,:])
        nll_signal = np.real(num_signal)**2/den_signal + np.imag(num_signal)**2/den_signal

        # if including the background term, calculate the angles describing the background vector
        if not signal_only:
            thetas = np.arctan((gammas[:,0,:] + gammas[:,1,:]/np.tan(psi))\
                                /(gammas[:,1,:] + gammas[:,0,:]/np.tan(psi)))[:,np.newaxis,:]
            phis = np.arctan(np.sum(gammas,axis=1)/((np.sin(thetas[:,0,:]) + np.cos(thetas[:,0,:]))\
                                                    *(1. - 1./np.tan(psi))))[:,np.newaxis,:]
        else:
            thetas = np.zeros_like(gammas)
            phis = np.zeros_like(gammas)                                         
        
        # constraints on the nuisance parameters
        tau_sigma = 0.1
        nll_deltas = (deltas*np.ones_like(gammas) - np.array(delta_means)[np.newaxis,:,np.newaxis])**2 \
                     /(2*np.array(delta_means)[np.newaxis,:,np.newaxis]**2)
        nll_phis = (phis*np.ones_like(gammas) - np.mean(phis,axis=-1,keepdims=True))**2/(2*phi_sigma**2)
        nll_taus = np.real(taus*np.ones_like(gammas) - 1.)**2/(2.*tau_sigma**2) + np.imag(taus*np.ones_like(gammas))**2/(2.*tau_sigma**2)
        nll_nuisance = nll_deltas + nll_phis + nll_taus + np.log(np.abs(alpha))/1e5
        
        return np.sum(nll_signal) + np.sum(nll_nuisance)
    
    return nll_func


def minimize_nll(agg_dict,file_inds=None,lamb=1e-5,num_gammas=1,\
                 delta_means=[0.1,0.1],phi_sigma=10.,spline=False,axes=['x','y'],harms=[],\
                 signal_only=False,background_only=False,alpha_guess=1e9,num_attempts=2):
    """Maximizes the likelihood function over both the signal and background
    parameters. Returns the unconditional maximum likelihood estimates for
    alpha and beta.
    """

    # the guess may be a single element or an array
    if np.shape(alpha_guess)!=():
        alpha_guess = alpha_guess[np.argmin(np.abs(agg_dict['template_params'][0]-lamb))]

    # bounds on the parameters
    if harms==[]:
        num_harms = (np.shape(agg_dict['qpd_ffts'])[-1])
    else:
        num_harms = len(harms)
    alpha_bound = 1e10*np.exp(2e-5/lamb)
    delta_bounds = ((-5*np.abs(delta_means[i]),5*np.abs(delta_means[i])) for i in range(2))
    gamma_bounds = ((-1e1,1e1) for i in range(len(axes)*num_gammas*num_harms))
    tau_bounds = ((0.1,2),(-1,1),(0.1,2),(-1,1))

    # create the function to be minimized
    nll_func = nll_with_background(agg_dict,file_inds,lamb,num_gammas,\
                                   delta_means,phi_sigma,spline,axes,harms,signal_only)
    
    # create the Minuit object
    harm_freqs = ['{:.0f}Hz'.format(f) for f in agg_dict['freqs'][agg_dict['good_inds']]]
    names = ['alpha'] + ['delta_'+axes[i] for i in range(2)] + \
            ['gamma_'+str(i+1)+'_'+axes[k]+'_'+harm_freqs[j] for i in range(num_gammas) \
             for k in range(len(axes)) for j in range(num_harms)] + ['tau_'+j+axes[i] for i in range(2) for j in ['re_','im_']]
    m = Minuit(nll_func,np.array([alpha_guess]+delta_means+[0.1]*len(axes)*num_gammas*num_harms+[1.,1.,0.,0.],dtype=np.float128),name=names)
    m.strategy = 2
    m.errordef = 0.5
    m.limits = [(-alpha_bound,alpha_bound)] + list(delta_bounds) + list(gamma_bounds) + list(tau_bounds)

    if signal_only:
        m.fixto(list(range(1,len(m.values)-2)),\
                list(np.zeros(len(m.values)-3)))
    if background_only:
        m.fixto([0,1,2],[0,0,0])

    # minimize the function
    i = 0
    while (i<num_attempts) and (not (m.valid and m.accurate)):
        m.simplex()
        m.migrad()
        if (i<num_attempts-1) and signal_only==False:
            m.values = np.array(m.values)*np.random.normal(1,0.1,len(m.values))
        i += 1

    if not (m.valid and m.accurate):
        print('Minimization for lambda={:.2e} failed!'.format(lamb))

    # # ensure the minimizer has found the true minimum by scanning
    # m.scan().migrad()

    return m


def profile_likelihood_limit(agg_dict,file_inds=None,lamb=1e-5,cl=0.95,num_points=20,\
                             use_parab=False,**kwargs):
    """Uses the profile likelihood ratio to set an exclusion limit on alpha at the desired
    confidence level.
    """

    print('Running unconditional NLL minimization for lambda={:.2e}...'.format(lamb))

    # threshold test statistic for a given confidence level, from Wilks' theorem
    con_val = chi2(1).ppf(cl)*0.5

    # create the minuit object and get the MLE for alpha
    minuit_obj = minimize_nll(agg_dict,file_inds,lamb=lamb,**kwargs)
    alpha_hat = minuit_obj.values[0]

    # profile the test statistic over alpha
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        alphas,qvals,successes = minuit_obj.mnprofile(vname=0,size=num_points,bound=2,subtract_min=True)

    if use_parab:
        # fit a parabola with a minimum at (alpha_hat,0) which has one free parameter
        # q = a * (alpha - alpha_hat)^2
        # analytic solution to the least-squares fitting
        a_hat = np.sum(2.*qvals*(alphas-alpha_hat)**2)/np.sum((alphas-alpha_hat)**4)

        # value at which the parabola crosses the CL threshold
        alpha_lim = alpha_hat + np.sign(alpha_hat)*np.sqrt(con_val/a_hat)
    else:
        # we want an upper limit, so ensure we select the correct side of the profile
        alphas_above = alphas[np.sign(alpha_hat)*alphas >= np.sign(alpha_hat)*alpha_hat]
        qvals_above = qvals[np.sign(alpha_hat)*alphas >= np.sign(alpha_hat)*alpha_hat]

        # get the limit on alpha from the profile likelihood
        alpha_lim = alphas_above[np.argmin(np.abs(2*qvals_above - con_val))]

    print('Finished computing the limit for lambda={:.2e}'.format(lamb))

    return alpha_lim


def profile_likelihood_abs_limit(agg_dict,pos_harms,neg_harms,file_inds=None,lamb=1e-5,cl=0.95,\
                                 num_points=20,use_parab=False,**kwargs):
    """Uses the profile likelihood ratio to set an exclusion limit on the absolute value ofalpha
    at the desired confidence level. Done by splitting harmonics up by the sign of alpha-hat
    and using only those with like sign to set the limit on the corresponding sign of alpha.
    The limit on the absolue value of alpha is then the maximum of the two limits.
    """

    harms_list = [pos_harms,neg_harms]
    alpha_lims = []
    for i in range(2):
        print('Running unconditional NLL minimization for lambda={:.2e}...'.format(lamb))

        # threshold test statistic for a given confidence level, from Wilks' theorem
        con_val = chi2(1).ppf(cl)*0.5

        # create the minuit object and get the MLE for alpha
        minuit_obj = minimize_nll(agg_dict,file_inds,lamb=lamb,harms=harms_list[i],**kwargs)
        alpha_hat = minuit_obj.values[0]

        # profile the test statistic over alpha
        alphas,qvals,successes = minuit_obj.mnprofile(vname=0,size=num_points,bound=2,subtract_min=True)

        if use_parab:
            # fit a parabola with a minimum at (alpha_hat,0) which has one free parameter
            # q = a * (alpha - alpha_hat)^2
            # analytic solution to the least-squares fitting
            a_hat = np.sum(2.*qvals*(alphas-alpha_hat)**2)/np.sum((alphas-alpha_hat)**4)

            # value at which the parabola crosses the CL threshold
            alpha_lims.append(alpha_hat + np.sign(alpha_hat)*np.sqrt(con_val/a_hat))
        else:
            # we want an upper limit, so ensure we select the correct side of the profile
            alphas_above = alphas[np.sign(alpha_hat)*alphas >= np.sign(alpha_hat)*alpha_hat]
            qvals_above = qvals[np.sign(alpha_hat)*alphas >= np.sign(alpha_hat)*alpha_hat]

            # get the limit on alpha from the profile likelihood
            alpha_lims.append(alphas_above[np.argmin(np.abs(2*qvals_above - con_val))])

    print('Finished computing the limit for lambda={:.2e}'.format(lamb))

    return np.amax(np.abs(alpha_lims))


def get_alpha_vs_lambda_background(agg_dict,file_inds=None,cl=0.95,num_points=20,\
                                   use_parab=False,harm_list=[],**kwargs):
    """Gets the 95% CL limit on alpha for the range of lambda values in agg_dict.
    """

    lambdas = agg_dict['template_params'][0]

    # if no list of harmonics is given, use all harmonics for each lambda
    if harm_list==[]:
        harm_list = [[] for i in range(len(lambdas))]
    
    print('Computing the limit at each lambda in parallel...')
    limit = Parallel(n_jobs=len(lambdas))(delayed(profile_likelihood_limit)\
                                          (agg_dict,file_inds,lamb,cl,num_points,\
                                           use_parab=use_parab,harms=harm_list[i],**kwargs)\
                                          for i,lamb in enumerate(tqdm(lambdas)))
    
    return np.array(limit)


def get_abs_alpha_vs_lambda_background(agg_dict,pos_harms,neg_harms,file_inds=None,\
                                       cl=0.95,num_points=20,use_parab=False,**kwargs):
    """Sets the limit on the absolute value of alpha for the range of lambdas, where harmonics
    with positive alpha-hat are used for the positive alpha limit, harmonics with negative
    alpha-hat are used for the negative alpha limit, and the limit on the absolute value is
    the greater of the absolute values of the two. The arguments pos_harms and neg_harms can
    be obtained from the function split_harms_by_alpha_sign.
    """

    harm_lists = [pos_harms,neg_harms]
    limits = []
    for i in range(2):
        limits.append(get_alpha_vs_lambda_background(agg_dict,file_inds,cl=cl,\
                                                     num_points=num_points,use_parab=use_parab,\
                                                     harm_list=harm_lists[i],**kwargs))
    
    return np.amax(np.vstack(np.abs(limits)),axis=0)

 
def split_harms_by_alpha_sign(agg_dict,num_cores=39,axes=['x','y']):
    """Returns two lists with the same length as the agg_dict's array of
    lambda values. Each contains arrays of the harmonics of a given sign
    for the corresponding value of lambda.
    """

    # choose which axes to include
    first_axis = 0
    second_axis = 2
    if 'x' not in axes:
        first_axis = 1
    if 'y' not in axes:
        second_axis = 1
    
    # use the signal-only model to determine whether each harmonic
    # has a positive or negative alpha-hat
    lambdas = agg_dict['template_params'][0]
    likelihood_coeffs = fit_alpha_all_files(agg_dict,num_cores=num_cores)
    likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='file')
    likelihood_coeffs = np.sum(likelihood_coeffs[:,first_axis:second_axis,...],axis=1)
    likelihood_coeffs[...,-1] = -likelihood_coeffs[...,1]/(2.*likelihood_coeffs[...,0])
    alpha_hats = likelihood_coeffs[:,:,-1]
    pos_harms = [np.array(agg_dict['freqs'][agg_dict['good_inds']]\
                [np.argwhere((alpha_hats>=0)[:,i])[:,0]]/3,dtype=int) for i in range(len(lambdas))]
    neg_harms = [np.array(agg_dict['freqs'][agg_dict['good_inds']]\
                [np.argwhere((alpha_hats<0)[:,i])[:,0]]/3,dtype=int) for i in range(len(lambdas))]
    
    return pos_harms,neg_harms
