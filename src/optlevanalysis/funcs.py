import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as sig

'''
Miscellaneous functions used for data reduction/calibration.
'''

def damped_osc_amp(f, A, f0, g):
    '''Fitting function for AMPLITUDE of a damped harmonic oscillator
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor
                   c, constant offset, default 0

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    gamma = 2. * np.pi * g
    denom = np.sqrt( (w0**2 - w**2)**2 + w**2 * gamma**2 )
    return A / denom




def damped_osc_phase(f, A, f0, g, phase0 = 0.):
    '''Fitting function for PHASE of a damped harmonic oscillator.
       Includes an arbitrary DC phase to fit over out of phase responses
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    gamma = 2. * np.pi * g
    # return A * np.arctan2(-w * g, w0**2 - w**2) + phase0
    return 1.0 * np.arctan2(-w * gamma, w0**2 - w**2) + phase0



def make_extrapolator(interpfunc, xs=[], ys=[], pts=(10, 10), order=(1,1), \
                      arb_power_law=(False, False), semilogx=False):
    '''Make a functional object that does nth order polynomial extrapolation
       of a scipy.interpolate.interp1d object (should also work for other 1d
       interpolating objects).

           INPUTS: interpfunc, inteprolating function to extrapolate
                   pts, points to include in linear regression
                   order, order of the polynomial to use in extrapolation
                   inverse, boolean specifying whether to use inverse poly
                            1st index for lower range, 2nd for upper

           OUTPUTS: extrapfunc, function object with extrapolation
    '''
    
    if not len(xs) or not len(ys):
        try:
            xs = interpfunc.x
            ys = interpfunc.y
        except:
            print('Need to provide data, or interpolating function needs to contain\
                   the data as class attributes (such as interp1d objectes)')
            return

    if arb_power_law[0]:
        xx = xs[:pts[0]]
        yy = ys[:pts[0]]
        meanx = np.mean(xx)
        meany = np.mean(yy)

        if semilogx:
            popt_l, _ = curve_fit(line, np.log10(xx), yy)

            p0_l = [meany / np.log10(meanx)]
            def fit_func_l(x, c):
                return popt_l[0] * np.log10(x) + c

        else:
            popt_l, _ = curve_fit(line, np.log10(xx), np.log10(yy))

            p0_l = [meany / (meanx**popt_l[0])]
            def fit_func_l(x, a):
                return a * x**popt_l[0]


        popt2_l, _ = curve_fit(fit_func_l, xx, yy, maxfev=100000, p0=p0_l)
        lower = lambda x: fit_func_l(x, popt2_l[0])

        # lower_params = ipolyfit(xs[:pts[0]], ys[:pts[0]], order[0])
        # lower = ipoly1d(lower_params)       
    else:
        lower_params = np.polyfit(xs[:pts[0]], ys[:pts[0]], order[0])
        lower = np.poly1d(lower_params)

    if arb_power_law[1]:
        xx = xs[-pts[1]:]
        yy = ys[-pts[1]:]
        meanx = np.mean(xx)
        meany = np.mean(yy)

        if semilogx:
            popt_u, _ = curve_fit(line, np.log10(xx), yy)

            p0_u = [meany / np.log10(meanx)]
            def fit_func_u(x, c):
                return popt_u[0] * np.log10(x) + c

        else:
            popt_u, _ = curve_fit(line, np.log10(xx), np.log10(yy))

            p0_u = [meany / (meanx**popt_u[0])]
            def fit_func_u(x, a):
                return a * x**popt_u[0]

        popt2_u, _ = curve_fit(fit_func_u, xx, yy, maxfev=100000, p0=p0_u)
        upper = lambda x: fit_func_u(x, popt2_u[0])
        # upper_params = ipolyfit(xs[-pts[1]:], ys[-pts[1]:], order[1])
        # upper = ipoly1d(upper_params) 
    else:
        upper_params = np.polyfit(xs[-pts[1]:], ys[-pts[1]:], order[1])
        upper = np.poly1d(upper_params) 

    def extrapfunc(x):

        ubool = x > xs[-1]
        lbool = x < xs[0]

        midval = interpfunc( x[ np.invert(ubool + lbool) ] )
        uval = upper( x[ubool] )
        lval = lower( x[lbool] )

        return np.concatenate((lval, midval, uval))

    return extrapfunc

def line(x, a, b):
    return a * x + b

def get_file_number(file_name):
    '''
    Returns the file number for sorting.
    '''
    return int((file_name.split('.h5')[-2]).split('_')[-1])

def lor(x,x_0,gamma,A):
    '''
    Lorentzian function for fitting peaks.
    '''
    return A*gamma/((x-x_0)**2.+0.25*gamma**2.)

def quadratic(x, a, b, c):
    '''
    Quadratic function in standard form
    '''
    return a*x**2 + b*x + c

def rayleigh(spectra):
    '''
    Compute the Rayleigh spectrum for a time series of data.
    '''
    stds = np.std(spectra,axis=0)
    means = np.mean(spectra,axis=0)

    return stds/means

def noise_inds(agg_dict):
    '''
    Return the indices corresponding to noise-only files in the agg_dict.
    '''
    return np.array([i[0] for i in np.argwhere(agg_dict['is_noise'])])

def shaking_inds(agg_dict):
    '''
    Return the indices corresponding to files for which the attractor was shaking.
    '''
    return np.array([i[0] for i in np.argwhere(~agg_dict['is_noise'])])

def gv_decimate(x, q, LPF, axis=-1):
    '''
    Implements same functionality as Scipy's decimate,
    but with SOS filtering.
    '''
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(None, None, q)
    y = sig.sosfiltfilt(LPF, x)
    return y[tuple(sl)]