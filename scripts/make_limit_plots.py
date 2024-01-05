import data_processing as dp
import plotting as pl
from matplotlib import style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import h5py
from stats import *
import argparse
style.use('optlevstyle.mplstyle')

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
parser.add_argument('descrip',type=str)
parser.add_argument('mode',type=int)
args = parser.parse_args()
path = args.path
descrip = args.descrip
mode = args.mode

# create the empty object and load the file specified
aggdat = dp.AggregateData()
aggdat.load_from_hdf5(path)
print('Computing limits for {} files...'.format(len(aggdat.file_list)))

# mode of combining harmonics and axes.
# 0: same as 2021 PRD
# 1: naive addition of log-likelihoods over all axes and harmonics
# 2: first combine axes, then compute test stat for harmonics independently
mode_titles = ['2021 PRD','likelihood sum','grouped harmonics']

# get likelihood functions for each dataset independently
# then compute limits for each axis and harmonic separately
likelihood_coeffs_full = fit_alpha_all_files(aggdat.agg_dict)
lambdas = aggdat.agg_dict['template_params'][0]
likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs_full,which='file')
alpha_lim_pos = np.zeros_like(likelihood_coeffs[...,0])
alpha_lim_neg = np.zeros_like(likelihood_coeffs[...,0])
for i in range(likelihood_coeffs.shape[0]):
    for j in range(likelihood_coeffs.shape[1]):
        for k in range(likelihood_coeffs.shape[2]):
            alpha_lim_pos[i,j,k] = get_limit_from_likelihoods(likelihood_coeffs=likelihood_coeffs[i,j,k,:],alpha_sign=1)
            alpha_lim_neg[i,j,k] = get_limit_from_likelihoods(likelihood_coeffs=likelihood_coeffs[i,j,k,:],alpha_sign=-1)

print('Combining limits using '+mode_titles[mode]+' method...')

# now compute the overall limit depending on the method of combining harmonics and axes
if mode==0:
    likelihood_coeffs_all = group_likelihoods_by_test(likelihood_coeffs)
    limit_pos,limit_neg = get_alpha_vs_lambda(likelihood_coeffs_all,lambdas)
elif mode==1:
    # likelihood_coeffs_full = likelihood_coeffs_full[:,:,2,:,:]
    while(len(likelihood_coeffs_full.shape)>2):
        likelihood_coeffs_full = np.sum(likelihood_coeffs_full,axis=0)
        likelihood_coeffs_full[...,-1] = -likelihood_coeffs_full[...,1]/(2.*likelihood_coeffs_full[...,0])
    limit_pos = np.zeros(len(lambdas))
    limit_neg = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        limit_pos[i] = get_limit_from_likelihoods(likelihood_coeffs=likelihood_coeffs_full[i,:],alpha_sign=1)
        limit_neg[i] = get_limit_from_likelihoods(likelihood_coeffs=likelihood_coeffs_full[i,:],alpha_sign=-1)
elif mode==2:
    likelihood_coeffs = np.sum(likelihood_coeffs,axis=1)
    likelihood_coeffs[...,-1] = -likelihood_coeffs[...,1]/(2.*likelihood_coeffs[...,0])
    limit_pos,limit_neg = get_alpha_vs_lambda(likelihood_coeffs,lambdas)

print('Making the plots...')

# make the legend entries for the limits from individual harmonics and axes
harm_colors = plt.get_cmap('plasma',alpha_lim_pos.shape[0])
axis_styles = [(0,(5,10)),(0,(5,5)),(0,(5,1)),(0,(3,10,1,10)),(0,(3,5,1,5)),(0,(3,1,1,1))]
lines = []
labels = []
for s in range(len(axis_styles)):
    lines.append(Line2D([0],[0],ls=axis_styles[s],color='k'))
labels = labels + [r'$\alpha>0$, $x$',r'$\alpha>0$, $y$',r'$\alpha>0$, $z$',\
                   r'$\alpha<0$, $x$',r'$\alpha<0$, $y$',r'$\alpha<0$, $z$']
for c in range(alpha_lim_pos.shape[0]):
    lines.append(Line2D([0],[0],color=harm_colors(c)))
    labels.append('{:.0f} Hz'.format(aggdat.agg_dict['freqs'][aggdat.agg_dict['good_inds']][c]))

# plot how the overall limit compares to the individual limits
fig1,ax1 = plt.subplots(figsize=(9,7))
for i in range(alpha_lim_pos.shape[0]):
    for j in range(alpha_lim_pos.shape[1]):
        ax1.loglog(lambdas*1e6,alpha_lim_pos[i,j,:],color=harm_colors(i),ls=axis_styles[j],lw=2)
        ax1.loglog(lambdas*1e6,np.abs(alpha_lim_neg[i,j,:]),color=harm_colors(i),ls=axis_styles[j+3],lw=2)
lp, = ax1.loglog(lambdas*1e6,np.array(limit_pos),ls='--',lw=4,color='b',alpha=0.3)
ln, = ax1.loglog(lambdas*1e6,np.array(limit_neg),ls='-.',lw=4,color='r',alpha=0.3)
labels = labels + [r'$\alpha>0$ combined',r'$\alpha<0$ combined']
lines = lines + [lp,ln]
ax1.set_xlabel(r'$\lambda$ [$\mu$m]',fontsize=18)
ax1.set_ylabel(r'$\alpha$',fontsize=18)
ax1.set_xlim([1e0,1e2])
ax1.set_ylim([1e6,1e12])
ax1.set_title('Limits by axis and harmonic for '+descrip+' data using the '+mode_titles[mode]+' method',fontsize=20)
ax1.grid(which='both')
ax1.legend(handles=lines,labels=labels,ncol=5,fontsize=10,handlelength=4)
fig1.savefig('/home/clarkeh/Figures/limits_by_axis_and_harm_'+\
             mode_titles[mode].replace(' ','_')+'_method_for_'+descrip.replace(' ','_')+'_data.png')

# get the other previously saved limits
lims = h5py.File('/home/clarkeh/limits_all.h5','r')

# plot the overall limit on the same plot as the wilson data and existing best limits
fig2,ax2 = plt.subplots()
colors = style.library['fivethirtyeight']['axes.prop_cycle'].by_key()['color']
ax2.loglog(lambdas*1e6,np.array(limit_pos),ls='-',lw=2,color=colors[0],alpha=0.6,label=r'This result $\alpha>0$')
ax2.loglog(lambdas*1e6,np.array(limit_neg),ls='--',lw=2,color=colors[1],alpha=0.6,label=r'This result $\alpha<0$')
ax2.loglog(np.array(lims['wilson/lambda_pos'])*1e6,np.array(lims['wilson/alpha_pos']),\
           ls='-',lw=2,color=colors[2],alpha=0.6,label=r'Wilson $\alpha>0$')
ax2.loglog(np.array(lims['wilson/lambda_neg'])*1e6,np.array(lims['wilson/alpha_neg']),\
           ls='--',lw=2,color=colors[3],alpha=0.6,label=r'Wilson $\alpha<0$')
ax2.loglog(np.array(lims['best/lambda'])*1e6,np.array(lims['best/alpha']),\
           ls=':',lw=2,color=colors[4],alpha=0.6)
ax2.fill_between(np.array(lims['best/lambda'])*1e6,np.array(lims['best/alpha']),\
                 1e15*np.ones_like(np.array(lims['best/alpha'])),color=colors[4],alpha=0.2)
ax2.set_title('Limits for '+descrip+' data using the '+mode_titles[mode]+' method')
ax2.set_xlabel(r'$\lambda$ [$\mu$m]')
ax2.set_ylabel(r'$\alpha$')
ax2.set_xlim([1e0,1e2])
ax2.set_ylim([1e2,1e12])
ax2.grid(which='both')
ax2.legend(ncol=2)
fig2.savefig('/home/clarkeh/Figures/limits_combined_'+\
             mode_titles[mode].replace(' ','_')+'_method_for_'+descrip.replace(' ','_')+'_data.png')