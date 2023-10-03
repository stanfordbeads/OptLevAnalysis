import numpy as np
from matplotlib import pyplot as plt

def polar_plots(agg_dict):

    freqs = agg_dict['freqs'][0]
    first_harm = 2
    harms = freqs[agg_dict['good_inds'][0]]
    cmap_polar = plt.get_cmap('inferno') 
    cmap_polar.set_under(color='white')
    rbins = np.logspace(-18,np.log10(1e-15), 30)
    abins = np.linspace(-np.pi,np.pi,60)
    axes = ['X','Y','Z']

    # loop through axes
    for k in range(3):
        fig,axs = plt.subplots(2, 4, figsize=(16,9), subplot_kw={'projection':'polar'})
        # loop through harmonics
        for h,ax in zip(range(len(harms)), axs.flatten()[:-1]):
            ffts = agg_dict['bead_ffts'][:,k,first_harm+h-1]
            _,_,_,im = ax.hist2d(np.angle(ffts),np.abs(ffts),bins=(abins,rbins),cmap=cmap_polar,vmin=1,vmax=50)
            """
            for jj, kk in enumerate([1e7, 1e8, 1e9]):
                ax.plot(np.angle(-1*templates['yukffts'][81, k, h])*np.ones(4),
                [1e-18,1e-17,1e-16, 1e-15], markersize=10, color='xkcd:electric green', ls='dotted', lw=5)
                ax.plot(np.angle(templates['yukffts'][81, k, h])*np.ones(4),
                [1e-18,1e-17,1e-16, 1e-15], markersize=10, color='xkcd:neon blue', ls='dotted', lw=5)
            """
            ax.set_yscale('log')
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_title('{:.0f} Hz'.format(harms[h+first_harm-1]),fontsize=18)
            ax.grid(False)
        axs.flatten()[-1].axis('off')

        y_range_str = '({:.0e},{:.0e})'.format(axs.flatten()[0].get_ylim()[0],axs.flatten()[0].get_ylim()[1])
        textStr = f'20 radial bins log-spaced in '+y_range_str+' N.\n Radial scale is log.\n'
        textStr += f"30 azimuthan bins, linearly spaced.\n"
        textStr += f"Diamonds are for $\\alpha$ ranging \n from $10^7$ to $10^9$.\n"
        textStr += f"Blue dashed line is for $\\alpha >0$.\n"
        textStr += f"Green dashed line is for $\\alpha <0$.\n"
        textStr += f"Faint dashed circles denote mean $\pm 1\sigma$ noise."
        axs.flatten()[-1].text(-0.2,0., textStr, transform = axs.flatten()[-1].transAxes)
        fig.colorbar(im, ax=axs.flatten()[-1], orientation='horizontal', fraction=0.1)
        fig.suptitle(f'Measurements for the '+axes[k]+' axis', fontsize=32)
        
    return fig,axs