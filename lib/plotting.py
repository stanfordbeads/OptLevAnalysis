import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import h5py
from funcs import *

def polar_plots(agg_dict,axis_ind=0):
    '''
    Plot 2d polar histograms binning the datasets by the amplitude and phase at a particular
    harmonic along a given axes.
    '''
    freqs = agg_dict['freqs'][0]
    first_harm = 1
    harms = freqs[agg_dict['good_inds'][0]]
    cmap_polar = plt.get_cmap('inferno') 
    cmap_polar.set_under(color='white')
    rbins = np.logspace(-17,np.log10(1e-14), 30)
    abins = np.linspace(-np.pi,np.pi,60)
    axes = ['X','Y','Z']

    fig,axs = plt.subplots(2, 4, figsize=(16,9), subplot_kw={'projection':'polar'})
    # loop through harmonics
    for h,ax in zip(range(len(harms)), axs.flatten()[:-1]):
        ffts = agg_dict['bead_ffts'][:,axis_ind,first_harm+h-1]
        _,_,_,im = ax.hist2d(np.angle(ffts),np.abs(ffts),bins=(abins,rbins),cmap=cmap_polar,vmin=1,vmax=100)
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
    fig.suptitle(f'Measurements for the '+axes[axis_ind]+' axis', fontsize=32)
        
    return fig,axs

def xy_on_qpd(qpd_diag_mat):
    '''
    Plot the x and y eigenmodes of the sphere overlaid on the QPD to see any
    misalignment and non-orthogonality.
    '''

    # calculate QPD coords in bead eignespace
    Q_trans_xy = np.matmul(qpd_diag_mat,np.diag(np.ones(4)))
    Q_trans_xy[:,-2:] = Q_trans_xy[:,:-3:-1]

    # calculate QPD axes in bead eigenspace
    Q_hor_xy = np.matmul(qpd_diag_mat,(1.,1.,-1.,-1.))
    Q_ver_xy = np.matmul(qpd_diag_mat,(1.,-1.,1.,-1.))

    # calculate bead eigenmodes in physical space
    xy_mode_phys = np.array(((qpd_diag_mat[1,0]-qpd_diag_mat[1,1],qpd_diag_mat[0,1]-qpd_diag_mat[0,0]),\
                             (-qpd_diag_mat[1,0]-qpd_diag_mat[1,1],qpd_diag_mat[0,0]+qpd_diag_mat[0,1])))
    x_mode_phys = xy_mode_phys[:,0]*10
    y_mode_phys = xy_mode_phys[:,1]*10

    # create the figure
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plotted axes are bead motion axes
    ax1,ax2 = axs
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-2,2])
    ax1.set_yticks([-2,-1,0,1,2])
    ax1.add_patch(plt.Polygon(Q_trans_xy.T,ls='-',lw=0.5,edgecolor='black',facecolor='lightgray'))
    ax1.axvline(0,color=colors[0],label='Bead eigenmodes')
    ax1.axhline(0,color=colors[0])
    ax1.plot(np.array([-Q_hor_xy[0],0,Q_hor_xy[0]]),np.array([-Q_hor_xy[1],0,Q_hor_xy[1]]),color=colors[1],label='QPD axes')
    ax1.plot(np.array([-Q_ver_xy[0],0,Q_ver_xy[0]]),np.array([-Q_ver_xy[1],0,Q_ver_xy[1]]),color=colors[1])
    ax1.set_title('Bead eigenspace')
    ax1.set_xlabel('Bead $x$ [arb]')
    ax1.set_ylabel('Bead $y$ [arb]')
    ax1.legend()
    ax1.set_aspect('equal')

    # Plotted axes are QPD space axes
    ax2.set_xlim([-2,2])
    ax2.set_ylim([-2,2])
    ax2.set_yticks([-2,-1,0,1,2])
    ax2.add_patch(plt.Polygon(np.array(((-1,-1,1,1),(-1,1,1,-1))).T,ls='-',lw=0.5,edgecolor='black',facecolor='lightgray'))
    ax2.set_title('Physical space')
    ax2.plot([-x_mode_phys[0],0,x_mode_phys[0]],[-x_mode_phys[1],0,x_mode_phys[1]],color=colors[0],label='Bead eigenmodes')
    ax2.plot([-y_mode_phys[0],0,y_mode_phys[0]],[-y_mode_phys[1],0,y_mode_phys[1]],color=colors[0])
    ax2.axvline(0,color=colors[1],label='QPD axes')
    ax2.axhline(0,color=colors[1])
    ax2.set_xlabel('QPD horizontal axis')
    ax2.set_ylabel('QPD vertical axis')
    ax2.legend()
    ax2.set_aspect('equal')

    return fig,axs

def cross_coupling(agg_dict,qpd_diag_mat,p_x=None,p_y=None,plot_inds=None):
    '''
    Plot the cross-coupling before and after diagonalization.
    '''
    if plot_inds is None:
        plot_inds = np.array(range(len(agg_dict['freqs'][0])))

    # get the raw QPD data to plot
    raw_qpd_1 = agg_dict['quad_amps'][plot_inds][:,0,:]
    raw_qpd_2 = agg_dict['quad_amps'][plot_inds][:,1,:]
    raw_qpd_3 = agg_dict['quad_amps'][plot_inds][:,2,:]
    raw_qpd_4 = agg_dict['quad_amps'][plot_inds][:,3,:]
    tot_vs_time = np.sum(agg_dict['quad_amps'][plot_inds][:,:4,:],axis=1)
    raw_qpd_1 = raw_qpd_1/tot_vs_time
    raw_qpd_2 = raw_qpd_2/tot_vs_time
    raw_qpd_3 = raw_qpd_3/tot_vs_time
    raw_qpd_4 = raw_qpd_4/tot_vs_time

    freqs = agg_dict['freqs'][0]
    nsamp = raw_qpd_1.shape[1]

    signal_mat = np.array((raw_qpd_1,raw_qpd_2,raw_qpd_3,raw_qpd_4))

    naive_mat = np.array(((1.,1.,-1.,-1.),\
                          (1.,-1.,1.,-1.)))

    # do the transformations from quadrants to x and y
    new_resp = np.einsum('ij,jkl->ikl',qpd_diag_mat,signal_mat)
    raw_resp = np.einsum('ij,jkl->ikl',naive_mat,signal_mat)

    # pick out the new x and y coordinates
    x_raw = raw_resp[0,:,:]
    y_raw = raw_resp[1,:,:]
    x_corr = new_resp[0,:,:]
    y_corr = new_resp[1,:,:]

    fft_x_raw = np.sqrt(np.mean(np.abs(np.fft.rfft(x_raw)*2./nsamp)**2,axis=0))
    fft_y_raw = np.sqrt(np.mean(np.abs(np.fft.rfft(y_raw)*2./nsamp)**2,axis=0))
    fft_x_corr = np.sqrt(np.mean(np.abs(np.fft.rfft(x_corr)*2./nsamp)**2,axis=0))
    fft_y_corr = np.sqrt(np.mean(np.abs(np.fft.rfft(y_corr)*2./nsamp)**2,axis=0))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig,ax = plt.subplots()
    ax.semilogy(freqs,fft_x_raw,label='Naive $x$',color=colors[0],alpha=0.5)
    ax.semilogy(freqs,fft_y_raw,label='Naive $y$',color=colors[1],alpha=0.5)
    ax.semilogy(freqs,fft_x_corr,label='Diag. $x$',color=colors[0],ls=':',lw=2)
    ax.semilogy(freqs,fft_y_corr,label='Diag. $y$',color=colors[1],ls=':',lw=2)
    if (p_x is not None) and (p_y is not None):
        ax.semilogy(freqs,lor(freqs,*p_x),label='Peak fit $x$',color=colors[2])
        ax.semilogy(freqs,lor(freqs,*p_y),label='Peak fit $y$',color=colors[3])
    ax.set_xlim([280,420])
    ax.set_ylim([1e-6,1e-2])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('ASD [1/$\sqrt{\mathrm{Hz}}$]')
    ax.set_title('QPD Diagonalization')
    ax.grid(which='both')
    ax.legend()

    return fig,ax


def transfer_funcs(path,sensor='QPD',nsamp=50000,fsamp=5000):
    '''
    Plot the transfer functions and the fits for all axes.
    '''

    # set some basic parameters
    freqs = np.fft.rfftfreq(int(nsamp),d=1/fsamp)
    cal_drive_freq = 71.
    drive_freq_ind = np.argmin(np.abs(freqs-cal_drive_freq))
    axes = ['x','y','z']

    # get the transfer function file with fits and raw data
    tf_file = h5py.File(path)
    tf_data = tf_file['measurements/'+sensor+'/TF'][:,:,:]
    tf_freqs = tf_file['measurements/'+sensor+'/ff'][:,:,:]

    # extract the transfer function matrix
    Harr = np.zeros((len(freqs), 3, 3), dtype=complex)
    Harr[:,0,0] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zXX'], tf_file['fits/'+sensor+'/pXX'], \
                                        tf_file['fits/'+sensor+'/kXX']/tf_file.attrs['scaleFactors'][0], 2*np.pi*freqs)[1]
    Harr[:,1,1] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zYY'], tf_file['fits/'+sensor+'/pYY'], \
                                        tf_file['fits/'+sensor+'/kYY']/tf_file.attrs['scaleFactors'][1], 2*np.pi*freqs)[1]
    Harr[:,2,2] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zZZ'], tf_file['fits/'+sensor+'/pZZ'], \
                                        tf_file['fits/'+sensor+'/kZZ']/tf_file.attrs['scaleFactors'][2], 2*np.pi*freqs)[1]
    
    # get force calibration factors
    force_cal_factors = []
    for i in range(3):
        force_cal_factors.append(np.abs(Harr[drive_freq_ind,i,i]))
    force_cal_factors = np.array(force_cal_factors)

    # get indices of frequencies driven during TF measurement
    tf_freq_inds = np.zeros_like(tf_freqs)
    for i in range(tf_freqs.shape[0]):
        for j in range(tf_freqs.shape[1]):
            for k in range(tf_freqs.shape[2]):
                tf_freq_inds[i,j,k] = np.argmin(np.abs(freqs-tf_freqs[i,j,k]))

    # plot the result
    rows = 3
    if sensor=='PSPD':
        rows = 2
    fig,ax = plt.subplots(rows,3,figsize=(9,2*rows+1),sharex=True,sharey=True)
    fig.suptitle('Transfer Functions for the '+sensor)
    for i in range(rows):
        ax[i,0].set_ylabel('Mag [N/N]')
        for j in range(3):
            ax[0,j].set_title('Drive $'+axes[j]+'$')
            ax[-1,j].set_xlabel('Frequency [Hz]')
            ax[i,j].loglog(tf_freqs[i,j],np.abs(tf_data[i,j]),linestyle='none',marker='o',ms=4,alpha=0.5,label='Measurement')
            ax[i,j].loglog(freqs[tf_freq_inds[i,j]],force_cal_factors[i]*np.ones(len(tf_data[i,j]))\
                           /np.abs(Harr[:,i,j][tf_freq_inds[i,j]]),lw=2,label='Fit')
            ax[i,j].grid(which='both')
        ax[i,i].legend(fontsize=10)

    return fig,ax