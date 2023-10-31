import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from datetime import datetime
from scipy import signal
import h5py
from funcs import *
from stats import *
np.seterr(all='ignore')


def polar_plots(agg_dict,indices=None,axis_ind=0,sensor='qpd',\
                amp_bins=np.logspace(-17,-15, 30),phase_bins=np.linspace(-np.pi,np.pi,60),\
                vmax=50,plot_templates=True,alphas=np.array((1e8,1e9,1e10))):
    '''
    Plot 2d polar histograms binning the datasets by the amplitude and phase at a particular
    harmonic along a given axes.
    '''
    if indices is None:
        indices = np.array(range(agg_dict['qpd_ffts'].shape[0]))
    freqs = agg_dict['freqs']
    harms = freqs[agg_dict['good_inds']]
    cmap_polar = plt.get_cmap('inferno') 
    cmap_polar.set_under(color='white')
    axes = ['X','Y','Z']

    fig,axs = plt.subplots(2, 4, figsize=(16,9), subplot_kw={'projection':'polar'})
    # loop through harmonics
    for h,ax in zip(range(len(harms)), axs.flatten()[:-1]):
        ffts = agg_dict[sensor+'_ffts'][indices][:,axis_ind,h]
        sens_title = sensor.upper()
        _,_,_,im = ax.hist2d(np.angle(ffts),np.abs(ffts),bins=(phase_bins,amp_bins),cmap=cmap_polar,vmin=1,vmax=vmax)
        if plot_templates:
            yuk_ffts = agg_dict['template_ffts'][indices][h]
            lambdas = agg_dict['template_params'][indices][h]
            ten_um_ind = np.argmin(np.abs(lambdas-1e-5))
            ax.plot(np.angle(-1*yuk_ffts[ten_um_ind,axis_ind,h])*np.ones(4),\
                    [1e-18,1e-17,1e-16, 1e-15], markersize=10, color='xkcd:electric green', ls='dotted', lw=5)
            ax.plot(np.angle(-1*np.ones(len(alphas))*yuk_ffts[ten_um_ind,axis_ind,h]),\
                    np.abs(yuk_ffts[ten_um_ind,axis_ind,h])*alphas,\
                    color='xkcd:electric green',ls='none',marker='d',ms=10)
            ax.plot(np.angle(yuk_ffts[ten_um_ind,axis_ind,h])*np.ones(4),\
                    [1e-18,1e-17,1e-16, 1e-15], markersize=10, color='xkcd:neon blue', ls='dotted', lw=5)
            ax.plot(np.angle(np.ones(len(alphas))*yuk_ffts[ten_um_ind,axis_ind,h]),\
                    np.abs(yuk_ffts[ten_um_ind,axis_ind,h])*alphas,\
                    color='xkcd:neon blue',ls='none',marker='d',ms=10)
        ax.set_yscale('log')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks(np.array([0,90,180,270])*np.pi/180.)
        ax.set_thetalim(0,2*np.pi)
        ax.set_title('{:.0f} Hz'.format(harms[h]),fontsize=18)
        ax.grid(False)
    axs.flatten()[-1].axis('off')

    y_range_str = '({:.0e},{:.0e})'.format(axs.flatten()[0].get_ylim()[0],axs.flatten()[0].get_ylim()[1])
    alpha_min = str(int(np.log10(min(alphas))))
    alpha_max = str(int(np.log10(max(alphas))))
    textStr = '{} radial bins log-spaced in '.format(len(amp_bins))+y_range_str+' N.\n Radial scale is log.\n'
    textStr += '{} azimuthan bins, linearly spaced.\n'.format(len(phase_bins))
    textStr += 'Diamonds are for $\\alpha$ ranging \n from $10^{'+alpha_min+'}$ to $10^{'+alpha_max+'}$.\n'
    textStr += 'Blue dashed line is for $\\alpha >0$.\n'
    textStr += 'Green dashed line is for $\\alpha <0$.\n'
    #textStr += f"Faint dashed circles denote mean $\pm 1\sigma$ noise."
    axs.flatten()[-1].text(-0.2,0., textStr, transform = axs.flatten()[-1].transAxes)
    fig.colorbar(im, ax=axs.flatten()[-1], orientation='horizontal', fraction=0.1)
    fig.suptitle(sens_title+' measurements for the '+axes[axis_ind]+' axis', fontsize=32)

    # for some reason memory is not released and in subsequent function calls this can cause errors
    del freqs,harms,ffts,lambdas,yuk_ffts
    
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

    freqs = agg_dict['freqs']
    fsamp = agg_dict['fsamp']
    nsamp = agg_dict['nsamp']
    fft_to_asd = np.sqrt(nsamp/2./fsamp)

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

    asd_x_raw = np.sqrt(np.mean(np.abs(np.fft.rfft(x_raw)*fft_to_asd)**2,axis=0))
    asd_y_raw = np.sqrt(np.mean(np.abs(np.fft.rfft(y_raw)*fft_to_asd)**2,axis=0))
    asd_x_corr = np.sqrt(np.mean(np.abs(np.fft.rfft(x_corr)*fft_to_asd)**2,axis=0))
    asd_y_corr = np.sqrt(np.mean(np.abs(np.fft.rfft(y_corr)*fft_to_asd)**2,axis=0))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig,ax = plt.subplots()
    ax.semilogy(freqs,asd_x_raw,label='Naive $x$',color=colors[0],alpha=0.5)
    ax.semilogy(freqs,asd_y_raw,label='Naive $y$',color=colors[1],alpha=0.5)
    ax.semilogy(freqs,asd_x_corr,label='Diag. $x$',color=colors[0],ls=':',lw=2)
    ax.semilogy(freqs,asd_y_corr,label='Diag. $y$',color=colors[1],ls=':',lw=2)
    if (p_x is not None) and (p_y is not None):
        ax.semilogy(freqs,lor(freqs,*p_x),label='Peak fit $x$',color=colors[2])
        ax.semilogy(freqs,lor(freqs,*p_y),label='Peak fit $y$',color=colors[3])
    ax.set_xlim([280,420])
    ax.set_ylim([1e-2,1e2])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('ASD [1/$\sqrt{\mathrm{Hz}}$]')
    ax.set_title('QPD Diagonalization')
    ax.grid(which='both')
    ax.legend()

    return fig,ax


def transfer_funcs(path,sensor='QPD',phase=False,nsamp=50000,fsamp=5000):
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
                                        tf_file['fits/'+sensor+'/kXX']/tf_file.attrs['scaleFactors_'+sensor][0], 2*np.pi*freqs)[1]
    Harr[:,1,1] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zYY'], tf_file['fits/'+sensor+'/pYY'], \
                                        tf_file['fits/'+sensor+'/kYY']/tf_file.attrs['scaleFactors_'+sensor][1], 2*np.pi*freqs)[1]
    Harr[:,2,2] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zZZ'], tf_file['fits/'+sensor+'/pZZ'], \
                                        tf_file['fits/'+sensor+'/kZZ']/tf_file.attrs['scaleFactors_'+sensor][2], 2*np.pi*freqs)[1]
    
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
    title = 'Magnitudes'
    if phase:
        title = 'Phases'
    fig,ax = plt.subplots(rows,3,figsize=(9,2*rows+1),sharex=True,sharey=True)
    fig.suptitle('Transfer Function '+title+' for the '+sensor)
    for i in range(rows):
        if phase:
            ax[i,0].set_ylabel('Phase [$^\circ$]')
            ax[i,0].set_ylim([-200,200])
            ax[i,0].set_yticks([-180,0,180])
        else:
            ax[i,0].set_ylabel('Mag [N/N]')
        for j in range(3):
            ax[0,j].set_title('Drive $'+axes[j]+'$')
            ax[-1,j].set_xlabel('Frequency [Hz]')
            if phase:
                ax[i,j].semilogx(tf_freqs[i,j],np.angle(tf_data[i,j])*180./np.pi,linestyle='none',\
                                                        marker='o',ms=4,alpha=0.5,label='Measurement')
                if i==j:
                    ax[i,j].semilogx(freqs[tf_freq_inds[i,j]],np.angle(np.ones(len(tf_data[i,j]))\
                                     /Harr[:,i,j][tf_freq_inds[i,j]])*180./np.pi,lw=2,label='Fit')
            else:
                ax[i,j].loglog(tf_freqs[i,j],np.abs(tf_data[i,j]),linestyle='none',\
                               marker='o',ms=4,alpha=0.5,label='Measurement')
                ax[i,j].loglog(freqs[tf_freq_inds[i,j]],force_cal_factors[i]*np.ones(len(tf_data[i,j]))\
                               /np.abs(Harr[:,i,j][tf_freq_inds[i,j]]),lw=2,label='Fit')
            ax[i,j].grid(which='both')
        ax[i,i].legend(fontsize=10)

    return fig,ax


def spectra(agg_dict,descrip=None,harms=[],which='roi',ylim=None):
    '''
    Plof of the QPD and PSPD spectra for a given dataset.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    freqs = agg_dict['freqs']
    fsamp = agg_dict['fsamp']
    nsamp = agg_dict['nsamp']
    fft_to_asd = np.sqrt(nsamp/2./fsamp)

    qpd_x_asds = np.abs(agg_dict['qpd_ffts_full'][:,0,:]*fft_to_asd)
    qpd_y_asds = np.abs(agg_dict['qpd_ffts_full'][:,1,:]*fft_to_asd)
    pspd_x_asds = np.abs(agg_dict['pspd_ffts_full'][:,0,:]*fft_to_asd)
    pspd_y_asds = np.abs(agg_dict['pspd_ffts_full'][:,1,:]*fft_to_asd)
    z_asds = np.abs(agg_dict['qpd_ffts_full'][:,2,:]*fft_to_asd)

    qpd_x_asd = np.sqrt(np.mean(qpd_x_asds**2,axis=0))
    qpd_y_asd = np.sqrt(np.mean(qpd_y_asds**2,axis=0))
    pspd_x_asd = np.sqrt(np.mean(pspd_x_asds**2,axis=0))
    pspd_y_asd = np.sqrt(np.mean(pspd_y_asds**2,axis=0))
    z_asd = np.sqrt(np.mean(z_asds**2,axis=0))

    fig,ax = plt.subplots()
    if len(harms):
        [ax.axvline(3*(i+1),ls='--',lw=1.5,alpha=0.5,color='black') for i in harms]
    ax.set_ylabel('ASD [N/$\sqrt{\mathrm{Hz}}$]')
    ax.set_xlabel('Frequency [Hz]')
    if which=='roi':
        if ylim is None:
            ylim = [1e-17,2e-14]
        ax.semilogy(freqs,qpd_x_asd,lw=1,alpha=0.65,label='QPD $x$')
        ax.semilogy(freqs,qpd_y_asd,lw=1,alpha=0.65,label='QPD $y$')
        ax.semilogy(freqs,pspd_x_asd,lw=1,alpha=0.65,label='PSPD $x$')
        ax.semilogy(freqs,pspd_y_asd,lw=1,alpha=0.65,label='PSPD $y$')
        ax.semilogy(freqs,z_asd,lw=1,alpha=0.65,label='$z$')
        ax.set_xlim([0.1,50])
        ax.set_ylim(ylim)
        ax.set_title('Calibrated spectra in ROI for '+descrip)
    elif which=='full':
        if ylim is None:
            ylim = [1e-18,2e-14]
        ax.loglog(freqs,qpd_x_asd,lw=1,alpha=0.65,label='QPD $x$')
        ax.loglog(freqs,qpd_y_asd,lw=1,alpha=0.65,label='QPD $y$')
        ax.loglog(freqs,pspd_x_asd,lw=1,alpha=0.65,label='PSPD $x$')
        ax.loglog(freqs,pspd_y_asd,lw=1,alpha=0.65,label='PSPD $y$')
        ax.loglog(freqs,z_asd,lw=1,alpha=0.65,label='$z$')
        ax.set_xlim([0.1,max(freqs)])
        ax.set_ylim(ylim)
        ax.set_title('Full calibrated spectra for '+descrip)
    elif which=='rayleigh':
        if ylim is None:
            ylim = [0,2]
        ax.plot(freqs,rayleigh(qpd_x_asds**2),lw=1,alpha=0.65,label='QPD $x$')
        ax.plot(freqs,rayleigh(qpd_y_asds**2),lw=1,alpha=0.65,label='QPD $y$')
        ax.plot(freqs,rayleigh(pspd_x_asds**2),lw=1,alpha=0.65,label='PSPD $x$')
        ax.plot(freqs,rayleigh(pspd_y_asds**2),lw=1,alpha=0.65,label='PSPD $y$')
        ax.plot(freqs,rayleigh(z_asds**2),lw=1,alpha=0.65,label='$z$')
        ax.set_ylabel('Rayleigh statistic [1/$\sqrt{\mathrm{Hz}}$]')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim([0.1,50])
        ax.set_ylim(ylim)
        ax.set_title('Rayleigh spectra in ROI for '+descrip)
    ax.grid(which='both')
    ax.legend(ncol=3)

    # for some reason memory is not released and in subsequent function calls this can cause errors
    del qpd_x_asds,qpd_y_asds,pspd_x_asds,pspd_y_asds,z_asds,\
        qpd_x_asd,qpd_y_asd,pspd_x_asd,pspd_y_asd,z_asd,freqs

    return fig,ax


def spectrogram(agg_dict,descrip=None,sensor='qpd',axis_ind=0,which='roi',\
                t_bin_width=600,vmin=None,vmax=None):
    '''
    Plot a spectrogram for the given dataset.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    freqs = agg_dict['freqs']
    fsamp = agg_dict['fsamp']
    nsamp = agg_dict['nsamp']
    fft_to_asd = np.sqrt(nsamp/2./fsamp)
    asds = np.abs(agg_dict[sensor+'_ffts_full'][:,axis_ind,:])*fft_to_asd

    times = agg_dict['times']
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.

    delta_t = (av_times[1]-av_times[0])*1e-9
    t_bins = int(round(t_bin_width/delta_t))
    num_t_bins = int(len(av_times)/t_bins)
    spec_ray = np.zeros((num_t_bins,len(freqs)))
    spec_asd = np.zeros((num_t_bins,len(freqs)))
    plot_times = np.zeros(num_t_bins)

    fig,ax = plt.subplots()
    if which=='roi':
        if (vmin is None) or (vmax is None):
            vmin = 2e-18
            vmax = 2e-14
        for i in range(spec_ray.shape[0]):
            spec_asd[i,:] = np.sqrt(np.mean(asds[i*t_bins:(i+1)*t_bins,:]**2,axis=0))
            plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])
        pcm = ax.pcolormesh(plot_times,freqs,spec_asd.T,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='magma')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim([0.1,50])
        ax.set_xlabel('Time since '+start_date+' [hours]')
        ax.set_title('ROI '+sensor.upper()+' spectrogram for '+descrip)
        cbar = fig.colorbar(pcm)
        cbar.set_label('ASD [N/$\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=15)
    elif which=='full':
        if (vmin is None) or (vmax is None):
            vmin = 2e-19
            vmax = 2e-14
        for i in range(spec_ray.shape[0]):
            spec_asd[i,:] = np.sqrt(np.mean(asds[i*t_bins:(i+1)*t_bins,:]**2,axis=0))
            plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])
        pcm = ax.pcolormesh(plot_times,freqs,spec_asd.T,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='magma')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim([0.1,max(freqs)])
        ax.set_xlabel('Time since '+start_date+' [hours]')
        ax.set_yscale('log')
        ax.set_title('Full '+sensor.upper()+' spectrogram for '+descrip)
        cbar = fig.colorbar(pcm)
        cbar.set_label('ASD [N/$\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=15)
    elif which=='rayleigh':
        if (vmin is None) or (vmax is None):
            vmin = 0
            vmax = 2
        for i in range(spec_ray.shape[0]):
            spec_ray[i,:] = rayleigh(asds[i*t_bins:(i+1)*t_bins,:]**2)
            plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])
        pcm = ax.pcolormesh(plot_times,freqs,spec_ray.T,vmin=vmin,vmax=vmax,cmap='coolwarm')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim([0.1,50])
        ax.set_xlabel('Time since '+start_date+' [hours]')
        ax.set_title(sensor.upper()+' Rayleigh spectrogram for '+descrip)
        cbar = fig.colorbar(pcm)
        cbar.set_label('Rayleigh statistic [1/$\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=15)

    # for some reason memory is not released and in subsequent function calls this can cause errors
    del freqs,asds,times,av_times,start_date,hours

    return fig,ax


def time_evolution(agg_dict,descrip=None,sensor='qpd',axis_ind=0,\
                   t_bin_width=600,ylim=None):
    '''
    Plot the time evolution of the measurement of a sensor along a given axis.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    # get timing information from the dictionary
    times = agg_dict['times']
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.
    freqs = agg_dict['freqs']
    fsamp = agg_dict['fsamp']
    nsamp = agg_dict['nsamp']
    fft_to_asd = np.sqrt(nsamp/2./fsamp)
    good_freqs = freqs[agg_dict['good_inds']]
    axes = ['x','y','z']
    colors = plt.get_cmap('plasma',len(good_freqs)+1)

    # get amplitude and phase for the sensor and axis
    asds = np.abs(agg_dict[sensor+'_ffts'][:,axis_ind,:])*fft_to_asd
    phases = np.angle(agg_dict[sensor+'_ffts'][:,axis_ind,:])*180./np.pi

    # average by time
    delta_t = (av_times[1]-av_times[0])*1e-9
    t_bins = int(round(t_bin_width/delta_t))
    num_t_bins = int(len(av_times)/t_bins)
    asd_t = np.zeros((num_t_bins,asds.shape[1]))
    phase_t = np.zeros((num_t_bins,asds.shape[1]))
    plot_times = np.zeros(num_t_bins)

    for i in range(num_t_bins):
        asd_t[i,:] = np.sqrt(np.mean(asds[i*t_bins:(i+1)*t_bins,:]**2,axis=0))
        phase_t[i,:] = np.mean(phases[i*t_bins:(i+1)*t_bins,:],axis=0)
        plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])

    # plot the results
    fig,ax = plt.subplots(2,1,figsize=(8,8),sharex=True)
    for i in range(len(good_freqs)):
        ax[0].semilogy(plot_times,asd_t[:,i],ls='none',marker='o',ms=2,alpha=0.65,\
                    label='{:.1f} Hz'.format(good_freqs[i]),color=colors(i))
        ax[1].plot(plot_times,phase_t[:,i],ls='none',marker='o',ms=2,alpha=0.65,\
                label='{:.1f} Hz'.format(good_freqs[i]),color=colors(i))
    ax[0].set_ylabel('ASD [N/$\sqrt{\mathrm{Hz}}$]')
    if ylim is not None:
        ax[0].set_ylim(ylim)
    ax[0].set_title('Time evolution of '+sensor.upper()+' $'+axes[axis_ind]+'$ for '+descrip)
    ax[0].grid(which='both')
    ax[0].legend(fontsize=8,ncol=4)

    ax[1].set_xlabel('Time since '+start_date+' [hours]')
    ax[1].set_ylabel('Phase [$^\circ$]')
    ax[1].set_xlim([0,max(plot_times)])
    ax[1].set_ylim([-200,200])
    ax[1].set_yticks([-180,0,180])
    ax[1].grid(which='both')

    # for some reason memory is not released and in subsequent function calls this can cause errors
    del freqs,asds,phases,times,av_times,start_date,hours,\
        plot_times,asd_t,phase_t,num_t_bins,t_bins,delta_t

    return fig,ax


def position_drift(agg_dict,descrip=None,t_bin_width=600):
    '''
    Plot the drift over time in the position of the bead and cantilever,
    along with the laser and transmitted power.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    # get parameters to plot
    lp = agg_dict['mean_laser_power']
    pt = agg_dict['mean_p_trans']
    bh = agg_dict['bead_height']
    cx = agg_dict['mean_cant_pos'][:,0]
    cz = agg_dict['mean_cant_pos'][:,2]
    x0 = cx[0]
    z0 = cz[0]
    cx = cx - x0
    cz = cz - z0
    
    times = agg_dict['times']
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.

    # average by time
    delta_t = (av_times[1]-av_times[0])*1e-9
    t_bins = int(round(t_bin_width/delta_t))
    num_t_bins = int(len(av_times)/t_bins)
    lp_t = np.zeros((num_t_bins))
    pt_t = np.zeros((num_t_bins))
    bh_t = np.zeros((num_t_bins))
    cx_t = np.zeros((num_t_bins))
    cz_t = np.zeros((num_t_bins))
    plot_times = np.zeros(num_t_bins)

    for i in range(num_t_bins):
        lp_t[i] = np.mean(lp[i*t_bins:(i+1)*t_bins])
        pt_t[i] = np.mean(pt[i*t_bins:(i+1)*t_bins])
        bh_t[i] = np.mean(bh[i*t_bins:(i+1)*t_bins])
        cx_t[i] = np.mean(cx[i*t_bins:(i+1)*t_bins])
        cz_t[i] = np.mean(cz[i*t_bins:(i+1)*t_bins])
        plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig,ax = plt.subplots(2,1,figsize=(8,8),sharex=True)
    l1 = ax[0].plot(plot_times,lp_t,color=colors[0],alpha=0.65,label='Laser power')
    l2 = ax[0].plot(plot_times,pt_t*40,color=colors[1],alpha=0.65,label='Trans. power ($\\times 40$)')
    ax[0].set_title('Time evolution of laser power, bead height, and cantilever position')
    ax[0].set_ylabel('Power [mW]')
    ax2 = ax[0].twinx()
    l3 = ax2.plot(plot_times,bh_t,color=colors[2],ls='-.',label='Bead height')
    ls = l1+l2+l3
    labs = [l.get_label() for l in ls]
    ax2.set_ylabel('Bead height [$\mu$m]',rotation=270,labelpad=15)
    ax[0].grid(which='both')
    ax[0].legend(ls,labs).set_zorder=(0)

    ax[1].plot(plot_times,cx_t,label='Cant. $x~-$ {:.1f} $\mu$m'.format(x0))
    ax[1].plot(plot_times,cz_t,label='Cant. $z~-$ {:.1f} $\mu$m'.format(z0))
    ax[1].set_ylabel('Cantilever position drift [$\mu$m]')
    ax[1].set_xlabel('Time since '+start_date+' [hours]')
    ax[1].set_xlim([min(plot_times),max(plot_times)])
    ax[1].grid(which='both')
    ax[1].legend()

    del lp,pt,bh,cx,cz,lp_t,pt_t,bh_t,cx_t,cz_t,times,plot_times,av_times,\
        start_date,hours,delta_t,num_t_bins,colors
    
    return fig,ax


def mles_vs_time(agg_dict,descrip=None,sensor='qpd',axis_ind=0,t_bin_width=600):
    '''
    Plot the MLE for alpha over time for a few harmonics.
    '''

    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')
    
    times = agg_dict['times']
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.
    harm_freqs = agg_dict['freqs'][agg_dict['good_inds']]
    axes = [' $x$',' $y$',' $z$']

    # index where lambda is 10 um
    lamb_ind = np.argmin(np.abs(agg_dict['template_params'][0,:]-1e-5))

    # get the best fit alphas
    likelihood_coeffs = fit_alpha_all_files(agg_dict,sensor=sensor)

    # average by time
    delta_t = (av_times[1]-av_times[0])*1e-9
    t_bins = int(round(t_bin_width/delta_t))
    num_t_bins = int(len(av_times)/t_bins)
    alpha_hat_t = np.zeros((num_t_bins,likelihood_coeffs.shape[1]))
    err_alpha_t = np.zeros((num_t_bins,likelihood_coeffs.shape[1]))
    plot_times = np.zeros(num_t_bins)

    for i in range(num_t_bins):
        # add parabolic log-likelihoods
        summed_likelihoods = np.sum(likelihood_coeffs[i*t_bins:(i+1)*t_bins,:,axis_ind,lamb_ind,:],axis=0)
        # then recompute the minimum with the new coefficients
        alpha_hat_t[i,:] = -summed_likelihoods[:,1]/(2.*summed_likelihoods[:,0])
        # one sigma error
        err_alpha_t[i,:] = 1./np.sqrt(2.*summed_likelihoods[:,0])
        plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])

    colors = plt.get_cmap('rainbow',len(harm_freqs))

    fig,ax = plt.subplots()
    for i in range(alpha_hat_t.shape[1]):
        ax.errorbar(plot_times,alpha_hat_t[:,i]/1e8,yerr=2.*err_alpha_t[:,i]/1e8,color=colors(i),ls='none',\
                    alpha=0.65,ms=3,marker='o',label='{:.0f} Hz'.format(harm_freqs[i]))
    ax.plot([], [], ' ', label='95\% CL errors',zorder=0)
    ax.set_title('MLE of $\\alpha(\lambda=10\mathrm{\mu m})$ for '+sensor.upper()+axes[axis_ind]+' over time')
    ax.set_xlabel('Time since '+start_date+' [hours]')
    ax.set_xlim([min(plot_times),max(plot_times)])
    ax.set_ylabel('$\hat{\\alpha} / 10^8$')
    ax.grid(which='both')
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(1,alpha_hat_t.shape[1]+1))+[0]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=4,fontsize=10)

    del times, av_times, start_date, hours, harm_freqs, lamb_ind, likelihood_coeffs,\
        delta_t, t_bins, num_t_bins, alpha_hat_t, plot_times, colors
    
    return fig,ax