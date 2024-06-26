import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,to_rgba
import matplotlib.style as style
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from datetime import datetime
from scipy import signal
import scipy.stats as st
import h5py
from funcs import *
from stats import *
np.seterr(all='ignore')


def polar_plots(agg_dict,descrip=None,indices=None,unwrap=False,axis_ind=0,sensor='qpd',\
                amp_bins=np.logspace(-17,-15, 30),phase_bins=np.linspace(-np.pi,np.pi,60),\
                vmax=None,plot_templates=True,alphas=np.array((1e8,1e9,1e10))):
    '''
    Plot 2d polar histograms binning the datasets by the amplitude and phase at a particular
    harmonic along a given axes.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')
    if indices is None:
        indices = np.array(range(agg_dict['qpd_ffts'].shape[0]))
    if vmax is None:
        vmax = int(len(agg_dict['timestamp'])/100)
    freqs = agg_dict['freqs']
    harms = freqs[agg_dict['good_inds']]
    cmap_polar = plt.get_cmap('inferno') 
    cmap_polar.set_under(color='white',alpha=0)
    axes = ['$x$','$y$','$z$','null']
    subplot_kw = {'projection':'polar'}
    if unwrap:
        subplot_kw = {}

    fig,axs = plt.subplots(2, 4, figsize=(16,9), sharex=True, sharey=True, subplot_kw=subplot_kw)
    # loop through harmonics
    for h,ax in zip(range(len(harms)), axs.flatten()[:-1]):
        ffts = agg_dict[sensor+'_ffts'][indices][:,axis_ind,h]
        sens_title = sensor.upper()
        # reference phase, will be set to alpha>0 if templates are to be plotted
        zero_phase = 0
        if plot_templates:
            yuk_ffts = agg_dict['template_ffts'][indices][h]
            lambdas = agg_dict['template_params'][indices][h]
            ten_um_ind = np.argmin(np.abs(lambdas-1e-5))
            temp_ind = axis_ind*(axis_ind<3)
            zero_phase = np.angle(yuk_ffts[ten_um_ind,temp_ind,h])
            ax.plot(np.pi*np.ones(len(alphas)),np.abs(yuk_ffts[ten_um_ind,temp_ind,h])*alphas,\
                    color='xkcd:electric green',ls='none',marker='d',ms=10)
            ax.plot(np.zeros(len(alphas)),np.abs(yuk_ffts[ten_um_ind,temp_ind,h])*alphas,\
                    color='xkcd:neon blue',ls='none',marker='d',ms=10)
        phase = np.angle(ffts*np.exp(-1j*zero_phase))
        _,_,_,im = ax.hist2d(phase,np.abs(ffts),bins=(phase_bins,amp_bins),cmap=cmap_polar,vmin=1,vmax=vmax)
        ax.set_yscale('log')
        if unwrap:
            ax.set_xticks([-np.pi,0,np.pi])
            ax.set_xticklabels([-1,0,1],fontsize=16)
            ax.set_ylim([min(amp_bins),max(amp_bins)])
            if h/axs.shape[1]>=1:
                ax.set_xlabel('Phase [$\pi$]',fontsize=18)
            if h%axs.shape[1]==0:
                ax.set_ylabel('Force [N]',fontsize=18)
                ax.yaxis.set_tick_params(labelsize=16)
            ax.grid(which='both')
        else:
            ax.set_xticks(np.array([0,45,90,135,180,225,270,315])*np.pi/180.)
            ax.set_thetalim(0,2*np.pi)
            ax.set_thetagrids(ax.get_xticks()*180./np.pi)
            ax.grid(which='both',axis='y')
            ax.set_rlim([min(amp_bins),max(amp_bins)])
            ax.set_yticks([])
            ax.set_yticklabels([])
        ax.set_title('{:.0f} Hz'.format(harms[h]),fontsize=20)
    axs.flatten()[-1].axis('off')

    y_range_str = '({:.0e},{:.0e})'.format(axs.flatten()[0].get_ylim()[0],axs.flatten()[0].get_ylim()[1])
    alpha_min = str(int(np.log10(min(alphas))))
    alpha_max = str(int(np.log10(max(alphas))))
    textStr = '{} force bins log-spaced in '.format(len(amp_bins))+y_range_str+' N.\n Force scale is log.\n'
    textStr += '{} phase bins, linearly spaced.\n'.format(len(phase_bins))
    if plot_templates:
        textStr += 'Diamonds are for $\\alpha$ ranging \n from $10^{'+alpha_min+'}$ to $10^{'+alpha_max+'}$.\n'
        textStr += 'Blue diamonds mean $\\alpha >0$.\n'
        textStr += 'Green diamonds mean $\\alpha <0$.\n'
        textStr += 'Phase is relative to $\\alpha >0$.\n'
    axs.flatten()[-1].text(-0.2,0., textStr, transform = axs.flatten()[-1].transAxes)
    cbar = fig.colorbar(im, ax=axs.flatten()[-1], orientation='horizontal', fraction=0.1)
    cbar.ax.set_xlabel('Number of 10s datasets',fontsize=18)
    fig.suptitle(sens_title+' measurements for the '+axes[axis_ind]+' axis for '+descrip, fontsize=32)

    # for some reason memory is not released and in subsequent function calls this can cause errors
    del freqs,harms,ffts
    if plot_templates:
        del lambdas, yuk_ffts
    
    return fig,axs


def xy_on_qpd(qpd_diag_mat):
    '''
    Plot the x and y eigenmodes of the sphere overlaid on the QPD to see any
    misalignment and non-orthogonality.
    '''

    # calculate QPD coords in bead eigenspace
    Q_trans_xy = np.matmul(qpd_diag_mat,np.array(((1,0,0,-1),(0,1,-1,0),(0,-1,1,0),(-1,0,0,1)))) #np.diag(np.ones(4)))

    # make the box representing the bead eigenmode QPD
    points = Q_trans_xy[:2,:].T
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    qpd_box = points[sorted_indices]

    # calculate QPD axes in bead eigenspace
    Q_hor_xy = 10*np.matmul(qpd_diag_mat,(1.,1.,-1.,-1.))
    Q_ver_xy = 10*np.matmul(qpd_diag_mat,(1.,-1.,1.,-1.))

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
    ax1.set_xlim([-4,4])
    ax1.set_ylim([-4,4])
    ax1.set_yticks([-2,-1,0,1,2])
    ax1.add_patch(plt.Polygon(qpd_box,ls='-',lw=0.5,edgecolor='black',facecolor='lightgray'))
    ax1.axhline(0,color=colors[0],label='Bead $x$ mode')
    ax1.axvline(0,color=colors[2],label='Bead $y$ mode')
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
    ax2.plot([-x_mode_phys[0],0,x_mode_phys[0]],[-x_mode_phys[1],0,x_mode_phys[1]],color=colors[0],label='Bead $x$ mode')
    ax2.plot([-y_mode_phys[0],0,y_mode_phys[0]],[-y_mode_phys[1],0,y_mode_phys[1]],color=colors[2],label='Bead $y$ mode')
    ax2.axvline(0,color=colors[1],label='QPD axes')
    ax2.axhline(0,color=colors[1])
    ax2.set_xlabel('QPD horizontal axis')
    ax2.set_ylabel('QPD vertical axis')
    ax2.legend()
    ax2.set_aspect('equal')

    return fig,axs


def cross_coupling(agg_dict,qpd_diag_mat,p_x=None,p_y=None,plot_inds=None,plot_null=False):
    '''
    Plot the cross-coupling before and after diagonalization.
    '''
    if plot_inds is None:
        plot_inds = shaking_inds(agg_dict)

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
    window_s1 = agg_dict['window_s1']
    window_s2 = agg_dict['window_s2']
    fft_to_asd = window_s1/np.sqrt(2.*fsamp*window_s2)

    signal_mat = np.array((raw_qpd_1,raw_qpd_2,raw_qpd_3,raw_qpd_4))

    naive_mat = np.array(((1.,1.,-1.,-1.),\
                          (1.,-1.,1.,-1.),\
                          (1.,-1.,-1.,1.)))

    # do the transformations from quadrants to x and y
    new_resp = np.einsum('ij,jkl->ikl',qpd_diag_mat,signal_mat)
    raw_resp = np.einsum('ij,jkl->ikl',naive_mat,signal_mat)

    # pick out the new x and y coordinates
    x_raw = raw_resp[0,:,:]
    y_raw = raw_resp[1,:,:]
    n_raw = raw_resp[2,:,:]
    x_corr = new_resp[0,:,:]
    y_corr = new_resp[1,:,:]
    n_corr = np.sqrt(np.sum(new_resp[2:,:,:]**2,axis=0))

    asd_x_raw = np.sqrt(np.mean(np.abs(np.fft.rfft(x_raw)*fft_to_asd)**2,axis=0))[:len(freqs)]
    asd_y_raw = np.sqrt(np.mean(np.abs(np.fft.rfft(y_raw)*fft_to_asd)**2,axis=0))[:len(freqs)]
    asd_n_raw = np.sqrt(np.mean(np.abs(np.fft.rfft(n_raw)*fft_to_asd)**2,axis=0))[:len(freqs)]
    asd_x_corr = np.sqrt(np.mean(np.abs(np.fft.rfft(x_corr)*fft_to_asd)**2,axis=0))[:len(freqs)]
    asd_y_corr = np.sqrt(np.mean(np.abs(np.fft.rfft(y_corr)*fft_to_asd)**2,axis=0))[:len(freqs)]
    asd_n_corr = np.sqrt(np.mean(np.abs(np.fft.rfft(n_corr)*fft_to_asd)**2,axis=0))[:len(freqs)]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig,ax = plt.subplots()
    ax.semilogy(freqs,asd_x_raw,label='Naive $x$',color=colors[2],lw=1,alpha=0.5)
    ax.semilogy(freqs,asd_y_raw,label='Naive $y$',color=colors[3],lw=1,alpha=0.5)
    if (p_x is not None) and (p_y is not None):
        ax.semilogy(freqs,lor(freqs,*p_x),label='Peak fit $x$',color=colors[2],alpha=0.8)
        ax.semilogy(freqs,lor(freqs,*p_y),label='Peak fit $y$',color=colors[3],alpha=0.8)
    ax.semilogy(freqs,asd_x_corr,label='Diag. $x$',color=colors[0],lw=1,alpha=0.8)
    ax.semilogy(freqs,asd_y_corr,label='Diag. $y$',color=colors[1],lw=1,alpha=0.8)
    if plot_null:
        ax.semilogy(freqs,asd_n_raw,label='Naive null',color=colors[4],alpha=0.5)
        ax.semilogy(freqs,asd_n_corr*1e2/4,label=r'100$\times$ diag. null',color=colors[5],alpha=0.8)
    ax.set_xlim([280,420])
    ax.set_ylim([1e-1,1e2])
    ax.set_xlim([0,50])
    ax.set_ylim([1e-2,1e4])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('ASD [arb/$\sqrt{\mathrm{Hz}}$]')
    ax.set_title('QPD diagonalization')
    ax.grid(which='both')
    ax.legend(ncol=3+plot_null*1)

    return fig,ax


def transfer_funcs(path,sensor='QPD',phase=False,nsamp=50000,fsamp=5000,agg_dict=None,\
                   resonance_drive_factors=[1.,1.,1.],diagonalize_qpd=False,plot_null=False):
    '''
    Plot the transfer functions and the fits for all axes.

    :path:                      path to the saved .h5 processed transfer function file
    :sensor:                    QPD or XYPD, depending on which should be plotted
    :phase:                     if True, plot the phases, otherwise plot the amplitudes
    :agg_dict:                  if provided, data from the agg_dict will be shown rather than the data in the TF file
    :resonance_drive_factors:   factor by which the driving force was scaled down near the resonance for [x,y,z]
    '''

    if (diagonalize_qpd or plot_null) and agg_dict is None:
        print('Error: you must pass an agg_dict when using diagonalize_qpd or plot_null')
        return

    # get the transfer function file with fits
    with h5py.File(path,'r') as tf_file:
        # ensure we are looking for the correct sensor name as multiple have been used for XYPD
        if sensor=='XYPD':
            sensors = list(tf_file['fits'].keys())
            sensor = [s for s in sensors if s not in ['QPD']][0]

        # get raw data from the processed file or the agg_dict argument
        if not phase and agg_dict is not None:
            tf_data = agg_dict[sensor.lower()+'_ffts_full'][:,:3,:]
            if plot_null:
                tf_data[:,2,:] = agg_dict[sensor.lower()+'_ffts_full'][:,3,:]
            freqs = agg_dict['freqs']
            tf_freqs = np.arange(1,700,1)
            tf_freq_inds = np.array([np.argmin(np.abs(freqs - f)) for f in tf_freqs])
            tf_data = tf_data[:,:,tf_freq_inds]
            # add two more axes since the saved data has an independent frequency array for each
            tf_freqs = tf_freqs[np.newaxis,:].repeat(3,axis=0)
            tf_freqs = tf_freqs[np.newaxis,:].repeat(3,axis=0)
            drive_ratios = np.ones_like(tf_freqs,dtype=float)
            for i in range(3):
                drive_ratios[i,:,:][(tf_freqs[i,:,:]>=300) & (tf_freqs[i,:,:]<450)] /= resonance_drive_factors[i]
        else:
            tf_data = tf_file['measurements/'+sensor+'/TF'][:,:,:]
            tf_freqs = tf_file['measurements/'+sensor+'/ff'][:,:,:]
            freqs = np.fft.rfftfreq(int(nsamp),d=1/fsamp)
            drive_ratios = np.ones_like(tf_freqs)

        # extract the transfer function matrix
        Harr = np.zeros((len(freqs), 3, 3), dtype=complex)
        Harr[:,0,0] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zXX'],tf_file['fits/'+sensor+'/pXX'], \
                                         tf_file['fits/'+sensor+'/kXX'],2*np.pi*freqs)[1]
        Harr[:,1,1] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zYY'],tf_file['fits/'+sensor+'/pYY'], \
                                         tf_file['fits/'+sensor+'/kYY'],2*np.pi*freqs)[1]
        Harr[:,2,2] = 1/signal.freqs_zpk(tf_file['fits/'+sensor+'/zZZ'],tf_file['fits/'+sensor+'/pZZ'], \
                                         tf_file['fits/'+sensor+'/kZZ'],2*np.pi*freqs)[1]

        # get the scale factor to calibrate counts to N
        suffix = ''
        if diagonalize_qpd:
            suffix = '_diag'
        force_cal_factors = np.array(tf_file.attrs['scaleFactors_QPD'+suffix])

    # if agg_dict is passed, convert to N and divide by the response at low f to scale to 1
    low_f_inds = np.array([np.argmin(np.abs(tf_freqs-f)) for f in np.arange(10,100)])
    if agg_dict is not None:
        for i in range(3):
            tf_data[:,i,:] *= force_cal_factors[i]
            tf_data[:,i,:] /= np.mean(np.abs(tf_data[i,i,low_f_inds]))

    # if we're plotting the response in the null, it's hard to assign physical units
    # instead we scale it so the noise floor matches that of the driven axis
    if plot_null:
        for i in range(2):
            tf_data[i,2,:] *= np.mean(tf_data[i,i,low_f_inds+15])/np.mean(tf_data[i,i,low_f_inds])
    
    # get indices of frequencies driven during TF measurement
    tf_freq_inds = np.zeros_like(tf_freqs,dtype=int)
    for i in range(tf_freqs.shape[0]):
        for j in range(tf_freqs.shape[1]):
            for k in range(tf_freqs.shape[2]):
                tf_freq_inds[i,j,k] = np.argmin(np.abs(freqs-tf_freqs[i,j,k]))

    # plot the result
    axes = ['$x$','$y$','$z$']
    columns = 3
    if plot_null:
        axes[2] = 'null'
        columns = 2
    rows = 3
    if sensor=='XYPD':
        rows = 2
    title = 'magnitudes'
    if phase:
        title = 'phases'
    fig,ax = plt.subplots(rows,columns,figsize=(3*columns+3,2*rows+4),sharex=True,sharey=True)
    fig.suptitle('Transfer function '+title+' for the '+sensor,fontsize=24)
    for i in range(rows):
        if phase:
            ax[i,0].set_ylabel('Phase [$^\circ$]')
        else:
            ax[i,0].set_ylabel('Mag. '+axes[i]+' [N/N]')
        for j in range(columns):
            ax[0,j].set_title('Drive '+axes[j])
            ax[-1,j].set_xlabel('Frequency [Hz]')
            if phase:
                ax[i,j].semilogx(tf_freqs[j,i],np.angle(tf_data[j,i])*180./np.pi,linestyle='none',\
                                                        marker='o',ms=4,alpha=0.3,label='Measurement')
                if i==j:
                    ax[i,j].semilogx(freqs[tf_freq_inds[j,i]],np.angle(1/Harr[:,j,i][tf_freq_inds[j,i]])\
                                     *180./np.pi,lw=2,label='Fit')
                ax[i,j].set_ylim([-200,200])
                ax[i,j].set_yticks([-180,-90,0,90,180])
            else:
                ax[i,j].loglog(tf_freqs[j,i],np.abs(tf_data[j,i]*drive_ratios[j,i]),linestyle='none',\
                               marker='o',ms=4,alpha=0.3,label='Measurement')
                if i==j:
                    ax[i,j].loglog(freqs[tf_freq_inds[i,j]],1/np.abs(Harr[:,i,j][tf_freq_inds[i,j]]),\
                                   lw=2,label='Fit')
                    ax[i,j].text(1.5,1e-2,r'{{{:.3e}}} N/count'.format(force_cal_factors[i]),fontsize=14)
                ax[i,j].set_ylim([5e-3,2e2])
                ax[i,j].set_yticks(np.logspace(-2,2,5))
            ax[i,j].grid(which='both')
        if i<rows and i<columns:
            ax[i,i].legend(loc='upper left',fontsize=10)

    return fig,ax


def visual_diag_mat(qpd_diag_mat,overlay=True):
    '''
    Visualize the effect of the QPD diagonalization matrix.
    '''

    # combine the two null channels into one
    reduced_mat = qpd_diag_mat[:3,:]
    reduced_mat[-1,:] = np.sum(qpd_diag_mat[3:,:],axis=0)/2.
    signs = np.sign(reduced_mat)
    if overlay:
        reduced_mat *= 2/np.linalg.norm(reduced_mat)
        areas = np.cumsum(np.abs(reduced_mat),axis=0)
        sidelengths = np.sqrt(areas)
    else:
        sidelengths = np.sqrt(np.abs(reduced_mat))

    fig,ax = plt.subplots(figsize=(5,5))
    ax.set_title('QPD weights from diagonalization matrix')
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal')
    colors = style.library['seaborn-v0_8-pastel']['axes.prop_cycle'].by_key()['color']
    linestyles = ['--','-']
    hatches = ['\\\\\\\\','']
    lines = []
    labels = ['$x$','$y$','null','weight $<$ 0']
    [x.set_linewidth(2) for x in ax.spines.values()]
    quadrants = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

    # loop through axes, then through quadrants
    for i in range(3):
        if overlay:
            lines.append(Rectangle([0,0],0,0,facecolor=to_rgba(colors[2-i],0.3),\
                                   edgecolor=to_rgba(colors[2-i],1.)))
        else:
            lines.append(Line2D([0,1],[0,1],linestyle='-',color=colors[i]))
        for j in range(4):
            if overlay:
                width = sidelengths[2-i,j]
                corner = -np.array(quadrants[j]==-1,dtype=int)
                ax.add_patch(Rectangle(corner*(width),width,width,color='white'))
                ax.add_patch(Rectangle(corner*(width),width,width,facecolor=to_rgba(colors[2-i],0.3),\
                                       edgecolor=to_rgba(colors[2-i],1.),hatch=hatches[signs[2-i,j]>0]))
            else:
                ver_edge = 0.5*(1+sidelengths[i,j]*quadrants[j,0]/2.)
                hor_edge = 0.5*(1+sidelengths[i,j]*quadrants[j,1]/2.)
                ax.axvline(sidelengths[i,j]*quadrants[j,0],0.5,hor_edge,color=colors[i],\
                           ls=linestyles[signs[i,j]>0],lw=2,alpha=0.8)
                ax.axhline(sidelengths[i,j]*quadrants[j,1],0.5,ver_edge,color=colors[i],\
                           ls=linestyles[signs[i,j]>0],lw=2,alpha=0.8)

    if overlay:
        lines = lines[::-1]
        lines.append(Rectangle([0,0],0,0,alpha=0.3,hatch=hatches[0],color='k'))
    else:
        lines.append(Line2D([0,1],[0,1],linestyle=linestyles[0],color='k'))
    ax.axvline(0,color='k',lw=2)
    ax.axhline(0,color='k',lw=2)
    ax.text(1.6,1.6,'1',fontsize=30)
    ax.text(1.6,-1.7,'2',fontsize=30)
    ax.text(-1.7,1.6,'3',fontsize=30)
    ax.text(-1.7,-1.7,'4',fontsize=30)
    ax.legend(lines,labels,loc='lower left',bbox_to_anchor=(0.5,0,0.5,0.5),\
              ncol=2,handlelength=1.3,columnspacing=0.6)

    return fig,ax


def spectra(agg_dict,descrip=None,harms=[],which='roi',ylim=None,accel=False,\
            xypd=True,null=True,plot_inds=None):
    '''
    Plof of the QPD and XYPD spectra for a given dataset.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    if plot_inds is None:
        plot_inds = shaking_inds(agg_dict)

    freqs = agg_dict['freqs']
    fsamp = agg_dict['fsamp']
    window_s1 = agg_dict['window_s1']
    window_s2 = agg_dict['window_s2']
    fft_to_asd = window_s1/np.sqrt(2.*fsamp*window_s2)

    qpd_x_asds = np.abs(agg_dict['qpd_ffts_full'][plot_inds,0,:]*fft_to_asd)
    qpd_y_asds = np.abs(agg_dict['qpd_ffts_full'][plot_inds,1,:]*fft_to_asd)
    qpd_n_asds = np.abs(agg_dict['qpd_ffts_full'][plot_inds,3,:]*fft_to_asd)
    xypd_x_asds = np.abs(agg_dict['xypd_ffts_full'][plot_inds,0,:]*fft_to_asd)
    xypd_y_asds = np.abs(agg_dict['xypd_ffts_full'][plot_inds,1,:]*fft_to_asd)
    z_asds = np.abs(agg_dict['qpd_ffts_full'][plot_inds,2,:]*fft_to_asd)
    if accel:
    # convert accelerometer data to m/s^2 using 1000 V/g calibration factor
        accel_ffts = np.fft.rfft(np.sqrt(np.sum(agg_dict['accelerometer']**2,axis=1))\
                                 *9.8/1000.,axis=-1)[:,:len(freqs)]*2./window_s1
        accel_asds = np.abs(accel_ffts*fft_to_asd)

    qpd_x_asd = np.sqrt(np.mean(qpd_x_asds**2,axis=0))
    qpd_y_asd = np.sqrt(np.mean(qpd_y_asds**2,axis=0))
    qpd_n_asd = np.sqrt(np.mean(qpd_n_asds**2,axis=0))
    xypd_x_asd = np.sqrt(np.mean(xypd_x_asds**2,axis=0))
    xypd_y_asd = np.sqrt(np.mean(xypd_y_asds**2,axis=0))
    z_asd = np.sqrt(np.mean(z_asds**2,axis=0))
    if accel:
        accel_asd = np.sqrt(np.mean(accel_asds**2,axis=0))

    fig,ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(harms):
        [ax.axvline(3*(i+1),ls='--',lw=1.5,alpha=0.5,color='black') for i in harms]
    ax.set_ylabel('ASD [N/$\sqrt{\mathrm{Hz}}$]')
    ax.set_xlabel('Frequency [Hz]')

     # if the calibration has not been applied, scale the limits so the data will still appear
    ylim_scale = 1
    if np.all(qpd_x_asds > 1e-14):
        ylim_scale = 1e11
        ax.set_ylabel('ASD [arb/$\sqrt{\mathrm{Hz}}$]')
    
    if which=='roi':
        if ylim is None:
            ylim = np.array([1e-17,2e-14])*ylim_scale
        ax.semilogy(freqs,qpd_x_asd,lw=1,alpha=0.65,label='QPD $x$')
        ax.semilogy(freqs,qpd_y_asd,lw=1,alpha=0.65,label='QPD $y$')
        if null:
            ax.semilogy(freqs,qpd_n_asd,lw=1,alpha=0.65,label='QPD null')
        ax.semilogy(freqs,z_asd,lw=1,alpha=0.65,label='ZPD')
        if xypd:
            ax.semilogy(freqs,xypd_x_asd,lw=1,alpha=0.65,label='XYPD $x$')
            ax.semilogy(freqs,xypd_y_asd,lw=1,alpha=0.65,label='XYPD $y$')
        if accel:
            ax2 = ax.twinx()
            ax2.semilogy(freqs,accel_asd,lw=1,alpha=0.65,color=colors[len(ax.lines)],label='Accel.')
            ax2.set_ylabel('ASD [$\mathrm{m/s^2}/\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=16)
        ax.set_xlim([0.1,50])
        ax.set_ylim(ylim)
        ax.set_title('Calibrated spectra in ROI for '+descrip)
    elif which=='full':
        if ylim is None:
            ylim = np.array([1e-18,2e-14])*ylim_scale
        ax.loglog(freqs,qpd_x_asd,lw=1,alpha=0.65,label='QPD $x$')
        ax.loglog(freqs,qpd_y_asd,lw=1,alpha=0.65,label='QPD $y$')
        if null:
            ax.loglog(freqs,qpd_n_asd,lw=1,alpha=0.65,label='QPD null')
        ax.loglog(freqs,z_asd,lw=1,alpha=0.65,label='ZPD')
        if xypd:
            ax.loglog(freqs,xypd_x_asd,lw=1,alpha=0.65,label='XYPD $x$')
            ax.loglog(freqs,xypd_y_asd,lw=1,alpha=0.65,label='XYPD $y$')
        if accel:
            ax2 = ax.twinx()
            ax2.loglog(freqs,accel_asd,lw=1,alpha=0.65,color=colors[len(ax.lines)],label='Accel.')
            ax2.set_ylabel('ASD [$\mathrm{m/s^2}/\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=16)
        ax.set_xlim([0.1,max(freqs)])
        ax.set_ylim(ylim)
        ax.set_title('Full calibrated spectra for '+descrip)
    elif which=='rayleigh':
        if ylim is None:
            ylim = [1e-1,1e1]
        ax.semilogy(freqs,rayleigh(qpd_x_asds**2),lw=1,alpha=0.65,label='QPD $x$')
        ax.semilogy(freqs,rayleigh(qpd_y_asds**2),lw=1,alpha=0.65,label='QPD $y$')
        ax.semilogy(freqs,rayleigh(xypd_x_asds**2),lw=1,alpha=0.65,label='XYPD $x$')
        ax.semilogy(freqs,rayleigh(xypd_y_asds**2),lw=1,alpha=0.65,label='XYPD $y$')
        ax.semilogy(freqs,rayleigh(z_asds**2),lw=1,alpha=0.65,label='ZPD')
        if accel:
            ax2 = ax.twinx()
            ax2.semilogy(freqs,rayleigh(accel_asds**2),lw=1,alpha=0.65,color=colors[len(ax.lines)],label='Accel.')
            ax2.set_ylim(ylim)
        ax.set_ylabel('Rayleigh statistic [1/$\sqrt{\mathrm{Hz}}$]')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim([0.1,50])
        ax.set_ylim(ylim)
        ax.set_title('Rayleigh spectra in ROI for '+descrip)
    ax.grid(which='both')
    if accel:
        lines = ax.lines + ax2.lines
        labels = [l.get_label() for l in lines]
        ax.legend(lines,labels,ncol=2+int(xypd))
    else:
        ax.legend(ncol=2+int(xypd))

    # for some reason memory is not released and in subsequent function calls this can cause errors
    del qpd_x_asds,qpd_y_asds,xypd_x_asds,xypd_y_asds,z_asds,\
        qpd_x_asd,qpd_y_asd,xypd_x_asd,xypd_y_asd,z_asd,freqs

    return fig,ax


def spectrogram(agg_dict,descrip=None,sensor='qpd',axis_ind=0,which='roi',\
                t_bin_width=None,vmin=None,vmax=None):
    '''
    Plot a spectrogram for the given dataset.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    if t_bin_width is None:
        t_bin_width = max(60,int((agg_dict['timestamp'][-1]-agg_dict['timestamp'][0])/100))

    axes = ['x','y','z']
    freqs = agg_dict['freqs']
    fsamp = agg_dict['fsamp']
    window_s1 = agg_dict['window_s1']
    window_s2 = agg_dict['window_s2']
    fft_to_asd = window_s1/np.sqrt(2.*fsamp*window_s2)
    sensor_title = sensor.upper()
    units = '\mathrm{N}'
    if sensor=='accel':
        # convert accelerometer data to m/s^2 using 1000 V/g calibration factor
        ffts = np.fft.rfft(np.sqrt(np.sum(agg_dict['accelerometer']**2,axis=1))\
                           *9.8/1000.,axis=-1)[:,:len(freqs)]*2./window_s1
        asds = np.abs(ffts*fft_to_asd)
        axis_ind = 2
        sensor_title = 'Accel.'
        units = '\mathrm{m/s^2}'
        if ((vmin is None) or (vmax is None)) and which!='rayleigh':
            vmin = 1e-6
            vmax = 1e-2
    else:
        asds = np.abs(agg_dict[sensor+'_ffts_full'][:,axis_ind,:])*fft_to_asd

    # if the calibration has not been applied, scale the limits so the data will still appear
    ylim_scale = 1
    if np.all(asds > 1e-14) and sensor!='accel':
        ylim_scale = 1e11
        units = '\mathrm{arb}'

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
            vmin = 2e-18*ylim_scale
            vmax = 2e-14*ylim_scale
        for i in range(spec_ray.shape[0]):
            spec_asd[i,:] = np.sqrt(np.mean(asds[i*t_bins:(i+1)*t_bins,:]**2,axis=0))
            plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])
        pcm = ax.pcolormesh(plot_times,freqs,spec_asd.T,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='magma')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim([0.1,50])
        ax.set_xlabel('Time since '+start_date+' [hours]')
        ax.set_title(sensor_title+' $'+axes[axis_ind]+'$ ROI spectrogram for '+descrip)
        cbar = fig.colorbar(pcm)
        cbar.set_label('ASD [$'+units+'/\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=16)
    elif which=='full':
        if (vmin is None) or (vmax is None):
            vmin = 2e-19*ylim_scale
            vmax = 2e-14*ylim_scale
        for i in range(spec_ray.shape[0]):
            spec_asd[i,:] = np.sqrt(np.mean(asds[i*t_bins:(i+1)*t_bins,:]**2,axis=0))
            plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])
        pcm = ax.pcolormesh(plot_times,freqs,spec_asd.T,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='magma')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim([0.1,max(freqs)])
        ax.set_xlabel('Time since '+start_date+' [hours]')
        ax.set_yscale('log')
        ax.set_title(sensor_title+' $'+axes[axis_ind]+'$ full spectrogram for '+descrip)
        cbar = fig.colorbar(pcm)
        cbar.set_label('ASD [$'+units+'/\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=16)
    elif which=='rayleigh':
        if (vmin is None) or (vmax is None):
            vmin = 1e-1
            vmax = 1e1
        for i in range(spec_ray.shape[0]):
            spec_ray[i,:] = rayleigh(asds[i*t_bins:(i+1)*t_bins,:]**2)
            plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])
        pcm = ax.pcolormesh(plot_times,freqs,spec_ray.T,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='coolwarm')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim([0.1,50])
        ax.set_xlabel('Time since '+start_date+' [hours]')
        ax.set_title(sensor_title+' $'+axes[axis_ind]+'$ Rayleigh spectrogram for '+descrip)
        cbar = fig.colorbar(pcm)
        cbar.set_label('Rayleigh statistic [1/$\sqrt{\mathrm{Hz}}$]',rotation=270,labelpad=16)

    # for some reason memory is not released and in subsequent function calls this can cause errors
    del freqs,asds,times,av_times,start_date,hours

    return fig,ax


def time_evolution(agg_dict,descrip=None,sensor='qpd',axis_ind=0,\
                   t_bin_width=None,ylim=None):
    '''
    Plot the time evolution of the measurement of a sensor along a given axis.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    if t_bin_width is None:
        t_bin_width = max(60,int((agg_dict['timestamp'][-1]-agg_dict['timestamp'][0])/20))

    # get timing information from the dictionary
    times = agg_dict['times']
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.
    freqs = agg_dict['freqs']
    fsamp = agg_dict['fsamp']
    window_s1 = agg_dict['window_s1']
    window_s2 = agg_dict['window_s2']
    fft_to_asd = window_s1/np.sqrt(2.*fsamp*window_s2)
    good_freqs = freqs[agg_dict['good_inds']]
    axes = ['x','y','z']
    colors = plt.get_cmap('plasma',len(good_freqs)+1)

    # get amplitude and phase for the sensor and axis
    asds = np.abs(agg_dict[sensor+'_ffts'][:,axis_ind,:])*fft_to_asd
    phases = np.angle(agg_dict[sensor+'_ffts'][:,axis_ind,:])*180./np.pi

    # take the average in each time interval
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
        ax[0].semilogy(plot_times,asd_t[:,i],ls='none',marker='o',ms=6,alpha=0.45,\
                    label='{:.1f} Hz'.format(good_freqs[i]),color=colors(i))
        ax[1].plot(plot_times,phase_t[:,i],ls='none',marker='o',ms=6,alpha=0.45,\
                label='{:.1f} Hz'.format(good_freqs[i]),color=colors(i))
    ax[0].set_ylabel('ASD [N/$\sqrt{\mathrm{Hz}}$]')
    if ylim is not None:
        ax[0].set_ylim(ylim)
    ax[0].set_title('Time evolution of '+sensor.upper()+' $'+axes[axis_ind]+'$ for '+descrip)
    ax[0].grid(which='both')
    ax[0].legend(fontsize=12,ncol=4)

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


def position_drift(agg_dict,descrip=None,t_bin_width=None):
    '''
    Plot the drift over time in the position of the bead and cantilever,
    along with the laser and transmitted power.
    '''
    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    if t_bin_width is None:
        t_bin_width = max(60,int((agg_dict['timestamp'][-1]-agg_dict['timestamp'][0])/20))

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

    # take the average in each time interval
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
    l1 = ax[0].plot(plot_times,lp_t*1e3,color=colors[0],alpha=0.65,label='Laser power')
    l2 = ax[0].plot(plot_times,pt_t*40e3,color=colors[1],alpha=0.65,label='Trans. power ($\\times 40$)')
    ax[0].set_title('Time evolution of parameters for '+descrip)
    ax[0].set_ylabel('Power [$\mu$W]')
    ax2 = ax[0].twinx()
    l3 = ax2.plot(plot_times,bh_t*1e3,color=colors[2],ls='-.',label='Bead height')
    ls = l1+l2+l3
    labs = [l.get_label() for l in ls]
    ax2.set_ylabel('Bead height [nm]',rotation=270,labelpad=16)
    ax[0].grid(which='both')
    ax2.legend(ls,labs)

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


def mles_vs_time(agg_dict,descrip=None,sensor='qpd',axis_ind=0,t_bin_width=None):
    '''
    Plot the MLE for alpha over time for a few harmonics.
    '''

    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    if t_bin_width is None:
        t_bin_width = max(60,int((agg_dict['timestamp'][-1]-agg_dict['timestamp'][0])/20))
    
    times = agg_dict['times']
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.
    harm_freqs = agg_dict['freqs'][agg_dict['good_inds']]
    axes = ['$x$','$y$','$z$']

    # index where lambda is 10 um
    lamb_ind = np.argmin(np.abs(agg_dict['template_params'][0,:]-1e-5))

    # get the best fit alphas
    likelihood_coeffs = fit_alpha_all_files(agg_dict,sensor=sensor)

    # take the average in each time interval
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
    ax.plot([], [], ' ', label='95\% CI errors',zorder=0)
    ax.set_title(r'MLE of $\alpha(\lambda=10\mu \mathrm{{m}})$ from {{{}}} {{{}}} for {{{}}}'\
                 .format(sensor.upper(),axes[axis_ind],descrip))
    ax.set_xlabel('Time since '+start_date+' [hours]')
    ax.set_xlim([min(plot_times),max(plot_times)])
    ax.set_ylabel(r'$\hat{\alpha} / 10^8$')
    ax.grid(which='both')
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(1,alpha_hat_t.shape[1]+1))+[0]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=4,fontsize=10)

    del times, av_times, start_date, hours, harm_freqs, lamb_ind, likelihood_coeffs,\
        delta_t, t_bins, num_t_bins, alpha_hat_t, plot_times, colors
    
    return fig,ax


def alpha_limit(agg_dict,descrip=None,sensor='qpd',title=None,lim_pos=None,lim_neg=None,\
                lim_abs=None,lim_noise=None):
    '''
    Plot the alpha-lambda limit for a dataset.
    '''

    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    limit_args = [lim_pos,lim_neg]
    
    lambdas = agg_dict['template_params'][0]
    if all(a is None for a in limit_args):
        # compute the limit for this dataset
        likelihood_coeffs = fit_alpha_all_files(agg_dict,sensor=sensor)
        likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='file')
        likelihood_coeffs = np.sum(likelihood_coeffs[:,:2,...],axis=1)
        likelihood_coeffs[...,-1] = -likelihood_coeffs[...,1]/(2.*likelihood_coeffs[...,0])
        lim_abs = get_abs_alpha_vs_lambda(likelihood_coeffs,lambdas)
        likelihood_coeffs = combine_likelihoods_over_dim(likelihood_coeffs,which='harm')
        lim_pos,lim_neg = get_alpha_vs_lambda(likelihood_coeffs,lambdas)
        

    # get the other previously saved limits
    lims = h5py.File('/home/clarkeh/limits_all.h5','r')

    fig,ax = plt.subplots(figsize=(6,5))
    colors = style.library['fivethirtyeight']['axes.prop_cycle'].by_key()['color']
    ax.loglog(lambdas*1e6,np.array(lim_pos),ls='--',lw=2,color=colors[0],alpha=0.6,label=r'This result $\alpha>0$')
    ax.loglog(lambdas*1e6,np.array(lim_neg),ls='-.',lw=2,color=colors[0],alpha=0.6,label=r'This result $\alpha<0$')
    if lim_abs is not None:
        ax.loglog(lambdas*1e6,np.array(lim_abs),ls=':',lw=2,color=colors[0],alpha=0.6,label=r'This result $|\alpha|$')
    if lim_noise is not None:
        ax.loglog(lambdas*1e6,np.array(lim_noise),ls='-',lw=2,color=colors[3],alpha=0.6,label='Noise limit')
    ax.loglog(np.array(lims['wilson/lambda_pos'])*1e6,np.array(lims['wilson/alpha_pos']),\
              ls='--',lw=2,color=colors[1],alpha=0.6,label=r'Wilson $\alpha>0$')
    ax.loglog(np.array(lims['wilson/lambda_neg'])*1e6,np.array(lims['wilson/alpha_neg']),\
              ls='-.',lw=2,color=colors[2],alpha=0.6,label=r'Wilson $\alpha<0$')
    ax.loglog(np.array(lims['best/lambda'])*1e6,np.array(lims['best/alpha']),\
              ls=(0,(1,1)),lw=2,color=colors[4],alpha=0.8)
    ax.fill_between(np.array(lims['best/lambda'])*1e6,np.array(lims['best/alpha']),\
                    1e15*np.ones_like(np.array(lims['best/alpha'])),color=colors[4],alpha=0.2)
    if title is None:
        ax.set_title(r'{{{}}} limits from {{{}}} files for {{{}}}'\
                    .format(sensor.upper(),len(agg_dict['timestamp']),descrip))
    else:
        ax.set_title(title)
    ax.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$\alpha$')
    ax.set_xlim([1e0,1e2])
    ax.set_ylim([1e2,1e12])
    ax.grid(which='both')
    ax.legend(ncol=3-int(all(a is None for a in limit_args)))

    del colors,lims,lambdas,lim_pos,lim_neg,lim_abs,lim_noise,descrip,sensor
    try:
        del likelihood_coeffs,likelihood_coeffs_all
    except NameError:
        pass
    
    return fig,ax


def limit_vs_integration(agg_dict,descrip=None,sensor='qpd'):
    '''
    Plot the evolution of the limit with increasing integration time.
    '''

    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    # sample subsets of the data with a given size and calculate alpha for each
    num_files = len(agg_dict['timestamp'])
    num_chunks = 5
    subset_inds = np.logspace(0,np.log10(num_files),num_chunks).astype(int)
    lambdas = agg_dict['template_params'][0]
    ten_um_ind = np.argmin(np.abs(lambdas-1e-5))
    num_samples = 10
    lim_pos = np.zeros((num_samples,len(subset_inds)))
    lim_neg = np.zeros((num_samples,len(subset_inds)))

    for i in range(num_samples):
        for j,ind in enumerate(subset_inds):
            indices = np.random.randint(0,num_files,ind)
            like_coeffs = fit_alpha_all_files(agg_dict,indices,sensor=sensor)
            like_coeffs = combine_likelihoods_over_dim(like_coeffs,which='file')
            like_coeffs = group_likelihoods_by_test(like_coeffs)
            lim_pos[i,j] = get_limit_from_likelihoods(like_coeffs[:,ten_um_ind,:],alpha_sign=1)
            lim_neg[i,j] = get_limit_from_likelihoods(like_coeffs[:,ten_um_ind,:],alpha_sign=-1)

    # find the mean for each number of datasets
    lim_pos_mean = []
    lim_neg_mean = []
    sigma_pos = []
    sigma_neg = []

    for i in range(num_chunks):
        lim_pos_mean.append(np.mean(lim_pos[:,i][~np.isnan(lim_pos[:,i])]))
        sigma_pos.append(np.std(lim_pos[:,i][~np.isnan(lim_pos[:,i])],axis=0))
        lim_neg_mean.append(-np.mean(lim_neg[:,i][~np.isnan(lim_neg[:,i])],axis=0))
        sigma_neg.append(np.std(lim_neg[:,i][~np.isnan(lim_neg[:,i])],axis=0))

    lim_pos_mean = np.array(lim_pos_mean)
    lim_neg_mean = np.array(lim_neg_mean)
    sigma_pos = np.array(sigma_pos)
    sigma_neg = np.array(sigma_neg)

    # show how integration time should increase in a noise-limited measurement
    first_lim_pos = lim_pos_mean[~np.isnan(lim_pos_mean)][0]
    integ_times = 10**np.arange(1,num_chunks+1)
    noise_lim_pos = first_lim_pos/np.sqrt(integ_times/10)

    first_lim_neg = lim_neg_mean[~np.isnan(lim_neg_mean)][0]
    noise_lim_neg = first_lim_neg/np.sqrt(integ_times/10)

    fig,ax = plt.subplots()
    colors = style.library['fivethirtyeight']['axes.prop_cycle'].by_key()['color']
    ax.fill_between(subset_inds*10,lim_pos_mean-sigma_pos,\
                    lim_pos_mean+sigma_pos,alpha=0.3,color=colors[0],lw=0,zorder=0)
    ax.fill_between(subset_inds*10,lim_neg_mean-sigma_neg,\
                    lim_neg_mean+sigma_neg,alpha=0.3,color=colors[1],lw=0,zorder=2)
    ax.loglog(subset_inds*10,lim_pos_mean,label=r'$\alpha>0$ measured $\pm1\sigma$',color=colors[0],zorder=1)
    ax.loglog(integ_times,noise_lim_pos,ls=':',alpha=0.7,label=r'$\alpha>0$ if noise limited',color=colors[0],zorder=4)
    ax.loglog(subset_inds*10,lim_neg_mean,label=r'$\alpha<0$ measured $\pm1\sigma$',color=colors[1],zorder=3)
    ax.loglog(integ_times,noise_lim_neg,ls=':',alpha=0.7,label=r'$\alpha<0$ if noise limited',color=colors[1],zorder=5)
    ax.set_xlabel('Integration time [s]')
    ax.set_ylabel(r'95\% CL limit on $\alpha(\lambda=10\mu\mathrm{m})$')
    ax.set_title(r'{{{}}} limits vs integration time for {{{}}}'.format(sensor.upper(),descrip))
    ax.set_xlim([min(subset_inds*10),max(subset_inds*10)])
    ax.set_ylim([0.5*min(noise_lim_neg[-1],noise_lim_pos[-1]),2*max(first_lim_pos,first_lim_neg)])
    ax.legend(ncol=2)
    ax.grid()

    del num_files,num_chunks,subset_inds,lambdas,ten_um_ind,num_samples,lim_pos,lim_neg,\
        lim_pos_mean,lim_neg_mean,sigma_pos,sigma_neg,first_lim_pos,first_lim_neg,integ_times,\
        noise_lim_pos,noise_lim_neg,colors

    return fig,ax


def mle_fingerprint(agg_dict,mle_result,file_inds=None,lamb=1e-5,single_beta=False,num_gammas=1,\
                    delta_means=[0.1,0.1],axes=['x','y'],harms=[],channel='motion',log=True,errors=False):
    '''
    Plot the measured and fitted spectral fingerprints, showing the contribution of
    background and signal to the total measurement.
    '''

    if file_inds is None:
        file_inds = shaking_inds(agg_dict)

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

    harm_labels = np.array(['{:.0f}'.format(i) for i in agg_dict['freqs'][agg_dict['good_inds']]])[harm_inds]
    
    # work with a single value of lambda
    lambdas = agg_dict['template_params'][0]
    lambda_ind = np.argmin(np.abs(lambdas-lamb))
    
    # extract the data for these datasets
    yuk_ffts = agg_dict['template_ffts'][file_inds,lambda_ind,...]
    qpd_ffts = agg_dict['qpd_ffts'][file_inds,:,:]
    qpd_sb_ffts = agg_dict['qpd_sb_ffts'][file_inds,:,:]
    num_sb = int(qpd_sb_ffts.shape[2]/qpd_ffts.shape[2])
    data_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,first_axis:second_axis,...]
    data_var = (1./(2.*num_sb))*np.sum(np.real(data_sb_ffts)**2+np.imag(data_sb_ffts)**2,axis=-1)
    background_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,3,:]
    background_var = (1./(2.*num_sb))*np.sum(np.real(background_sb_ffts)**2+np.imag(background_sb_ffts)**2,axis=-1)

    alpha,gammas,deltas,taus = reshape_nll_args(x=mle_result,data_shape=qpd_ffts[...,harm_inds].shape,\
                                                num_gammas=num_gammas,delta_means=delta_means,axes=axes,harms=harms)
    
    # plot the real and imaginary parts separately unless doing a log plot of magnitudes
    if log:
        funcs = [np.abs]
    else:
        funcs = [np.real,np.imag]
    
    # swap mean and absolute value to get the measurement and fits to match
    if channel=='motion':
        noise = np.sqrt(data_var[...,harm_inds] + background_var[:,np.newaxis,harm_inds])
        measurements = qpd_ffts[:,first_axis:second_axis,harm_inds]
        signal_fits = alpha*taus*yuk_ffts[:,first_axis:second_axis,harm_inds]
        background_fits = gammas*(qpd_ffts[:,np.newaxis,3,harm_inds] \
                                  - alpha*np.sum(deltas*taus*yuk_ffts[:,first_axis:second_axis,harm_inds],axis=1)[:,np.newaxis,:])
    elif channel=='null':
        noise = np.sqrt(background_var[:,np.newaxis,harm_inds])
        measurements = qpd_ffts[:,np.newaxis,3,harm_inds]
        signal_fits = alpha*np.sum(deltas*taus*yuk_ffts[:,first_axis:second_axis,harm_inds],axis=1)[:,np.newaxis,:]
        background_fits = measurements - signal_fits
        axes = ['null']
    
    fig,axs = plt.subplots(len(axes),2-int(log),figsize=(6,2*len(axes)+2),sharex=True,sharey=True,squeeze=False)
    colors = [plt.get_cmap('magma',7)(i) for i in range(1,6)]
    titles = ['Real','Imaginary']

    for i in range(np.shape(axs)[0]):
        for j,func in enumerate(funcs):
            axs[i,j].bar(harm_labels,np.mean(noise,axis=0)[i],width=0.8,color=colors[3],label='Noise',zorder=11)
            axs[i,j].bar(harm_labels,np.mean(-noise,axis=0)[i],width=0.8,color=colors[3],zorder=10)
            axs[i,j].bar(harm_labels,func(np.mean(measurements,axis=0))[i],width=0.6,color=colors[1],label='Measurement',zorder=12)
            if errors:
                axs[i,j].errorbar(harm_labels,func(np.mean(measurements,axis=0))[i],yerr=np.std(func(measurements),axis=0)[i],\
                                  ls='none',marker='o',ms=4,lw=2,color=colors[2],label='$\pm1\sigma$',zorder=16)
            sig_bar = func(np.mean(signal_fits,axis=0))[i]
            bkg_bar = func(np.mean(background_fits,axis=0))[i]
            offsets = sig_bar*np.amax(np.vstack((np.sign(sig_bar)*np.sign(bkg_bar),np.zeros_like(bkg_bar))),axis=0)
            axs[i,j].bar(harm_labels,sig_bar,width=0.3,color=colors[4],label='Yukawa fit',zorder=15)
            axs[i,j].bar(harm_labels,bkg_bar+offsets,width=0.3,color=colors[0],label='Background fit',zorder=14)
            if j==0:
                if axes[i]=='null':
                    axs[i,j].set_ylabel('Force in null [arb]'.format(axes[i]))
                else:
                    axs[i,j].set_ylabel('Force in ${}$ [N]'.format(axes[i]))
            if i==0 and not log:
                axs[i,j].set_title(titles[j],fontsize=14)
            if i==axs.shape[0]-1:
                axs[i,j].set_xlabel('Harmonic [Hz]')
            if log:
                axs[i,j].set_yscale('log')
            axs[i,j].grid(which='both',zorder=0)

    handles, labels = axs[-1,-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=5,bbox_to_anchor=(0.55,0.96),columnspacing=0.8,handlelength=1.5)
    fig.suptitle('Best fit to spectral fingerprint in the {} channel'.format(channel),y=1.02)

    return fig,axs


def q_alpha_fit(alphas,q_vals,alpha_hat,range=[-1,2],sigma_sys=None):
    '''
    Plot the fit to the test statistic vs alpha, from which the 95% CL limit is obtained.
    '''

    # alpha values for the fitted parabola plot
    alpha_vals = np.linspace(range[0]*alpha_hat,range[1]*alpha_hat,1000)

    # parameter for the best fit parabola that has the minimum in the correct spot
    a_hat = np.sum(q_vals*(alphas-alpha_hat)**2)/np.sum((alphas-alpha_hat)**4)

    if sigma_sys is not None:
        a_new = a_hat*(1 - sigma_sys**2/(a_hat**-1 + sigma_sys**2))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig,ax = plt.subplots(2,1,figsize=(6,6),sharex=True,gridspec_kw = {'wspace':0,'hspace':0})
    ax[0].semilogy(alphas[q_vals>10],q_vals[q_vals>10],ls='none',marker='o',ms=4,markeredgewidth=1.5,\
                   fillstyle='none',label='Computed',zorder=10)
    ax[0].semilogy(alpha_vals,a_hat*(alpha_vals-alpha_hat)**2,label='Fit')
    if sigma_sys is not None:
        ax[0].semilogy(alpha_vals,a_new*(alpha_vals-alpha_hat)**2,color=colors[3],label='Sys. error added')
        ax[1].plot(alpha_vals,a_new*(alpha_vals-alpha_hat)**2,color=colors[3])
    ax[0].axhline(0,color=colors[2],ls='--',label='95\% CL threshold')
    ax[1].plot(alphas,q_vals,ls='none',marker='o',ms=4,markeredgewidth=1.5,fillstyle='none',zorder=11)
    ax[1].plot(alpha_vals,a_hat*(alpha_vals-alpha_hat)**2)
    ax[1].axhline(st.norm.ppf(1.-0.05/2.)**2,color=colors[2],ls='--')
    y_max = ax[0].get_ylim()[1]
    ax[0].set_ylim([10,y_max])
    ax[0].set_yticks(np.logspace(2,np.floor(np.log10(y_max)),max(int(np.floor(np.log10(y_max))-1),1)))
    ax[1].set_ylim([0,10])
    ax[1].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$q_{\alpha}$ (log scale)')
    ax[1].set_ylabel(r'$q_{\alpha}$ (linear scale)')
    ax[0].set_title('Quadratic fit to test statistic')
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    ax[0].legend(ncol=3-(sigma_sys is not None))

    return fig,ax


def mles_vs_time_background(agg_dict,file_inds=None,lamb=1e-5,single_beta=False,num_gammas=1,\
                            delta_means=[0.1,0.1],phi_sigma=10.,axes=['x','y'],harms=[],spline=False,\
                            alpha_guess=1e9,descrip=None,t_bin_width=None):
    '''
    Plot the MLE parameters over time.
    '''

    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    if t_bin_width is None:
        t_bin_width = max(60,int((agg_dict['timestamp'][-1]-agg_dict['timestamp'][0])/20))

    if file_inds is None:
        file_inds = shaking_inds(agg_dict)
    
    times = agg_dict['times'][file_inds]
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.
    harm_freqs = agg_dict['freqs'][agg_dict['good_inds']]

    # index where lambda is 10 um
    lamb_ind = np.argmin(np.abs(agg_dict['template_params'][0,:]-1e-5))

    # choose which harmonics to include
    if harms==[]:
        harm_inds = np.array(range(len(agg_dict['good_inds'])))
    else:
        harms_full = agg_dict['freqs'][agg_dict['good_inds']]
        harm_inds = [np.argmin(np.abs(3*h - harms_full)) for h in harms]

    # take the average in each time interval
    delta_t = (av_times[1]-av_times[0])*1e-9
    t_bins = int(round(t_bin_width/delta_t))
    num_t_bins = int(len(av_times)/t_bins)
    alpha_hat_t = np.zeros(num_t_bins)
    err_alpha_t = np.zeros(num_t_bins)
    gamma_hat_t = np.zeros((num_t_bins,2,len(harm_inds)))
    delta_hat_t = np.zeros((num_t_bins,2))
    plot_times = np.zeros(num_t_bins)

    # minimize the NLL for each time bin
    for i in range(num_t_bins):
        mle_result = minimize_nll(agg_dict,file_inds=file_inds,delta_means=delta_means,phi_sigma=phi_sigma,lamb=lamb,\
                                  num_gammas=num_gammas,axes=axes,spline=spline,harms=harms,alpha_guess=alpha_guess)
        alpha,gammas,deltas,taus = reshape_nll_args(x=np.array(mle_result.values),data_shape=agg_dict['qpd_ffts'][...,harm_inds].shape,\
                                                    num_gammas=num_gammas,delta_means=delta_means,axes=axes,harms=harms)
        alpha_hat_t[i] = alpha
        err_alpha_t[i] = mle_result.errors['alpha']
        gamma_hat_t[i,0,:] = gammas[0,0,:]
        gamma_hat_t[i,1,:] = gammas[0,1,:]
        delta_hat_t[i,:] = deltas[0,:,0]
        plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])

    colors = [plt.get_cmap('rainbow',len(harm_inds))(i) for i in range(len(harm_inds))]
    fig,ax = plt.subplots(2,2,figsize=(12,10),sharex=True)
    ax[0,0].errorbar(plot_times,alpha_hat_t/1e8,yerr=2.*err_alpha_t/1e8,color=colors[0],ls='none',\
                     ms=6,marker='o',label=r'$\hat{\alpha}\pm2\sigma$')
    ax[0,1].plot(plot_times,delta_hat_t[:,0],ls='-',marker='o',ms=6,color=colors[1],label=r'$x$')
    ax[0,1].plot(plot_times,delta_hat_t[:,1],ls='-',marker='o',ms=6,color=colors[2],label=r'$y$')
    for i in range(len(harm_inds)):
        ax[1,0].plot(plot_times,gamma_hat_t[:,0,i],ls='-',marker='o',ms=6,\
                 color=colors[i],label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
        ax[1,1].plot(plot_times,gamma_hat_t[:,1,i],ls='-',marker='o',ms=6,\
                 color=colors[i],label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
    for i in range(2):
        for j in range(2):
            ax[i,j].grid(which='both')
            ax[i,j].legend(ncol=4)
    x_ylim = ax[1,0].get_ylim()
    y_ylim = ax[1,1].get_ylim()
    ax[1,0].set_ylim([min(x_ylim[0],y_ylim[0]),max(x_ylim[1],y_ylim[1])])
    ax[1,1].set_ylim([min(x_ylim[0],y_ylim[0]),max(x_ylim[1],y_ylim[1])])
    ax[0,0].set_ylabel(r'$\hat{\alpha} / 10^8$')
    ax[0,1].set_ylabel(r'$\hat{\delta}$')
    ax[1,0].set_ylabel(r'$\hat{\gamma}_x$')
    ax[1,1].set_ylabel(r'$\hat{\gamma}_y$')
    ax[1,0].set_xlabel('Time since '+start_date+' [hours]')
    ax[1,1].set_xlabel('Time since '+start_date+' [hours]')
    ax[1,0].set_xlim([min(plot_times),max(plot_times)])
    ax[1,1].set_xlim([min(plot_times),max(plot_times)])
    ax[0,0].set_title(r'Yukawa strength, $\alpha$')
    ax[0,1].set_title('Coupling of signal into null channel')
    ax[1,0].set_title('Coupling of background into $x$ channel')
    ax[1,1].set_title('Coupling of background into $y$ channel')

    fig.suptitle('Time evolution of MLE parameters for '+descrip,fontsize=24)

    del times, av_times, start_date, hours, harm_freqs, lamb_ind, mle_result,\
        delta_t, t_bins, num_t_bins, alpha_hat_t, err_alpha_t, plot_times, colors,\
        alpha, gammas, deltas, taus, harm_inds, descrip, t_bin_width
    
    return fig,ax


def response_vs_time(agg_dict,file_inds=None,harms=[],descrip=None,t_bin_width=None,bands=False):
    '''
    Plot the response in the motion and null channels over time.
    '''

    if descrip is None:
        descrip = datetime.fromtimestamp(agg_dict['timestamp'][0]).strftime('%Y%m%d')

    if t_bin_width is None:
        t_bin_width = max(60,int((agg_dict['timestamp'][-1]-agg_dict['timestamp'][0])/20))

    if file_inds is None:
        file_inds = shaking_inds(agg_dict)

    if harms==[]:
        harm_inds = np.array(range(len(agg_dict['good_inds'])))
    else:
        harms_full = agg_dict['freqs'][agg_dict['good_inds']]
        harm_inds = [np.argmin(np.abs(3*h - harms_full)) for h in harms]

    # get the data to be plotted
    qpd_ffts = agg_dict['qpd_ffts'][file_inds,:,:]
    qpd_sb_ffts = agg_dict['qpd_sb_ffts'][file_inds,:,:]
    num_sb = int(qpd_sb_ffts.shape[2]/qpd_ffts.shape[2])
    data_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,:2,...]
    data_var = (1./(2.*num_sb))*np.sum(np.real(data_sb_ffts)**2+np.imag(data_sb_ffts)**2,axis=-1)
    background_sb_ffts = qpd_sb_ffts.reshape(qpd_sb_ffts.shape[0],qpd_sb_ffts.shape[1],-1,num_sb)[:,3,:]
    background_var = (1./(2.*num_sb))*np.sum(np.real(background_sb_ffts)**2+np.imag(background_sb_ffts)**2,axis=-1)

    # take the average in each time interval
    times = agg_dict['times'][file_inds]
    av_times = np.mean(times,axis=1)
    start_date = datetime.fromtimestamp(av_times[0]*1e-9).strftime('%b %d, %H:%M:%S')
    hours = (av_times-av_times[0])*1e-9/3600.
    harm_freqs = agg_dict['freqs'][agg_dict['good_inds']]
    delta_t = (av_times[1]-av_times[0])*1e-9
    t_bins = int(round(t_bin_width/delta_t))
    num_t_bins = int(len(av_times)/t_bins)
    plot_times = np.zeros(num_t_bins)
    resp_x_real = np.zeros((num_t_bins,len(harm_freqs)))
    resp_y_real = np.zeros((num_t_bins,len(harm_freqs)))
    resp_n_real = np.zeros((num_t_bins,len(harm_freqs)))
    resp_x_imag = np.zeros((num_t_bins,len(harm_freqs)))
    resp_y_imag = np.zeros((num_t_bins,len(harm_freqs)))
    resp_n_imag = np.zeros((num_t_bins,len(harm_freqs)))
    errs_x = np.zeros((num_t_bins,len(harm_freqs)))
    errs_y = np.zeros((num_t_bins,len(harm_freqs)))
    errs_n = np.zeros((num_t_bins,len(harm_freqs)))

    for i in range(num_t_bins):
        resp_x_real[i,:] = np.mean(np.real(qpd_ffts[i*t_bins:(i+1)*t_bins,0,:]),axis=0)
        resp_y_real[i,:] = np.mean(np.real(qpd_ffts[i*t_bins:(i+1)*t_bins,1,:]),axis=0)
        resp_n_real[i,:] = np.mean(np.real(qpd_ffts[i*t_bins:(i+1)*t_bins,3,:]),axis=0)
        resp_x_imag[i,:] = np.mean(np.imag(qpd_ffts[i*t_bins:(i+1)*t_bins,0,:]),axis=0)
        resp_y_imag[i,:] = np.mean(np.imag(qpd_ffts[i*t_bins:(i+1)*t_bins,1,:]),axis=0)
        resp_n_imag[i,:] = np.mean(np.imag(qpd_ffts[i*t_bins:(i+1)*t_bins,3,:]),axis=0)
        errs_x[i,:] = np.sqrt(np.mean(data_var[i*t_bins:(i+1)*t_bins,0,:],axis=0))
        errs_y[i,:] = np.sqrt(np.mean(data_var[i*t_bins:(i+1)*t_bins,1,:],axis=0))
        errs_n[i,:] = np.sqrt(np.mean(background_var[i*t_bins:(i+1)*t_bins,:],axis=0))
        plot_times[i] = np.mean(hours[i*t_bins:(i+1)*t_bins])

    if bands:
        # plot the plus/minus one sigma bands
        x_real_lower = np.mean(resp_x_real,axis=0) - np.std(resp_x_real,axis=0)
        x_real_upper = np.mean(resp_x_real,axis=0) + np.std(resp_x_real,axis=0)
        y_real_lower = np.mean(resp_y_real,axis=0) - np.std(resp_y_real,axis=0)
        y_real_upper = np.mean(resp_y_real,axis=0) + np.std(resp_y_real,axis=0)
        n_real_lower = np.mean(resp_n_real,axis=0) - np.std(resp_n_real,axis=0)
        n_real_upper = np.mean(resp_n_real,axis=0) + np.std(resp_n_real,axis=0)
        x_imag_lower = np.mean(resp_x_imag,axis=0) - np.std(resp_x_imag,axis=0)
        x_imag_upper = np.mean(resp_x_imag,axis=0) + np.std(resp_x_imag,axis=0)
        y_imag_lower = np.mean(resp_y_imag,axis=0) - np.std(resp_y_imag,axis=0)
        y_imag_upper = np.mean(resp_y_imag,axis=0) + np.std(resp_y_imag,axis=0)
        n_imag_lower = np.mean(resp_n_imag,axis=0) - np.std(resp_n_imag,axis=0)
        n_imag_upper = np.mean(resp_n_imag,axis=0) + np.std(resp_n_imag,axis=0)

    fig,ax = plt.subplots(3,2,figsize=(12,16),sharex=True,sharey=True)
    colors = [plt.get_cmap('twilight',len(harm_inds)+1)(i) for i in range(len(harm_inds))]
    axes = ['$x$','$y$','null']
    for i in range(len(harm_inds)):
        ax[0,0].errorbar(plot_times,resp_x_real[:,i],yerr=errs_x[:,i],ms=6,marker='o',color=colors[i],\
                         label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
        ax[0,1].errorbar(plot_times,resp_x_imag[:,i],yerr=errs_x[:,i],ms=6,marker='o',color=colors[i],\
                         label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
        ax[1,0].errorbar(plot_times,resp_y_real[:,i],yerr=errs_y[:,i],ms=6,marker='o',color=colors[i],\
                         label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
        ax[1,1].errorbar(plot_times,resp_y_imag[:,i],yerr=errs_y[:,i],ms=6,marker='o',color=colors[i],\
                         label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
        ax[2,0].errorbar(plot_times,resp_n_real[:,i],yerr=errs_n[:,i],ms=6,marker='o',color=colors[i],\
                         label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
        ax[2,1].errorbar(plot_times,resp_n_imag[:,i],yerr=errs_n[:,i],ms=6,marker='o',color=colors[i],\
                         label='{:.0f} Hz'.format(harm_freqs[harm_inds[i]]))
        
        if bands:
            ax[0,0].fill_between(plot_times,x_real_lower[i],x_real_upper[i],facecolor=colors[i],edgecolor='none',alpha=0.3)
            ax[0,1].fill_between(plot_times,x_imag_lower[i],x_imag_upper[i],facecolor=colors[i],edgecolor='none',alpha=0.3)
            ax[1,0].fill_between(plot_times,y_real_lower[i],y_real_upper[i],facecolor=colors[i],edgecolor='none',alpha=0.3)
            ax[1,1].fill_between(plot_times,y_imag_lower[i],y_imag_upper[i],facecolor=colors[i],edgecolor='none',alpha=0.3)
            ax[2,0].fill_between(plot_times,n_real_lower[i],n_real_upper[i],facecolor=colors[i],edgecolor='none',alpha=0.3)
            ax[2,1].fill_between(plot_times,n_imag_lower[i],n_imag_upper[i],facecolor=colors[i],edgecolor='none',alpha=0.3)

    for i in range(3):
        ax[i,0].set_ylabel('Force in {} [N]'.format(axes[i]))
        for j in range(2):
            ax[i,j].grid(which='both')
            ax[i,j].legend(ncol=4)
    ax[0,0].set_title('Real')
    ax[0,1].set_title('Imaginary')
    ax[2,0].set_xlabel('Time since '+start_date+' [hours]')
    ax[2,1].set_xlabel('Time since '+start_date+' [hours]')
    ax[1,0].set_xlim([min(plot_times),max(plot_times)])
    ax[1,1].set_xlim([min(plot_times),max(plot_times)])

    fig.suptitle('Time evolution of response in the motion and null channels for '+descrip,fontsize=24)

    del times, av_times, start_date, hours, harm_freqs, delta_t, t_bins, num_t_bins,\
        plot_times, resp_x_real, resp_y_real, resp_n_real, resp_x_imag, resp_y_imag, resp_n_imag,\
        descrip, t_bin_width
    
    return fig,ax
