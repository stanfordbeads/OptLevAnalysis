import sys
import os
import time
import argparse
from matplotlib import pyplot as plt
from matplotlib import style
style.use('../src/optlevanalysis/optlevstyle.mplstyle')
from optlevanalysis.data_processing import AggregateData
import optlevanalysis.plotting as pl
from optlevanalysis.funcs import *

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('agg_path',type=str)
parser.add_argument('-descrip',type=str,default='')
args = parser.parse_args()
agg_path = args.agg_path
descrip = args.descrip

# create folder to save the figures in
fig_path = '/'.join(agg_path.split('/')[:-1]) + '/figures/'

if not os.path.exists(fig_path):
    os.mkdir(fig_path)
else:
    print('Warning: figure folder already exists! Overwriting existing figures in ')
    for i in range(5):
        print('{}...'.format(5-i))
        time.sleep(1)

# load the processed data
aggdat = AggregateData()
aggdat.load_from_hdf5(agg_path)

# reconstruct the path to the transfer functions
file_path = aggdat.file_list[0]
tf_path = file_path.split('new_trap')[0]+'new_trap_processed/calibrations/transfer_funcs'\
          +file_path.split('new_trap')[1].split('Bead')[0]+'TF.h5'

filter_path = '/data/new_trap_processed/calibrations/data_processing_filters/freq_domain/'\
              +'/'.join(file_path.split('/data/new_trap/')[-1].split('/')[:-1])+'wienerFilters.h5'

if descrip=='':
    descrip = '/'.join('/'.join(file_path.split('/data/new_trap/')[-1].split('/Bead0/Gravity/')).split('/')[:-1])[:-1]

print('Making the standard plots...')

# plot transfer functions
try:
    fig,ax = pl.transfer_funcs(tf_path,sensor='QPD')
    fig.savefig(fig_path+'tf_amp_qpd.png',bbox_inches='tight')
    fig,ax = pl.transfer_funcs(tf_path,sensor='XYPD')
    fig.savefig(fig_path+'tf_amp_xypd.png',bbox_inches='tight')
    fig,ax = pl.transfer_funcs(tf_path,sensor='QPD',phase=True)
    fig.savefig(fig_path+'tf_phase_qpd.png',bbox_inches='tight')
    fig,ax = pl.transfer_funcs(tf_path,sensor='XYPD',phase=True)
    fig.savefig(fig_path+'tf_phase_xypd.png',bbox_inches='tight')
except:
    fig,ax = plt.subplots()
    ax.text(0.23, 0.5, 'Insufficient data for this plot', fontsize=20)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(fig_path+'tf_amp_qpd.png',bbox_inches='tight')
    fig.savefig(fig_path+'tf_amp_xypd.png',bbox_inches='tight')
    fig.savefig(fig_path+'tf_phase_qpd.png',bbox_inches='tight')
    fig.savefig(fig_path+'tf_phase_xypd.png',bbox_inches='tight')
plt.close('all')

# plot wiener filters
try:
    fig,ax = pl.wiener_filters(filter_path,sensor='QPD')
    fig.savefig(fig_path+'filter_amp_qpd.png',bbox_inches='tight')
    fig,ax = pl.wiener_filters(filter_path,sensor='XYPD')
    fig.savefig(fig_path+'filter_amp_xypd.png',bbox_inches='tight')
    fig,ax = pl.wiener_filters(filter_path,sensor='QPD',phase=True)
    fig.savefig(fig_path+'filter_phase_qpd.png',bbox_inches='tight')
    fig,ax = pl.wiener_filters(filter_path,sensor='XYPD',phase=True)
    fig.savefig(fig_path+'filter_phase_xypd.png',bbox_inches='tight')
except:
    fig,ax = plt.subplots()
    ax.text(0.23, 0.5, 'Insufficient data for this plot', fontsize=20)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(fig_path+'filter_amp_qpd.png',bbox_inches='tight')
    fig.savefig(fig_path+'filter_amp_xypd.png',bbox_inches='tight')
    fig.savefig(fig_path+'filter_phase_qpd.png',bbox_inches='tight')
    fig.savefig(fig_path+'filter_phase_xypd.png',bbox_inches='tight')
plt.close('all')

# plot spectra
fig,ax = pl.spectra(aggdat.agg_dict,descrips=descrip,which='roi')
fig.savefig(fig_path+'spectra_roi.png',bbox_inches='tight')
fig,ax = pl.spectra(aggdat.agg_dict,descrips=descrip,which='full')
fig.savefig(fig_path+'spectra_full.png',bbox_inches='tight')
fig,ax = pl.spectra(aggdat.agg_dict,descrips=descrip,which='rayleigh')
fig.savefig(fig_path+'spectra_rayleigh.png',bbox_inches='tight')
plt.close('all')

# plot accelerometer/microphone spectra
fig,ax = pl.env_noise(aggdat.agg_dict,which='roi',sensor='accel')
fig.savefig(fig_path+'spectra_accel_roi.png',bbox_inches='tight')
fig,ax = pl.env_noise(aggdat.agg_dict,which='full',sensor='accel')
fig.savefig(fig_path+'spectra_accel_full.png',bbox_inches='tight')
fig,ax = pl.env_noise(aggdat.agg_dict,which='rayleigh',sensor='accel')
fig.savefig(fig_path+'spectra_accel_rayleigh.png',bbox_inches='tight')
fig,ax = pl.env_noise(aggdat.agg_dict,which='roi',sensor='mic')
fig.savefig(fig_path+'spectra_mic_roi.png',bbox_inches='tight')
fig,ax = pl.env_noise(aggdat.agg_dict,which='full',sensor='mic')
fig.savefig(fig_path+'spectra_mic_full.png',bbox_inches='tight')
fig,ax = pl.env_noise(aggdat.agg_dict,which='rayleigh',sensor='mic')
fig.savefig(fig_path+'spectra_mic_rayleigh.png',bbox_inches='tight')
plt.close('all')

# plot spectrograms
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='qpd',which='roi')
fig.savefig(fig_path+'spectrogram_qpd_roi.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='qpd',which='full')
fig.savefig(fig_path+'spectrogram_qpd_full.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='qpd',which='rayleigh')
fig.savefig(fig_path+'spectrogram_qpd_rayleigh.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='xypd',which='roi')
fig.savefig(fig_path+'spectrogram_xypd_roi.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='xypd',which='full')
fig.savefig(fig_path+'spectrogram_xypd_full.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='xypd',which='rayleigh')
fig.savefig(fig_path+'spectrogram_xypd_rayleigh.png',bbox_inches='tight')
plt.close('all')

# for the accelerometer
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='accel',which='roi')
fig.savefig(fig_path+'spectrogram_accel_roi.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='accel',which='full')
fig.savefig(fig_path+'spectrogram_accel_full.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='accel',which='rayleigh')
fig.savefig(fig_path+'spectrogram_accel_rayleigh.png',bbox_inches='tight')
plt.close('all')

# for the microphones
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='mic',which='roi')
fig.savefig(fig_path+'spectrogram_mic_roi.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='mic',which='full')
fig.savefig(fig_path+'spectrogram_mic_full.png',bbox_inches='tight')
fig,ax = pl.spectrogram(aggdat.agg_dict,descrip=descrip,sensor='mic',which='rayleigh')
fig.savefig(fig_path+'spectrogram_mic_rayleigh.png',bbox_inches='tight')
plt.close('all')

# plot time evolution of measurements
fig,ax = pl.time_evolution(aggdat.agg_dict,descrip=descrip,sensor='qpd',axis_ind=0)
fig.savefig(fig_path+'time_ev_qpd_x.png',bbox_inches='tight')
fig,ax = pl.time_evolution(aggdat.agg_dict,descrip=descrip,sensor='qpd',axis_ind=1)
fig.savefig(fig_path+'time_ev_qpd_y.png',bbox_inches='tight')
fig,ax = pl.time_evolution(aggdat.agg_dict,descrip=descrip,sensor='qpd',axis_ind=2)
fig.savefig(fig_path+'time_ev_qpd_z.png',bbox_inches='tight')
fig,ax = pl.time_evolution(aggdat.agg_dict,descrip=descrip,sensor='xypd',axis_ind=0)
fig.savefig(fig_path+'time_ev_xypd_x.png',bbox_inches='tight')
fig,ax = pl.time_evolution(aggdat.agg_dict,descrip=descrip,sensor='xypd',axis_ind=1)
fig.savefig(fig_path+'time_ev_xypd_y.png',bbox_inches='tight')
fig,ax = pl.position_drift(aggdat.agg_dict,descrip=descrip,pem_sensors=True)
fig.savefig(fig_path+'position_drift.png',bbox_inches='tight')
plt.close('all')

# plot the measurements at each harmonic in polar form
try:
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=0,sensor='qpd')
    fig.savefig(fig_path+'polar_qpd_x.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=1,sensor='qpd')
    fig.savefig(fig_path+'polar_qpd_y.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=2,sensor='qpd')
    fig.savefig(fig_path+'polar_qpd_z.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=0,sensor='xypd')
    fig.savefig(fig_path+'polar_xypd_x.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=1,sensor='xypd')
    fig.savefig(fig_path+'polar_xypd_y.png',bbox_inches='tight')
except:
    fig,ax = plt.subplots()
    ax.text(0.23, 0.5, 'Insufficient data for this plot', fontsize=20)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(fig_path+'polar_qpd_x.png',bbox_inches='tight')
    fig.savefig(fig_path+'polar_qpd_y.png',bbox_inches='tight')
    fig.savefig(fig_path+'polar_qpd_z.png',bbox_inches='tight')
    fig.savefig(fig_path+'polar_xypd_x.png',bbox_inches='tight')
    fig.savefig(fig_path+'polar_xypd_y.png',bbox_inches='tight')
plt.close('all')

# plot the measurements at each harmonic in unwrapped form
try:
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=0,sensor='qpd',unwrap=True)
    fig.savefig(fig_path+'unwrapped_qpd_x.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=1,sensor='qpd',unwrap=True)
    fig.savefig(fig_path+'unwrapped_qpd_y.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=2,sensor='qpd',unwrap=True)
    fig.savefig(fig_path+'unwrapped_qpd_z.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=0,sensor='xypd',unwrap=True)
    fig.savefig(fig_path+'unwrapped_xypd_x.png',bbox_inches='tight')
    fig,ax = pl.polar_plots(aggdat.agg_dict,descrip=descrip,axis_ind=1,sensor='xypd',unwrap=True)
    fig.savefig(fig_path+'unwrapped_xypd_y.png',bbox_inches='tight')
except:
    fig,ax = plt.subplots()
    ax.text(0.23, 0.5, 'Insufficient data for this plot', fontsize=20)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(fig_path+'unwrapped_qpd_x.png',bbox_inches='tight')
    fig.savefig(fig_path+'unwrapped_qpd_y.png',bbox_inches='tight')
    fig.savefig(fig_path+'unwrapped_qpd_z.png',bbox_inches='tight')
    fig.savefig(fig_path+'unwrapped_xypd_x.png',bbox_inches='tight')
    fig.savefig(fig_path+'unwrapped_xypd_y.png',bbox_inches='tight')
plt.close('all')

# plot the time evolution of the MLE of alpha
try:
    fig,ax = pl.mles_vs_time(aggdat.agg_dict,descrip=descrip,sensor='qpd',axis_ind=0,pem_sensors=True)
    fig.savefig(fig_path+'mles_qpd_x.png',bbox_inches='tight')
    fig,ax = pl.mles_vs_time(aggdat.agg_dict,descrip=descrip,sensor='qpd',axis_ind=1,pem_sensors=True)
    fig.savefig(fig_path+'mles_qpd_y.png',bbox_inches='tight')
    fig,ax = pl.mles_vs_time(aggdat.agg_dict,descrip=descrip,sensor='qpd',axis_ind=2,pem_sensors=True)
    fig.savefig(fig_path+'mles_qpd_z.png',bbox_inches='tight')
    fig,ax = pl.mles_vs_time(aggdat.agg_dict,descrip=descrip,sensor='xypd',axis_ind=0,pem_sensors=True)
    fig.savefig(fig_path+'mles_xypd_x.png',bbox_inches='tight')
    fig,ax = pl.mles_vs_time(aggdat.agg_dict,descrip=descrip,sensor='xypd',axis_ind=1,pem_sensors=True)
    fig.savefig(fig_path+'mles_xypd_y.png',bbox_inches='tight')
except:
    fig,ax = plt.subplots()
    ax.text(0.23, 0.5, 'Insufficient data for this plot', fontsize=20)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(fig_path+'mles_qpd_x.png',bbox_inches='tight')
    fig.savefig(fig_path+'mles_qpd_y.png',bbox_inches='tight')
    fig.savefig(fig_path+'mles_qpd_z.png',bbox_inches='tight')
    fig.savefig(fig_path+'mles_xypd_x.png',bbox_inches='tight')
    fig.savefig(fig_path+'mles_xypd_y.png',bbox_inches='tight')
plt.close('all')

# plot the limits and limits vs integration time
try:
    fig,ax = pl.alpha_limit(aggdat.agg_dict,descrip=descrip,sensor='qpd')
    fig.savefig(fig_path+'limit_qpd.png',bbox_inches='tight')
    fig,ax = pl.alpha_limit(aggdat.agg_dict,descrip=descrip,sensor='xypd')
    fig.savefig(fig_path+'limit_xypd.png',bbox_inches='tight')
    fig,ax = pl.limit_vs_integration(aggdat.agg_dict,descrip=descrip,sensor='qpd')
    fig.savefig(fig_path+'integ_qpd.png',bbox_inches='tight')
    fig,ax = pl.limit_vs_integration(aggdat.agg_dict,descrip=descrip,sensor='xypd')
    fig.savefig(fig_path+'integ_xypd.png',bbox_inches='tight')
except Exception as e:
    print(e)
    fig,ax = plt.subplots()
    ax.text(0.23, 0.5, 'Insufficient data for this plot', fontsize=20)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(fig_path+'limit_qpd.png',bbox_inches='tight')
    fig.savefig(fig_path+'limit_xypd.png',bbox_inches='tight')
    fig.savefig(fig_path+'integ_qpd.png',bbox_inches='tight')
    fig.savefig(fig_path+'integ_xypd.png',bbox_inches='tight')
plt.close('all')
print('Plots saved to '+fig_path)