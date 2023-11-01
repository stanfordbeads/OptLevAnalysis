import sys
import os
import time
import argparse
sys.path.insert(0,'../lib/')
from matplotlib import pyplot as plt
from matplotlib import style
style.use('optlevstyle.mplstyle')
from data_processing import AggregateData
import plotting as pl
from funcs import *
plt.rcParams['axes.prop_cycle'] = style.library['fivethirtyeight']['axes.prop_cycle']

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('agg_path',type=str)
args = parser.parse_args()
agg_path = args.agg_path

# create folder to save the figures in
fig_path = '../figures/' + agg_path.split('/')[-1][:-3] + '/'
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

# plot transfer functions
fig,ax = pl.transfer_funcs(tf_path,sensor='QPD')
fig.savefig(fig_path+'tf_amp_qpd.png')
fig,ax = pl.transfer_funcs(tf_path,sensor='PSPD')
fig.savefig(fig_path+'tf_amp_pspd.png')
fig,ax = pl.transfer_funcs(tf_path,sensor='QPD',phase=True)
fig.savefig(fig_path+'tf_phase_qpd.png')
fig,ax = pl.transfer_funcs(tf_path,sensor='PSPD',phase=True)
fig.savefig(fig_path+'tf_phase_pspd.png')
plt.close('all')

# plot spectra
fig,ax = pl.spectra(aggdat.agg_dict,which='roi')
fig.savefig(fig_path+'spectra_roi.png')
fig,ax = pl.spectra(aggdat.agg_dict,which='full')
fig.savefig(fig_path+'spectra_full.png')
fig,ax = pl.spectra(aggdat.agg_dict,which='rayleigh')
fig.savefig(fig_path+'spectra_rayleigh.png')
plt.close('all')

# plot spectrograms for x
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=0,which='roi')
fig.savefig(fig_path+'spectrogram_qpd_x_roi.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=0,which='full')
fig.savefig(fig_path+'spectrogram_qpd_x_full.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=0,which='rayleigh')
fig.savefig(fig_path+'spectrogram_qpd_x_rayleigh.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='pspd',axis_ind=0,which='roi')
fig.savefig(fig_path+'spectrogram_pspd_x_roi.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='pspd',axis_ind=0,which='full')
fig.savefig(fig_path+'spectrogram_pspd_x_full.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='pspd',axis_ind=0,which='rayleigh')
fig.savefig(fig_path+'spectrogram_pspd_x_rayleigh.png')
plt.close('all')
# for y
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=1,which='roi')
fig.savefig(fig_path+'spectrogram_qpd_y_roi.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=1,which='full')
fig.savefig(fig_path+'spectrogram_qpd_y_full.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=1,which='rayleigh')
fig.savefig(fig_path+'spectrogram_qpd_y_rayleigh.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='pspd',axis_ind=1,which='roi')
fig.savefig(fig_path+'spectrogram_pspd_y_roi.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='pspd',axis_ind=1,which='full')
fig.savefig(fig_path+'spectrogram_pspd_y_full.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='pspd',axis_ind=1,which='rayleigh')
fig.savefig(fig_path+'spectrogram_pspd_y_rayleigh.png')
plt.close('all')
# for z
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=2,which='roi')
fig.savefig(fig_path+'spectrogram_qpd_z_roi.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=2,which='full')
fig.savefig(fig_path+'spectrogram_qpd_z_full.png')
fig,ax = pl.spectrogram(aggdat.agg_dict,sensor='qpd',axis_ind=2,which='rayleigh')
fig.savefig(fig_path+'spectrogram_qpd_z_rayleigh.png')
plt.close('all')

# plot time evolution of measurements
fig,ax = pl.time_evolution(aggdat.agg_dict,sensor='qpd',axis_ind=0)
fig.savefig(fig_path+'time_ev_qpd_x.png')
fig,ax = pl.time_evolution(aggdat.agg_dict,sensor='qpd',axis_ind=1)
fig.savefig(fig_path+'time_ev_qpd_y.png')
fig,ax = pl.time_evolution(aggdat.agg_dict,sensor='qpd',axis_ind=2)
fig.savefig(fig_path+'time_ev_qpd_z.png')
fig,ax = pl.time_evolution(aggdat.agg_dict,sensor='pspd',axis_ind=0)
fig.savefig(fig_path+'time_ev_pspd_x.png')
fig,ax = pl.time_evolution(aggdat.agg_dict,sensor='pspd',axis_ind=1)
fig.savefig(fig_path+'time_ev_pspd_y.png')
fig,ax = pl.position_drift(aggdat.agg_dict)
fig.savefig(fig_path+'position_drift.png')
plt.close('all')

# plot the measurements at each harmonic in polar form
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=0,sensor='qpd')
fig.savefig(fig_path+'polar_qpd_x.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=1,sensor='qpd')
fig.savefig(fig_path+'polar_qpd_y.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=2,sensor='qpd')
fig.savefig(fig_path+'polar_qpd_z.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=0,sensor='pspd')
fig.savefig(fig_path+'polar_pspd_x.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=1,sensor='pspd')
fig.savefig(fig_path+'polar_pspd_y.png')
plt.close('all')

# plot the measurements at each harmonic in unwrapped form
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=0,sensor='qpd',unwrap=True)
fig.savefig(fig_path+'unwrapped_qpd_x.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=1,sensor='qpd',unwrap=True)
fig.savefig(fig_path+'unwrapped_qpd_y.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=2,sensor='qpd',unwrap=True)
fig.savefig(fig_path+'unwrapped_qpd_z.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=0,sensor='pspd',unwrap=True)
fig.savefig(fig_path+'unwrapped_pspd_x.png')
fig,ax = pl.polar_plots(aggdat.agg_dict,axis_ind=1,sensor='pspd',unwrap=True)
fig.savefig(fig_path+'unwrapped_pspd_y.png')
plt.close('all')

# plot the time evolution of the MLE of alpha
fig,ax = pl.mles_vs_time(aggdat.agg_dict,sensor='qpd',axis_ind=0)
fig.savefig(fig_path+'mles_qpd_x.png')
fig,ax = pl.mles_vs_time(aggdat.agg_dict,sensor='qpd',axis_ind=1)
fig.savefig(fig_path+'mles_qpd_y.png')
fig,ax = pl.mles_vs_time(aggdat.agg_dict,sensor='qpd',axis_ind=2)
fig.savefig(fig_path+'mles_qpd_z.png')
fig,ax = pl.mles_vs_time(aggdat.agg_dict,sensor='pspd',axis_ind=0)
fig.savefig(fig_path+'mles_pspd_x.png')
fig,ax = pl.mles_vs_time(aggdat.agg_dict,sensor='pspd',axis_ind=1)
fig.savefig(fig_path+'mles_pspd_y.png')
plt.close('all')
print('Plots saved to '+fig_path)