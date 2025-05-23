import numpy as np
import h5py
import os
import re
import pickle
import yaml
import time
import ast
import scipy.interpolate as interp
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.linalg import null_space
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
from copy import deepcopy
import subprocess
from optlevanalysis.funcs import *
from optlevanalysis.plotting import *
import optlevanalysis.signals as s


# ************************************************************************ #
# This module contains the FileData class, used to extract data from a
# single HDF5 file, and the AggregateData class, used to aggregate data
# relevant to the physics analysis from many FileData objects. This code
# builds on previous analysis code written by Chas Blakemore, Alex Rider,
# David Moore, and others.

# CONVENTIONS USED IN THIS MODULE:
# Variables with "fft" in their name are the discrete Fourier transforms
# of a given time series, scaled such that their magnitude at each frequency
# gives the peak amplitude. For plotting, we prefer to show the amplitude
# spectral density, calculated from the RMS amplitude spectrum by dividing
# each point by the frequency bin width (equal to the sampling frequency
# divided by the number of samples). With this definition, the square of
# the amplitude spectral density is equal to the PSD as given by
# scipy.signal.welch, up to the effects of overlap, windowing, and detrending.
# 
#                   -- Clarke Hardy, October 2023 --
# ************************************************************************ #


class FileData:
    """Class used to manage individual data files.
    """

    def __init__(self,path=''):
        """Initializes a RawData object with some metadata attributes and a dict containing
        the raw data, while setting other attributes to default values.

        :param path: Path to the raw HDF5 file, defaults to ''
        :type path: str, optional
        """
        self.file_name = path
        self.data_dict = {}
        self.date = ''
        self.times = np.array(())
        self.accelerometer = np.array(())
        self.microphone = np.array(())
        self.fsamp = 0
        self.nsamp = 0
        self.window_s1 = 0
        self.window_s2 = 0
        self.window = np.array(())
        self.freqs = np.array(())
        self.cant_raw_data = np.array(((),(),()))
        self.quad_raw_data = np.array(((),(),()))
        self.xypd_raw_data = np.array(((),(),()))
        self.qpd_dc_offsets = np.array(())
        self.xypd_dc_offsets = np.array(())
        self.qpd_sum = np.array(())
        self.quad_amps = np.array(((),(),(),(),()))
        self.quad_phases = np.array(((),(),(),(),()))
        self.cant_pos_calibrated = np.array(((),(),()))
        self.mean_cant_pos = np.array(())
        self.qpd_force_calibrated = np.array(((),(),()))
        self.xypd_force_calibrated = np.array(((),(),()))
        self.good_inds = np.array(())
        self.cant_inds = np.array(())
        self.drive_ind = 0
        self.fund_ind = 0
        self.qpd_ffts = np.array(((),(),()))
        self.qpd_ffts_full = np.array(((),(),()))
        self.qpd_sb_ffts = np.array(((),(),()))
        self.xypd_ffts = np.array(((),(),()))
        self.xypd_ffts_full = np.array(((),(),()))
        self.xypd_sb_ffts = np.array(((),(),()))
        self.template_ffts = np.array(())
        self.template_params = np.array(())
        self.motion_likeness = np.array(((),))
        self.cant_fft = np.array(())
        self.force_cal_factors_qpd = np.array([0,0,0])
        self.force_cal_factors_xypd = np.array([0,0,0])
        self.is_bad = False
        self.error_log = ''
        self.qpd_diag_mat = np.array(((1.,1.,-1.,-1.),\
                                      (1.,-1.,1.,-1.),\
                                      (1.,-1.,-1.,1.),\
                                      (1.,1.,1.,1.)))
        self.diagonalize_qpd = False
        self.is_noise = False


    def load_data(self,tf_path=None,cal_drive_freq=71.0,max_freq=2500.,num_harmonics=10,\
                  harms=[],width=0,noise_bins=10,qpd_diag_mat=None,downsample=True,\
                  wiener=[False,True,False,False,False],time_domain=False,cant_drive_freq=3.0,\
                  signal_model=None,ml_model=None,p0_bead=None,mass_bead=0,lightweight=False,\
                  no_tf=False,force_cal_factors=[],window=None,het_phase=False):
        """Applies calibrations to the cantilever data and QPD data, then gets FFTs for both.

        :param tf_path: Path to the HDF5 file containing the transfer function data and fits, defaults to None
        :type tf_path: str, optional
        :param cal_drive_freq: Frequency to be used when calibrating the bead response into force units, defaults to 71.0
        :type cal_drive_freq: float, optional
        :param max_freq: The maximum frequency to keep in the data, defaults to 2500. 
        :type max_freq: float, optional
        :param num_harmonics: Number of harmonics of the drive at which the response should be measured, defaults to 10
        :type num_harmonics: int, optional
        :param width: Width of the mask used to select harmonics, defaults to 0. Not used in typical analyses.
        :type width: int, optional
        :param noise_bins: Number of sideband frequency bins used to compute the noise level, defaults to 10
        :type noise_bins: int, optional
        :param qpd_diag_mat: Matrix used to transform QPD quadrants into x and y motion, defaults to None
        :type qpd_diag_mat: numpy.ndarray, optional
        :param downsample: Downsample the data, defaults to True
        :type downsample: bool, optional
        :param wiener: a list that specifies which signal streams the noise data should be subtracted from, 
        defaults to [False, True, False, False, False]
        :type wiener: list of bools, optional
        :param time_domain: Apply the Wiener filter in the time domain, defaults to False
        :type time_domain: bool, optional
        :param cant_drive_freq: Frequency at which the cantilever is driven, defaults to 3.0
        :type cant_drive_freq: float, optional
        :param signal_model: Signal model to be tested, defaults to None
        :type signal_model: SignalModel, optional
        :param p0_bead: Position of the bead, defaults to None
        :type p0_bead: list, optional
        :param mass_bead: Mass of the bead in picograms, defaults to 0. If not provided, the mass is computed 
        from the nominal radius and density.
        :type mass_bead: int, optional
        :param lightweight: Drop some data after loading, defaults to False
        :type lightweight: bool, optional
        :param no_tf: Don't apply the transfer function calibration, defaults to False
        :type no_tf: bool, optional
        """
        self.date = re.search(r"\d{8,}", self.file_name)[0]
        self.read_hdf5()
        self.fsamp = self.data_dict['fsamp']
        self.accelerometer = self.data_dict['accelerometer']
        self.microphone = self.data_dict['microphone']
        self.bead_height = self.data_dict['bead_height']
        self.camera_status = self.data_dict['camera_status']
        self.get_laser_power()
        if qpd_diag_mat is not None:
            self.qpd_diag_mat = qpd_diag_mat
            self.diagonalize_qpd = True
        self.get_xyz_from_quad(het_phase=het_phase)
        self.xypd_raw_data = self.data_dict['xypd_data']
        self.qpd_dc_offsets = np.concatenate((np.mean(self.quad_raw_data, axis=-1), np.mean(self.quad_null, axis=-1)))
        self.xypd_dc_offsets = np.mean(self.xypd_raw_data, axis=-1)
        self.cant_raw_data = self.data_dict['cant_data']
        self.nsamp = len(self.times)
        self.window_s1 = np.copy(self.nsamp)
        self.window_s2 = np.copy(self.nsamp)
        self.window = np.ones(self.nsamp)
        freqs = np.fft.rfftfreq(self.nsamp, d=1.0/self.fsamp)
        self.freqs = freqs[freqs<=max_freq]
        if downsample:
            self.downsample_raw_data(wiener, time_domain, window)
        self.calibrate_stage_position()
        self.calibrate_bead_response(tf_path=tf_path,sensor='QPD',cal_drive_freq=cal_drive_freq,no_tf=no_tf,\
                                     force_cal_factors=force_cal_factors,time_domain=time_domain,wiener=wiener,\
                                     het_phase=het_phase)
        self.calibrate_bead_response(tf_path=tf_path,sensor='XYPD',cal_drive_freq=cal_drive_freq,no_tf=no_tf,\
                                     time_domain=time_domain,wiener=wiener)
        self.xypd_force_calibrated[2,:] = np.copy(self.qpd_force_calibrated[2,:])
        self.get_boolean_cant_mask(num_harmonics=num_harmonics,harms=harms,\
                                   cant_drive_freq=cant_drive_freq,width=width)
        self.get_ffts_and_noise(noise_bins=noise_bins)
        if ml_model is not None:
            self.get_motion_likeness(ml_model=ml_model)
        if signal_model is not None:
            self.make_templates(signal_model,p0_bead,mass_bead=mass_bead)
        # for use with AggregateData, don't carry around all the raw data
        if lightweight:
            self.drop_raw_data()


    def read_hdf5(self):
        """Reads raw data and metadata from an HDF5 file directly into a dict. The file structure
        may change over time as new sensors or added or removed, so all checks to ensure backwards
        compatibility should be done in this function.
        """
        dd = {}
        with h5py.File(self.file_name,'r') as f:
            
            # these are the name pairs of equivalent fields between datasets, and the list
            # will be populated with the name used in this particular dataset for each variable
            equiv_fields = [['seismometer','acc'],['PSPD','DCQPD']]
            names = ['' for i in range(len(equiv_fields))]
            these_fields = list(f.keys())
            these_attrs = list(f.attrs.keys())
            for i,field in enumerate(equiv_fields):
                name = [n for n in field if n in these_fields]
                if len(name):
                    names[i] = name[0]

            try:
                dd['cant_data'] = np.array(f['cant_data'],dtype=np.float64)
            except:
                dd['cant_data'] = np.zeros((3, len(f['quad_data'])//12))
            if np.prod(dd['cant_data'].shape)==0:
                dd['cant_data'] = np.zeros((3, len(f['quad_data'])//12))
            dd['quad_data'] = np.array(f['quad_data'],dtype=np.float64)
            if names[0] != '':
                accel = np.array(f[names[0]])
                if len(accel.shape) < 2:
                    dd['accelerometer'] = np.array([np.zeros_like(accel),\
                                                    np.zeros_like(accel),\
                                                    accel])
                else:
                    dd['accelerometer'] = accel
            else:
                dd['accelerometer'] = np.zeros_like(dd['cant_data'])
            if 'mic1' in these_fields and len(f['mic1']) > 0:
                dd['microphone'] = np.array(np.array(f['mic1'])[np.newaxis,:])
            else:
                dd['microphone'] = np.zeros((1, len(dd['cant_data'][0])))
            if 'laser_power' in these_fields and len(f['laser_power']) > 0:
                dd['laser_power'] = np.array(f['laser_power'])
            else:
                dd['laser_power'] = np.zeros_like(dd['cant_data'][0])
            if 'p_trans' in these_fields:
                dd['p_trans'] = np.array(f['p_trans'])
            else:
                dd['p_trans'] = np.zeros_like(dd['cant_data'][0])
            if names[1] != '':
                dd['xypd_data'] = np.array(f[names[1]])
            else:
                dd['xypd_data'] = np.zeros_like(dd['cant_data'])
            dd['timestamp_ns'] = os.stat(self.file_name).st_mtime*1e9
            if 'FsampFPGA' in these_attrs:
                dd['fsamp'] = f.attrs['FsampFPGA']/f.attrs['downsampFPGA']
            else:
                dd['fsamp'] = f.attrs['Fsamp']/f.attrs['downsamp']
            if 'cantilever_settings' in these_fields and len(f['cantilever_settings'].shape) > 0:
                dd['cantilever_axis'] = np.argmax([list(f['cantilever_settings'])[i] for i in [1,3,5]])
            elif 'cantilever_axis' in these_attrs:
                dd['cantilever_axis'] = f.attrs['cantilever_axis']
            else:
                dd['cantilever_axis'] = 0
            dd['bead_height'] = f.attrs['bead_height']
            if 'camEXPstat' in these_fields:
                # threshold determined empirically from mean (~4.82) and sigma (~0.004) of TTL signal
                dd['camera_status'] = np.array(np.array(f['camEXPstat']) < 4.8,dtype=int)
            else:
                dd['camera_status'] = np.zeros_like(dd['cant_data'][0],dtype=int)

        self.data_dict = dd


    def drop_raw_data(self):
        """Drops raw data that will not be used by the AggregateData class.
        """
        self.data_dict = {}
        self.laser_power_full = np.array(())
        self.p_trans_full = np.array(())
        self.cant_raw_data = np.array(((),(),()))
        self.quad_raw_data = np.array(((),(),()))
        self.xypd_raw_data = np.array(((),(),()))
        self.quad_amps = np.array(((),(),(),(),()))
        self.quad_phases = np.array(((),(),(),(),()))
        self.cant_pos_calibrated = np.array(((),(),()))
        self.qpd_force_calibrated = np.array(((),(),()))
        self.xypd_force_calibrated = np.array(((),(),()))

    
    def get_laser_power(self):
        """Adds the full time series of laser power and the mean for this file
        as attributes.
        """

        # calibration factor for laser and transmitted power readings
        milliwatts_per_count = 2.72e-9 # PLACEHOLDER, NEED TO ADD CORRECT CALIBRATION FACTOR
        milliwatts_per_volt = 1./3.3e2

        # save the full array of laser power readings
        self.laser_power_full = milliwatts_per_count*self.data_dict['laser_power']

        # save the mean laser power for this file
        self.mean_laser_power = np.mean(self.laser_power_full)

        # save the full array of transmitted power readings
        self.p_trans_full = milliwatts_per_volt*self.data_dict['p_trans']

        # save the mean transmitted power for this file
        self.mean_p_trans = np.mean(self.p_trans_full)
    

    def extract_quad(self):
        """De-interleaves the quad_data to extract timestamp, amplitude, and phase data.
        Since the read request is asynchronous, the timestamp may not be the first entry.
        First step is to identify it, then all other values can be extracted based on the index
        of the first timestamp.

        :raises Exception: If the file creation time does not match the timestamp data in the file.
        :raises Exception: If the timestamps are scrambled, indicating that the file is bad.
        :return: times, amplitudes, and phases of the QPD data
        :rtype: tuple of numpy.ndarray
        """

        # get the data and timestamp from the data_dict
        quad_data = self.data_dict['quad_data']
        timestamp_ns = self.data_dict['timestamp_ns']

        # first timestamp should start at index 10
        tind = 10

        # try reconstructing a timestamp by making a 64-bit object from consecutive 32-bit objects
        ts_attempt = (np.uint32(quad_data[tind]).astype(np.uint64) << np.uint64(32)) + np.uint32(quad_data[tind+1])

        # if it is close to the timestamp from the HDF5 file metadata, we've found the first timestamp
        # within an hour should be good enough for now, but in future it could be calculated dynamically
        diff_thresh = 3600.*1e9*365.
        if(abs(ts_attempt - timestamp_ns) > diff_thresh):
            raise Exception('Error: timestamp does not match file creation time in '+self.file_name)

        # now get the full array of timestamps
        time_part1 = np.uint32(quad_data[tind::12])
        time_part2 = np.uint32(quad_data[tind+1::12])
        if len(time_part1) != len(time_part2):
            time_part1 = time_part1[:-1]
        quad_times = np.left_shift(time_part1.astype(np.uint64), np.uint64(32)) + time_part2.astype(np.uint64)
        
        # now amplitude and phase data
        amps = [quad_data[tind-10::12],quad_data[tind-9::12],quad_data[tind-8::12],quad_data[tind-7::12],quad_data[tind-6::12]]
        phases = [quad_data[tind-5::12],quad_data[tind-4::12],quad_data[tind-3::12],quad_data[tind-2::12],quad_data[tind-1::12]]

        # find the shortest list
        min_length = 1e9
        for i in range(5):
            if len(amps[i])<min_length:
                min_length = len(amps[i])
            if len(phases[i])<min_length:
                min_length = len(phases[i])

        # then ensure they're all the same length
        quad_times = np.array(quad_times[:min_length])
        for i in range(5):
            amps[i] = amps[i][:min_length]
            phases[i] = phases[i][:min_length]

        # check for scrambling of timestamps, and raise an exception if found
        timesteps = np.diff(quad_times)*1e-9
        if np.sum(timesteps > 1e2/self.fsamp)>10:
            raise Exception('Error: timestamps scrambed in '+self.file_name)

        # store the raw QPD data so it can be used to diagonalize the x/y responses
        self.quad_amps = np.array(amps)
        self.quad_phases = np.array(phases)

        # return numpy arrays
        return quad_times, np.array(amps), np.array(phases)
    
    
    def calibrate_stage_position(self):
        """Converts voltage in `cant_data` into microns.
        """

        # may want to look at datasets with no cant_data
        if not len(self.cant_raw_data):
            self.cant_raw_data = np.zeros_like(self.quad_raw_data)

        # could eventually move this elsewhere and reference a config file instead of hard-coded params
        cant_cal_func = lambda x: list(np.repeat([[50.418,50.418,10.0]],self.cant_raw_data.shape[1],axis=0).T*x\
              + np.repeat([[0.0766,0.0766,0]],self.cant_raw_data.shape[1],axis=0).T)

        # save calibrated data as attributes of FileData object
        self.cant_pos_calibrated = np.array(cant_cal_func(self.cant_raw_data))

        # also save the mean position, used for rough binning of datasets later
        self.mean_cant_pos = np.mean(self.cant_pos_calibrated,axis=1)
    
    
    def get_xyz_from_quad(self, het_phase=False):
        """Calculates x, y, and z from the quadrant photodiode amplitude and phase data.
        Uses a QPD diagonalization matrix if one is provided.
        """
        
        self.times,amps,phases = self.extract_quad()

        # mapping from cartesian plane to QPD indices:
        # I -> 0
        # II -> 2
        # III -> 3
        # IV -> 1

        if not het_phase:
            # multiply vector of qpd amps by calibration matrix
            xy_vec = np.matmul(self.qpd_diag_mat, amps[:4,:])
            x = xy_vec[0,:]
            y = xy_vec[1,:]
            n1 = xy_vec[2,:]
            n2 = xy_vec[3,:]

            # total light to normalize by
            quad_sum = np.sum(amps[:4,:],axis=0)

        else:
            xy_vec = np.matmul(self.qpd_diag_mat, phases[:4,:])
            x = xy_vec[0,:]*gv_to_float(np.array([1]),15)#*1.064/2/np.pi)
            y = xy_vec[1,:]*gv_to_float(np.array([1]),15)#*1.064/2/np.pi)
            n1 = xy_vec[2,:]
            n2 = xy_vec[3,:]
            quad_sum = 1.

        # set object attribute with a numpy array of x,y,z
        self.quad_raw_data = np.array([x.astype(np.float64)/quad_sum,y.astype(np.float64)/quad_sum,phases[4]])
        self.quad_null = np.array([n1.astype(np.float64)/quad_sum, n2.astype(np.float64)/quad_sum])
        self.qpd_sum = quad_sum


    def downsample_raw_data(self,wiener=[False,True,False,False,False],time_domain=False,window=None):
        """Downsamples the time series for all sensors, then use a pre-trained Wiener filter to subtract
        coherent noise coupling in from the table. Input `wiener` is a list of bools which specifies
        whether to subtract coherent accelerometer z noise from [QPD x, QPD y, z, XYPD x, XYPD y]

        :param wiener: a list that specifies which signal streams the noise data should be subtracted from, 
        defaults to [False, True, False, False, False]
        :type wiener: list of bools, optional
        """

        # set the downsampling factor
        ds_factor = 20

        # design a low pass filter for downsampling
        LPF = sig.iirdesign(125, 150, 0.01, 160, output='sos', fs=self.fsamp)

        # get arrays of the raw time series
        qpd_x,qpd_y,qpd_z = tuple(self.quad_raw_data)
        qpd_n1, qpd_n2 = tuple(self.quad_null)
        xypd_x,xypd_y,_ = tuple(self.xypd_raw_data)
        accel_x,accel_y,accel_z = tuple(self.accelerometer)
        mic_1 = self.microphone[0]
        cant_x,cant_y,cant_z = tuple(self.cant_raw_data)

        # detrend the data
        qpd_x -= np.mean(qpd_x)
        qpd_y -= np.mean(qpd_y)
        qpd_z -= np.mean(qpd_z)
        qpd_n1 -= np.mean(qpd_n1)
        qpd_n2 -= np.mean(qpd_n2)
        xypd_x -= np.mean(xypd_x)
        xypd_y -= np.mean(xypd_y)
        accel_x -= np.mean(accel_x)
        accel_y -= np.mean(accel_y)
        accel_z -= np.mean(accel_z)
        mic_1 -= np.mean(mic_1)
        mean_cant_x = np.mean(cant_x)
        mean_cant_y = np.mean(cant_y)
        mean_cant_z = np.mean(cant_z)

        # downsample the data prior to applying the Wiener filter
        qpd_x_lpf = gv_decimate(qpd_x, ds_factor, LPF)
        qpd_y_lpf = gv_decimate(qpd_y, ds_factor, LPF)
        qpd_z_lpf = gv_decimate(qpd_z, ds_factor, LPF)
        qpd_n1_lpf = gv_decimate(qpd_n1, ds_factor, LPF)
        qpd_n2_lpf = gv_decimate(qpd_n2, ds_factor, LPF)
        xypd_x_lpf = gv_decimate(xypd_x,ds_factor, LPF)
        xypd_y_lpf = gv_decimate(xypd_y, ds_factor, LPF)
        cant_x_lpf = gv_decimate(cant_x-mean_cant_x, ds_factor, LPF) + mean_cant_x
        cant_y_lpf = gv_decimate(cant_y-mean_cant_y, ds_factor, LPF) + mean_cant_y
        cant_z_lpf = gv_decimate(cant_z-mean_cant_z, ds_factor, LPF) + mean_cant_z
        laser_power = gv_decimate(self.laser_power_full - self.mean_laser_power, ds_factor, LPF) + self.mean_laser_power
        p_trans = gv_decimate(self.p_trans_full-self.mean_p_trans, ds_factor, LPF) + self.mean_p_trans
        times = self.times[::ds_factor]

        accel_x_lpf = gv_decimate(accel_x, ds_factor, LPF)
        accel_y_lpf = gv_decimate(accel_y, ds_factor, LPF)
        accel_z_lpf = gv_decimate(accel_z, ds_factor, LPF)
        mic_1_lpf = gv_decimate(mic_1, ds_factor, LPF)
        accel_lpf = [accel_x_lpf, accel_y_lpf, accel_z_lpf]

        # apply the Wiener filter in the time domain
        # included for backwards compatibility, but the preferred method is to do the filtering at a later
        # step in the frequency domain
        if time_domain:
            preds = self.filter_time_domain(wiener=wiener, accel_lpf=accel_lpf)
        else:
            preds = np.zeros(5)

        # subtract off the coherent noise
        qpd_x_w = qpd_x_lpf - preds[0]
        qpd_y_w = qpd_y_lpf - preds[1]
        qpd_z_w = qpd_z_lpf - preds[2]
        xypd_x_w = xypd_x_lpf - preds[3]
        xypd_y_w = xypd_y_lpf - preds[4]

        # window the data to remove transient artifacts from filtering
        # but not the cantilever data as force(window(cant_position))=/=window(force(cant_position))
        if window is None:
            win = sig.get_window(('tukey',0.05), len(qpd_x_w))
        else:
            win = sig.get_window(window, len(qpd_x_w))
        self.window_s1 = np.sum(win)
        self.window_s2 = np.sum(win**2)
        qpd_x_w *= win
        qpd_y_w *= win
        qpd_z_w *= win
        qpd_n1_w = qpd_n1_lpf*win
        qpd_n2_w = qpd_n2_lpf*win
        xypd_x_w *= win
        xypd_y_w *= win
        accel_lpf *= win
        mic_1_lpf *= win

        # replace existing class attributes with filtered data
        self.quad_raw_data = np.array((qpd_x_w, qpd_y_w, qpd_z_w))
        self.quad_null = np.array((qpd_n1_w, qpd_n2_w))
        self.xypd_raw_data = np.array((xypd_x_w, xypd_y_w, qpd_z_w))
        self.accelerometer = accel_lpf
        self.microphone = np.array((mic_1_lpf,))
        self.cant_raw_data = np.array((cant_x_lpf,cant_y_lpf,cant_z_lpf))
        self.laser_power_full = laser_power
        self.p_trans_full = p_trans
        self.times = times

        # reset the frequency, number of samples, sampling frequency, and window attributes
        self.freqs = np.fft.rfftfreq(len(qpd_x_lpf),ds_factor/self.fsamp)
        self.nsamp = len(qpd_x_lpf)
        self.fsamp = self.fsamp/ds_factor
        self.window = win


    def filter_time_domain(self, wiener, accel_lpf):
        """Apply the Wiener filters in the time domain. This function is included to maintain backwards
        compatibility, but the preferred method is to do the filtering at a later step in the frequency
        domain. Since the microphone data was not recorded at the time this function was used, only the
        accelerometer data is used in the filtering.

        :param wiener: a list that specifies which signal streams the noise data should be subtracted from, 
        defaults to [False, True, False, False, False]
        :type wiener: list of bools, optional
        :param accel_lpf: low-pass filtered accelerometer data
        :type accel_lpf: list of numpy.ndarray
        :return: numpy.ndarray of filtered data
        """        
        try:
            # load the pre-trained filters for this data
            with h5py.File('/data/new_trap_processed/calibrations/data_processing_filters/' +\
                            self.date + '/time_domain/wienerFilters.h5','r') as filter_file:

                # sensor naming conventions have changed, so accept multiple names
                sensor_name = [s for s in list(filter_file.keys()) if s not in ['QPD', 'LPF']][0]

                # start by ensuring the filters are longer than necessary
                W_qpd = np.zeros((3,3,len(self.freqs)))
                W_xypd = np.zeros((2,3,len(self.freqs)))

                axes = ['x','y','z']
                for accel_ind in range(len(axes)):
                    for sens_ind in range(len(axes)):
                        # if the filter doesn't exist, use an array of zeros
                        try:
                            this_filter = np.array(filter_file['QPD/accel_' + axes[accel_ind] + '/W_' + axes[sens_ind]][0,:])
                            W_qpd[sens_ind,accel_ind,:len(this_filter)] = this_filter
                        except KeyError:
                            pass
                        if axes[sens_ind]!='z':
                            try:
                                this_filter = np.array(filter_file[sensor_name + '/accel_' + axes[accel_ind] + '/W_' + axes[sens_ind]][0,:])
                                W_xypd[sens_ind,accel_ind,:len(this_filter)] = this_filter
                            except KeyError:
                                pass

                # truncate the filters to the minimum length
                min_lens = []
                if ~np.all(W_qpd==0):
                    min_lens.append(np.argwhere(W_qpd>0).max())
                if ~np.all(W_xypd==0):
                    min_lens.append(np.argwhere(W_xypd>0).max())
                if len(min_lens):
                    W_qpd = W_qpd[:,:,:np.amin(min_lens)+1]
                    W_xypd = W_xypd[:,:,:np.amin(min_lens)+1]

        except FileNotFoundError:
            W_qpd = np.zeros((3,3,len(self.freqs)))
            W_xypd = np.zeros((2,3,len(self.freqs)))
            self.error_log = 'No filters found for ' + self.file_name + \
                             '. Wiener filtering not applied to this dataset.'

        # loop through and apply the filter for all sensors specified by the input argument
        preds = []
        for i in range(3):
            filters = np.vstack([W_qpd[:,i],W_xypd[:,i]])
            pred = []
            for w,filter in zip(wiener,filters):
                if w:
                    pred.append(sig.lfilter(filter, 1.0, accel_lpf[i]))
                else:
                    pred.append(np.zeros_like(accel_lpf[i]))
            preds.append(pred)

        # sum the predicted cross-coupling from all accelerometer axes for each sensor
        preds = np.sum(np.array(preds),axis=0)

        return preds


    def get_motion_likeness(self,ml_model=None):
        """Option to use a motion-likeness metric to measure/subtract scattered light
        backrounds. Argument `ml_model` is a function that takes the `amps` array and
        returns motion likeness in x and y.

        :param ml_model: A function which returns the motion-likeness in x and y, defaults to None
        :type ml_model: callable, optional
        """

        _,amps,_ = self.extract_quad()

        # by default, everything is considered motion-like.
        if ml_model is None:
            motion_likeness_x = np.ones_like(amps[0])
            motion_likeness_y = np.ones_like(amps[0])
        else:
            motion_likeness_x,motion_likeness_y = ml_model(amps,self.good_inds)

        # set object attribute with a numpy array of x and y
        self.motion_likeness = np.array([motion_likeness_x,motion_likeness_y])


    def calibrate_bead_response(self,tf_path=None,sensor='QPD',cal_drive_freq=71.0,\
                                no_tf=False,force_cal_factors=[],time_domain=False,\
                                wiener=[False,True,False,False,False], het_phase=False):
        """Applies correction using the transfer function to calibrate the
        x, y, and z responses.

        :param tf_path: Path to the transfer function data and fits, defaults to None
        :type tf_path: str, optional
        :param sensor: Which sensor (QPD or XYPD), defaults to 'QPD'
        :type sensor: str, optional
        :param cal_drive_freq: Frequency to use when determining the counts to force units scaling factor, 
        defaults to 71.0
        :type cal_drive_freq: float, optional
        :param no_ft: Don't apply the transfer function calibration, defaults to False
        :type no_tf: bool, optional
        :param force_cal_factors: List of calibration factors for x, y, and z to use if `no_tf` is True, defaults to []
        :type force_cal_factors: list, optional
        """

        if not no_tf:
            # for data from 2023 and later, the code will automatically find the transfer
            # function in the right format. For old data, specify the path manually
            if int(self.date) > 20230101 or sensor=='QPD':
                Harr,force_cal_factors = self.tf_array_fitted(self.freqs,sensor,tf_path=tf_path,\
                                                              diagonalize_qpd=self.diagonalize_qpd,\
                                                              het_phase=het_phase)
            else:
                tf_path = '/data/new_trap_processed/calibrations/transfer_funcs/20200320.trans'
                Harr,force_cal_factors = self.tf_array_interpolated(self.freqs,tf_path=tf_path,\
                                                                    cal_drive_freq=cal_drive_freq)

        else:
            Harr = np.zeros((len(self.freqs), 3, 3))
            if force_cal_factors==[]:
                force_cal_factors = np.ones(3)
            for i in range(3):
                for j in range(3):
                    if i==j:
                        # scale all frequencies by a constant
                        Harr[:,i,j] = force_cal_factors[i]
        
        # get the raw data for the correct sensor
        if sensor=='QPD':
            raw_data = self.quad_raw_data
            self.force_cal_factors_qpd = force_cal_factors
            null_raw = self.quad_null - np.mean(self.quad_null, axis=-1, keepdims=True)
            null_fft = np.fft.rfft(null_raw, axis=-1)[:,:len(self.freqs)]
        elif sensor=='XYPD':
            raw_data = self.xypd_raw_data
            self.force_cal_factors_xypd = force_cal_factors

        # calculate the DFT of the data, then correct using the transfer function matrix
        data_fft = raw_data - np.mean(raw_data,axis=1,keepdims=True)
        data_fft = np.fft.rfft(raw_data)[:,:len(self.freqs)]

        # apply the wiener filter to the uncalibrated spectra
        if not time_domain and not all(~np.array(wiener)):
            data_fft = self.filter_freq_domain(data_fft, sensor=sensor, wiener=wiener, het_phase=het_phase)

        # matrix multiplication with index contraction made explicit
        # 'kj,ki' = matrix multiplication along second two indices (the 3x3 part)
        # output has one free index (j). '->ji' = output uncontracted indices in this order
        calibrated_fft = np.einsum('ikj,ki->ji', Harr, data_fft)

        # take care of nans from inverting transfer functions
        nan_inds = np.isnan(calibrated_fft)
        calibrated_fft[nan_inds] = 0.0+0.0j

        # inverse DFT to get the now-calibrated position data
        bead_force_cal = np.fft.irfft(calibrated_fft)

        # normalize to get the peak amplitude spectrum
        norm_factor = 2./self.window_s1

        # set calibrated time series and ffts as class attributes
        if sensor=='QPD':
            self.qpd_force_calibrated = bead_force_cal
            calibrated_fft = np.append(calibrated_fft, np.repeat(force_cal_factors[0], 1)*null_fft, axis=0)
            self.qpd_ffts_full = calibrated_fft*norm_factor
            self.qpd_dc_offsets *= np.concatenate((self.force_cal_factors_qpd, np.repeat(self.force_cal_factors_qpd[0], 2)))
        elif sensor=='XYPD':
            self.xypd_force_calibrated = bead_force_cal
            self.xypd_ffts_full = calibrated_fft*norm_factor
            self.xypd_dc_offsets *= self.force_cal_factors_xypd
            self.xypd_dc_offsets[-1] = self.qpd_dc_offsets[2]/self.force_cal_factors_qpd[2]


    def filter_freq_domain(self, sensor_ffts, sensor='QPD', wiener=[False,True,False,False,False], \
                           het_phase=False):
        """Applies the Wiener filter in the frequency domain.

        :param wiener: a list that specifies which signal streams the noise data should be subtracted from, 
        defaults to [False, True, False, False, False]
        :type wiener: list of bools, optional
        :param accel_lpf: low-pass filtered accelerometer data
        :type accel_lpf: list of numpy.ndarray
        """

        filter_filename = '/wienerFilters' + ['', '_phaseQuad'][int(het_phase)] + '.h5'
        witness_channels = ['accel_x', 'accel_y', 'accel_z', 'mic_1']

        # DFTs of the accelerometer and microphone channels
        accel_ffts = np.fft.rfft(self.accelerometer)
        mic_ffts = np.fft.rfft(self.microphone)
        witness_ffts = np.vstack([accel_ffts, mic_ffts])
        
        if sensor == 'QPD':
            sensor_channels = ['qpd_x', 'qpd_y', 'zpd']
            skip = ~np.array(wiener[0:3])
        elif sensor == 'XYPD':
            sensor_channels = ['xypd_x', 'xypd_y']
            skip = ~np.array(wiener[3:])
        which_filters = ['filters_shaking', 'filters_noise'][self.is_noise]

        # load the pre-trained filters for this data
        with h5py.File('/data/new_trap_processed/calibrations/data_processing_filters/freq_domain/' + \
                       self.file_name.replace('/data/new_trap/','').replace(self.file_name.split('/')[-1],'') + \
                       filter_filename) as filter_file:
            for i, chan in enumerate(sensor_channels):
                if skip[i]: continue
                filters = []
                max_length = 0
                for w in witness_channels:
                    try:
                        this_filter = filter_file[which_filters][w + '_' + chan]
                        max_length = max(max_length, len(this_filter))
                    except KeyError:
                        continue
                for w in witness_channels:
                    try:
                        filters.append(filter_file[which_filters][w + '_' + chan])
                    except KeyError:
                        filters.append(np.zeros(max_length))
                filters = np.array(filters)
                if filters.shape[-1] != witness_ffts.shape[-1]:
                    filters = np.pad(filters, ((0, 0), (0, witness_ffts.shape[-1] - filters.shape[-1])),\
                                     mode='constant', constant_values=0)
                preds = witness_ffts * filters
                preds = np.sum(preds, axis=0)
                # scale z to physical units to match the scaling in the wiener filter files
                if sensor_channels[i]=='zpd':
                    if het_phase:
                        preds /= (gv_to_float(np.array([1]),15))
                    else:
                        preds /= (1.064*gv_to_float(np.array([1]),15)/2/np.pi)
                sensor_ffts[i] -= preds

        return sensor_ffts


    def tf_array_fitted(self,freqs,sensor,tf_path=None,diagonalize_qpd=False,het_phase=False):
        """Gets the transfer function array from the HDF5 file containing the fitted poles,
        zeros, and gain from the measured transfer functions along x, y, and z, and returns it.

        :param freqs: Frequency array for the transfer functions
        :type freqs: numpy.ndarray
        :param sensor: Which sensor to use for the transfer function (QPD or XYPD)
        :type sensor: str
        :param tf_path: Path to the file containing the transfer function data and fits, defaults to None
        :type tf_path: str, optional
        :param diagonalize_qpd: Calibrate the x and y data streams using the bead eigenmodes, defaults to False
        :type diagonalize_qpd: bool, optional
        :return: Tuple of the transfer function matrix array and the calibration factors
        :rtype: tuple of numpy.ndarray
        """

        # transfer function data should be stored here in a folder named by the date
        if tf_path is None:
            tf_path = '/data/new_trap_processed/calibrations/transfer_funcs/' + str(self.date) \
                      + '/TF' + ['', '_phaseQuad'][int(het_phase)] + '.h5'

        # select the right calibration factor if diagonalizing the QPD
        suffix = ''
        if diagonalize_qpd and (sensor=='QPD') and (int(self.date) > 20230101):
            suffix = '_diag'

        with h5py.File(tf_path,'r') as tf_file:
            # ensure we are looking for the correct sensor name as multiple have been used for XYPD
            if sensor=='XYPD':
                sensors = list(tf_file['fits'].keys())
                sensor = [s for s in sensors if s not in ['QPD']][0]

            # Compute TF at frequencies of interest. Appropriately inverts so we can map response -> drive
            Harr = np.zeros((len(freqs), 3, 3), dtype=complex)
            Harr[:,0,0] = 1/sig.freqs_zpk(tf_file['fits/'+sensor+'/zXX'],tf_file['fits/'+sensor+'/pXX'], \
                                          tf_file['fits/'+sensor+'/kXX']/tf_file.attrs['scaleFactors_'+sensor+suffix][0], 2*np.pi*freqs)[1]
            Harr[:,1,1] = 1/sig.freqs_zpk(tf_file['fits/'+sensor+'/zYY'],tf_file['fits/'+sensor+'/pYY'], \
                                          tf_file['fits/'+sensor+'/kYY']/tf_file.attrs['scaleFactors_'+sensor+suffix][1], 2*np.pi*freqs)[1]
            Harr[:,2,2] = 1/sig.freqs_zpk(tf_file['fits/'+sensor+'/zZZ'],tf_file['fits/'+sensor+'/pZZ'], \
                                          tf_file['fits/'+sensor+'/kZZ']/tf_file.attrs['scaleFactors_'+sensor+suffix][2], 2*np.pi*freqs)[1]
            force_cal_factors = np.array(tf_file.attrs['scaleFactors_'+sensor+suffix])

        return Harr,force_cal_factors
    

    def tf_array_interpolated(self,freqs,tf_path=None,cal_drive_freq=71.,suppress_off_diag=False):
        """Extracts the interpolated transfer function array from a .trans file and returns it.

        :param freqs: Frequency array for the transfer functions
        :type freqs: numpy.ndarray
        :param tf_path: Path to the file containing the transfer function data and fits, defaults to None
        :type tf_path: str, optional
        :param cal_drive_freq: Frequency to use when determining the counts to force units scaling factor, defaults to 71.
        :type cal_drive_freq: float, optional
        :param suppress_off_diag: Ignore off-diagonal elements in the transfer function matrices, defaults to False
        :type suppress_off_diag: bool, optional
        :return: Tuple of the transfer function matrix array and the calibration factors
        :rtype: tuple of numpy.ndarray
        """

        # this tends to cause a ton of divide by zero errors that are handled later, so
        # just temporarily disable warnings
        np.seterr(all='ignore')

        # get the .trans file from the specified path and load the fits and interpolated data
        Hfunc = pickle.load(open(tf_path, 'rb'))
        fits, interps = Hfunc
        # interps is a length-3 list of length-3 lists containing True if matrix element exists
        # and False otherwise
        
        # Number of frequencies at which transfer function should be interpolated
        Nfreq = len(freqs)

        # Harr[f,:,:] is a 3x3 inverse transfer function matrix at frequency f
        Harr = np.zeros((Nfreq,3,3),dtype=np.complex128)

        # for the driving function on each axis
        for drive in [0,1,2]:
            # for the bead response on each axis
            for resp in [0,1,2]:
                if suppress_off_diag and (drive != resp):
                    # skip over off-diagonal elements
                    continue

                # get the interpolated transfer function at each combination of drive/response axes
                interpolate = interps[resp][drive]
                fit = fits[resp][drive]
                if interpolate:

                    # frequency, magnitude, and phase values for the fitted function
                    freqs_old = fit[0][0]
                    mag_old = fit[0][1]
                    phase_old = fit[1][1]
                    
                    # weights for spline fitting
                    mag_weight = (1.0 / np.std(mag_old[:10])) * np.ones(len(freqs_old))
                    phase_weight = (1.0 / np.std(phase_old[:10])) * np.ones(len(freqs_old))

                    # spline interpolation function for both magnitude and phase
                    mag_func = interp.UnivariateSpline(freqs_old, mag_old, w=mag_weight, k=2, s=1.)
                    phase_func = interp.UnivariateSpline(freqs_old, phase_old, w=phase_weight, k=2, s=1.)

                    mag_extrap = \
                        make_extrapolator(mag_func, xs=freqs_old, ys=mag_old, \
                                          pts=fit[0][2], arb_power_law=fit[0][3])

                    phase_extrap = \
                        make_extrapolator(phase_func, xs=freqs_old, ys=phase_old, \
                                          pts=fit[1][2], arb_power_law=fit[1][3], semilogx=True)
                    mag = mag_extrap(freqs)
                    phase = phase_extrap(freqs)

                else:
                    mag = damped_osc_amp(freqs, *fit[0])
                    phase = damped_osc_phase(freqs, *fit[1], phase0=fit[2])

                Harr[:,drive,resp] = mag * np.exp(1.0j * phase)
        
        Hout = np.linalg.inv(Harr)

        # after inversion, ensure that the off-diagonal components are still set to 0
        if suppress_off_diag:
            for drive in [0,1,2]:
                for resp in [0,1,2]:
                    if drive == resp:
                        continue
                    Hout[:,drive,resp] = 0.0 + 0.0j

        # force calibration factor at driven frequency
        drive_freq_ind = np.argmin(np.abs(self.freqs - cal_drive_freq))
        response_matrix = Hout[drive_freq_ind,:,:]
        force_cal_factors = [0,0,0]
        for i in [0,1,2]:
            # assume the response is purely diagonal, and take from it the x, y, and z
            # force calibration factors
            force_cal_factors[i] = np.abs(response_matrix[i,i])

        return Hout,force_cal_factors
    

    def build_drive_mask(self,cant_fft,freqs,num_harmonics=10,width=0,harms=[],cant_drive_freq=3.0):
        """Identifies the fundamental drive frequency and makes an array of harmonics specified
        by the function arguments, then makes a notch mask of the width specified around these harmonics.

        :param cant_fft: The Fourier transform of the cantilever position data
        :type cant_fft: numpy.ndarray
        :param freqs: Frequency array for the cantilever data
        :type freqs: numpy.ndarray
        :param num_harmonics: Number of harmonics of the drive to use, defaults to 10
        :type num_harmonics: int, optional
        :param width: Width of the notch mask, defaults to 0
        :type width: int, optional
        :param harms: Which of the drive harmonics to include, defaults to []
        :type harms: list, optional
        :param cant_drive_freq: Frequency of the cantilever, defaults to 3.0
        :type cant_drive_freq: float, optional
        :return: Tuple of the drive mask and the index of the fundamental frequency
        :rtype: tuple
        """

        # find the drive frequency, ignoring the DC bin
        fund_ind = np.argmin(np.abs(self.freqs-cant_drive_freq))
        drive_freq = freqs[fund_ind]

        # mask is  initialized with 1 at the drive frequency and 0 elsewhere
        drive_mask = np.zeros(len(cant_fft))
        drive_mask[fund_ind] = 1.0

        # can make the notch mask wider than 1 bin with the 'width' argument
        if width:
            lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
            upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
            drive_mask[lower_ind:upper_ind+1] = 1.0

        # create default array of harmonics if an input is not provided
        if len(harms) == 0:
            harms = np.array([x+1 for x in range(num_harmonics)])
        # remove the fundamental frequency if 1 is not in the provided list of harmonics
        elif 1 not in harms:
            drive_mask[fund_ind] = 0.0
            if width:
                drive_mask[lower_ind:upper_ind+1] = 0.0

        # loop over harmonics and add them to the mask
        for n in harms:
            harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
            drive_mask[harm_ind] = 1.0 
            if width:
                h_lower_ind = np.argmin(np.abs(n * drive_freq - 0.5 * width - freqs))
                h_upper_ind = np.argmin(np.abs(n * drive_freq + 0.5 * width - freqs))
                drive_mask[h_lower_ind:h_upper_ind+1] = 1.0
        
        # a boolean array is ultimately needed so do that conversion here
        drive_mask = drive_mask > 0

        return drive_mask, fund_ind


    def get_boolean_cant_mask(self,num_harmonics=10,harms=[],width=0,cant_harms=5,cant_drive_freq=3.0):
        """Builds a boolean mask of a given width for the cantilever drive for the specified harmonics

        :param num_harmonics: Number of harmonics of the drive to use, defaults to 10
        :type num_harmonics: int, optional
        :param harms: Which of the drive harmonics to use, defaults to []
        :type harms: list, optional
        :param width: Width of the mask, defaults to 0
        :type width: int, optional
        :param cant_harms: Number of harmonics of the cantilever drive to use when reconstructing the cantilever position 
        time series, defaults to 5
        :type cant_harms: int, optional
        :param cant_drive_freq: Drive frequency of the cantilever, defaults to 3.0
        :type cant_drive_freq: float, optional
        """

        # driven axis is the one with the maximum amplitude of driving voltage
        drive_ind = int(self.data_dict['cantilever_axis'])

        # fft of the cantilever position vector along the given axis
        cant_fft = np.fft.rfft(self.cant_pos_calibrated[drive_ind])[:len(self.freqs)]

        # get the notch mask for the given harmonics, as well as the index of the fundamental
        # frequency and the drive frequency
        drive_mask, fund_ind = self.build_drive_mask(cant_fft, self.freqs, num_harmonics=num_harmonics,\
                                                     harms=harms, width=width, cant_drive_freq=cant_drive_freq)

        # create array containing the indices of the values that survive the mask
        good_inds = np.arange(len(drive_mask)).astype(int)[drive_mask]

        all_inds = np.arange(len(self.freqs)).astype(int)
        cant_inds = []
        for i in range(cant_harms):
            if width != 0:
                cant_inds += list(all_inds[np.abs(self.freqs - (i+1)*self.freqs[fund_ind]) < 0.5*width])
            else:
                cant_inds.append(np.argmin(np.abs(self.freqs - (i+1)*self.freqs[fund_ind])))

        # compare the drive amplitude to the noise level to determine if this is a noise file
        noise_inds = np.ones_like(cant_fft)
        noise_inds[cant_inds] = 0
        noise_inds[self.freqs<1] = 0
        noise_inds = noise_inds.astype(bool)
        cant_noise = np.mean(np.abs(cant_fft[noise_inds]))
        drive_amp = np.amax(cant_fft[cant_inds])
        if drive_amp < 1e2*cant_noise:
            self.is_noise = True

        # set indices as class attributes
        self.good_inds = good_inds # indices of frequencies where physics search is to be done
        self.cant_inds = cant_inds # indices of frequencies used to reconstruct cantilever stroke
        self.drive_ind = drive_ind # index of the driven cantilever axis
        self.fund_ind = fund_ind # index of the fundamental frequency


    def get_ffts_and_noise(self,noise_bins=10):
        """Get the fft of the x, y, and z data at each of the harmonics and
        some side-band frequencies.

        :param noise_bins: Number of bins to use when computing the sideband noise, defaults to 10
        :type noise_bins: int, optional
        """
        
        # frequency values at the specified harmonics
        harm_freqs = self.freqs[self.good_inds]

        # make sure it's an array even if theres only one harmonic given
        if type(harm_freqs) == np.float64:
            harm_freqs = np.array([harm_freqs])

        # # initialize an array for the qpd ffts of x, y, and z
        qpd_ffts = np.zeros((4, len(self.good_inds)), dtype=np.complex128)
        qpd_sb_ffts = np.zeros((4, len(self.good_inds)*noise_bins), dtype=np.complex128)

        # # initialize arrays for the xypd ffts of x, y, and z
        xypd_ffts = np.zeros((3, len(self.good_inds)), dtype=np.complex128)
        xypd_sb_ffts = np.zeros((3, len(self.good_inds)*noise_bins), dtype=np.complex128)

        # get the fft for the cantilever data along the driven axis
        cant_fft_full = np.fft.rfft(self.cant_pos_calibrated[self.drive_ind])

        # now select only the indices for the chosen number of drive harmonics
        cant_fft = cant_fft_full[self.cant_inds]

        # loop through the axes
        for resp in [0,1,2,3]:

            # get the ffts for the given axis
            qpd_fft = self.qpd_ffts_full[resp,:]
            if resp<3:
                xypd_fft = self.xypd_ffts_full[resp,:]

            # add the fft to the existing array, which was initialized with zeros
            qpd_ffts[resp] += qpd_fft[self.good_inds]
            if resp<3:
                xypd_ffts[resp] += xypd_fft[self.good_inds]

            # now create a list of some number of indices on either side of the harmonic
            sideband_inds = []
            # loop through harmonics
            for freq_ind, freq in enumerate(harm_freqs):
                # this is just a way to get 'noise_bins' bins on either side of the
                # harmonic of interest while skipping the bin immediately above and below
                pos_ind = self.good_inds[freq_ind] + 2
                neg_ind = self.good_inds[freq_ind] - 2
                sign = -1 # tracks whether we're adding indices above or below the harmonic
                # alternate adding indices above and below the harmonic until enough are added
                for i in range(noise_bins):
                    if sign > 0:
                        sideband_inds.append(pos_ind)
                        pos_ind += 1
                    else:
                        sideband_inds.append(neg_ind)
                        neg_ind -= 1
                    sign *= -1
            # sort the indices since they were added by alternating above and below
            sideband_inds.sort()

            # finally, add the data at these indices to the array for this axis
            qpd_sb_ffts[resp] += qpd_fft[sideband_inds]
            if resp<3:
                xypd_sb_ffts[resp] += xypd_fft[sideband_inds]

        # save the ffts as class attributes
        self.qpd_ffts = qpd_ffts
        self.xypd_ffts = xypd_ffts
        self.cant_fft = cant_fft
        self.qpd_sb_ffts = qpd_sb_ffts
        self.xypd_sb_ffts = xypd_sb_ffts


    def make_templates(self,signal_model,p0_bead,mass_bead=0,cant_vec=None,num_harms=10):
        """Make a template of the response to a given signal model. This is intentionally
        written to be applicable to a generic model, but for now will only be used
        for the Yukawa-modified gravity model in which the only parameter is lambda.

        :param signal_model: The signal model to be tested
        :type signal_model: optlevanalysis.signals.SignalModel
        :param p0_bead: List containing the position of the bead in the same coordinate system as the attractor
        :type p0_bead: list of float
        :param mass_bead: Mass of the bead in picograms. If not given the mass is computed from the density and radius, defaults to 0
        :type mass_bead: int, optional
        :param cant_vec: Cantilever position vector, defaults to None
        :type cant_vec: list, optional
        :param num_harms: Number of harmonics of the drive to use, defaults to 10
        :type num_harms: int, optional
        """

        # mass_bead argument should be in picograms. If no bead mass provided,
        # use the nominal mass from the template
        if mass_bead==0:
            mass_fac = 1.
        else:
            mass_bead *= 1e-15 # convert from picograms to kg
            mass_nom = 4./3.*np.pi*signal_model.rad_bead**3*signal_model.rho_bead
            mass_fac = mass_bead/mass_nom

        if cant_vec is None:
            cant_vec = [self.mean_cant_pos[0],self.cant_pos_calibrated[1],self.mean_cant_pos[2]]

        # get the bead position in the coordinate system of the attractor
        bead_x = p0_bead[0] - cant_vec[0]
        bead_y = p0_bead[1] - cant_vec[1]
        bead_z = p0_bead[2] - cant_vec[2]

        # if a file hasn't been loaded, make fake good_inds using the typical sampling params
        good_inds = self.good_inds
        nsamp = self.nsamp
        if len(good_inds)==0:
            nsamp = int(5e4)
            fsamp = 5e3
            drive_freq = 3.
            freqs = np.fft.rfftfreq(nsamp,d=1./fsamp)
            fund_ind = np.argmin(np.abs(freqs-drive_freq))
            good_inds = np.arange(1,num_harms+1)*fund_ind

        # can only use as many harmonics as there are indices for
        num_harms = np.min([len(good_inds), num_harms])

        # get additional parameters 
        params,param_inds = signal_model.get_params_and_inds()

        positions = np.ones((len(bead_y),3))*1e-6
        positions[:,0] *= bead_x
        positions[:,1] *= bead_y
        positions[:,2] *= bead_z

        # for each value of the model parameter, append the array of force for all
        # cantilever positions
        forces = []
        for ind in param_inds:
            forces.append(signal_model.get_force_at_pos(positions,[ind]))
        forces = np.array(forces)*mass_fac

        # detrend the force data before windowing
        forces = forces - np.mean(forces,axis=1,keepdims=True)

        # apply the same window to the force data as the quad data
        forces = forces*self.window[None,:,None]

        # now do ffts of the forces along the cantilever drive to get the harmonics
        force_ffts = np.fft.rfft(forces,axis=1)[:,good_inds,:]*2./self.window_s1
        
        self.template_ffts = force_ffts.swapaxes(1,2)
        self.template_params = params



class AggregateData:
    """Class used to manage collections of data files.
    """

    def __init__(self, data_dirs=[], file_prefixes=[], descrips=[], num_to_load=1000000,\
                 first_index=0, configs=None, file_suffixes=[]):
        """Takes a list of directories containing the files to be aggregated, and optionally
        a list of file prefixes. If given, the list of file prefixes should be the same length as
        the list of data directories. If files with multiple prefixes are required from the same
        directory, add the directory to the list multiple times with the corresponding prefixes
        in the file_prefixes argument.

        :param data_dirs: Directory or directories containing the datasets to be included, defaults to []
        :type data_dirs: str or list, optional
        :param file_prefixes: File prefix or prefixes corresponding to the included datasets, defaults to []
        :type file_prefixes: list, optional
        :param descrips: Description to use to identify the different datasets, defaults to []
        :type descrips: list, optional
        :param num_to_load: Number of files to load from each dataset, defaults to 1000000
        :type num_to_load: int, optional
        :param configs: List of config dictionaries to use instead of the config file in each dataset 
        directory, defaults to None
        :type configs: list of dict, optional
        :raises Exception: If the length of data_dirs and file_prefixes do not match
        :raises Exception: If the length of data_dirs and descrips do not match
        :raises Exception: If the length of data_dirs and num_to_load do not match
        :raises Exception: If the length of data_dirs and configs do not match
        """
        if isinstance(data_dirs,str):
            data_dirs = [data_dirs]
        self.data_dirs = np.array(data_dirs)
        if not isinstance(file_prefixes,str):
            if len(file_prefixes) != len(data_dirs):
                raise Exception('Error: length of data_dirs and file_prefixes do not match.')
        else:
            file_prefixes = [file_prefixes]*len(data_dirs)
        if not isinstance(descrips,str):
            if len(descrips) != len(data_dirs):
                raise Exception('Error: length of data_dirs and descrips do not match.')
        else:
            descrips = [descrips]*len(data_dirs)
        self.file_prefixes = np.array(file_prefixes)
        self.descrips = np.array(descrips)
        if type(num_to_load) is not list:
            num_to_load = [num_to_load]*len(data_dirs)
        else:
            if len(num_to_load) != len(data_dirs):
                raise Exception('Error: length of data_dirs and num_to_load do not match.')
        if configs is not None:
            if type(configs) is not list:
                configs = [configs]*len(data_dirs)
            else:
                if len(configs) != len(data_dirs):
                    raise Exception('Error: length of data_dirs and configs do not match. ')
        self.configs = configs
        if len(file_suffixes)>0:
            if not isinstance(file_suffixes[0], str):
                if len(file_suffixes[0]) != len(data_dirs):
                    raise Exception('Error: length of data_dirs and file_suffixes do not match.')
            else:
                file_suffixes = [file_suffixes]*len(data_dirs)
        else:
            file_suffixes = [file_suffixes]*len(data_dirs)
        self.file_suffixes = file_suffixes
        # anything added here should be properly handled in merge_objects() and load_from_hdf5()
        self.num_to_load = np.array(num_to_load)
        self.num_files = np.array([])
        self.file_list = np.array([])
        self.diagonalize = False
        self.noise_subtracted = False
        self.p0_bead = np.array(())
        self.diam_bead = np.array(())
        self.mass_bead = np.array(())
        self.qpd_diag_mats = np.array(())
        self.naive_mat = np.array(((1.,1.,-1.,-1.),\
                                   (1.,-1.,1.,-1.),\
                                   (1.,-1.,-1.,1.),\
                                   (1.,1.,1.,1.)))
        self.file_data_objs = []
        self.bin_indices = np.array(())
        self.agg_dict = {}
        self.cant_bins_x = np.array((0,))
        self.cant_bins_z = np.array((0,))
        self.bad_files = np.array([])
        self.error_logs = np.array([])
        self.qpd_asds = np.array(())
        self.xypd_asds = np.array(())
        self.signal_models = []
        self.lightweight = True
        self.het_phase = False
        self.first_index = first_index


    def __get_file_list(self,no_config=False):
        """Gets a list of all file paths given the directories and prefixes specified
        when the object was created and sets it as an object attribute.

        :param no_config: If true, do not look for a config file, defaults to False
        :type no_config: bool, optional
        :raises Exception: If a config file is not found in the directory and no_config is False
        """
        file_list = []
        num_files = []
        p0_bead = []
        diam_bead = []
        mass_bead = []
        qpd_diag_mats = []
        configs = []
        for i,dir in enumerate(self.data_dirs):
            # get the bead position wrt the stage for each directory
            if not no_config:
                if self.configs is None:
                    try:
                        with open(dir+'/config.yaml','r') as infile:
                            config = yaml.safe_load(infile)
                            p0_bead.append(config['p0_bead'])
                            diam_bead.append(config['diam_bead'])
                            mass_bead.append(config['mass_bead'])
                            if 'qpd_diag_mat' in list(config.keys()):
                                qpd_diag_mats.append(config['qpd_diag_mat'])
                            else:
                                qpd_diag_mats.append(self.naive_mat)
                            configs.append(config)
                    except FileNotFoundError:
                        raise Exception('Error: config file not found in directory.')
                else:
                    p0_bead.append(self.configs[i]['p0_bead'])
                    diam_bead.append(self.configs[i]['diam_bead'])
                    mass_bead.append(self.configs[i]['mass_bead'])
                    if 'qpd_diag_mat' in list(self.configs[i].keys()):
                        qpd_diag_mats.append(self.configs[i]['qpd_diag_mat'])
                    else:
                        qpd_diag_mats.append(self.naive_mat)
            else:
                p0_bead.append([0,0,0])
                diam_bead.append(0)
                mass_bead.append(0)
                qpd_diag_mats.append(self.naive_mat)
            files = os.listdir(str(dir))
            # only add files, not folders, and ensure they end with .h5 and have the correct prefix
            files = [str(dir)+'/'+f for f in files if (os.path.isfile(str(dir)+'/'+f) and \
                     (self.file_prefixes[i] in f and f.endswith('.h5') and \
                     ((f[len(self.file_prefixes[i]):-3] in self.file_suffixes[i]) or len(self.file_suffixes[i])==0)))]
            files.sort(key=get_file_number)
            files = files[self.first_index:]
            num_to_load = min(self.num_to_load[i],len(files))
            file_list += files[:num_to_load]
            # keep track of the number of files loaded for each directory
            num_files.append(num_to_load)
        self.file_list = np.array(file_list)
        self.p0_bead = np.array(p0_bead)
        self.diam_bead = np.array(diam_bead)
        self.mass_bead = np.array(mass_bead)
        self.qpd_diag_mats = np.array(qpd_diag_mats)
        self.num_files = np.array(num_files)
        if self.configs is None:
            self.configs = np.array(configs)
        self.__bin_by_config_data()


    def load_file_data(self,num_cores=30,diagonalize_qpd=False,load_templates=False,harms=[],\
                       max_freq=500.,downsample=True,wiener=[False,True,False,False,False],\
                       time_domain=False,no_tf=False,force_cal_factors=[],no_config=False,\
                       ml_model=None,window=None,lightweight=True,het_phase=False):
        """Creates a FileData object for each of the files in the file list and loads
        in the relevant data for physics analysis.

        :param num_cores: Number of CPU cores to use when loading the data, defaults to 30
        :type num_cores: int, optional
        :param diagonalize_qpd: Calibrate the x and y data streams using the bead eigenmodes, defaults to False
        :type diagonalize_qpd: bool, optional
        :param load_templates: Load signal templates, defaults to False
        :type load_templates: bool, optional
        :param harms: Which harmonics of the drive to use, defaults to []
        :type harms: list, optional
        :param downsample: Downsample the data, defaults to True
        :type downsample: bool, optional
        :param wiener: Apply Wiener filter noise subtraction to the QPD x, y, and z and XYPD x and y data streams, 
        defaults to [False,True,False,False,False]
        :type wiener: list, optional
        :param force_cal_factors: Calibration factors to use if transfer function correction is not applied, defaults to []
        :type force_cal_factors: list, optional
        :param no_config: Whether to look for a config file in the data directories, defaults to False
        :type no_config: bool, optional
        :param ml_model: Motion-likeness model to apply to the x and y data, defaults to None
        :type ml_model: callable, optional
        :param lightweight: Drop the raw data that is not typically used in the analysis after loading, defaults to True
        :type lightweight: bool, optional
        """
        if load_templates:
            if not len(self.signal_models):
                print('Signal model not loaded! Load a signal model first. Aborting.')
                return
            signal_models = self.signal_models
        else:
            self.__get_file_list(no_config=no_config)
            signal_models = [None]*len(self.diam_bead)
        if diagonalize_qpd:
            qpd_diag_mats = self.qpd_diag_mats
            self.diagonalize = True
        else:
            qpd_diag_mats = [None]*len(self.qpd_diag_mats)
        if downsample and any(wiener):
            self.noise_subtracted = True
        if het_phase:
            self.het_phase = True
        print('Loading data from {} files...'.format(len(self.file_list)))
        file_data_objs = Parallel(n_jobs=num_cores)(delayed(self.process_file)\
                                                    (file_path,qpd_diag_mats[self.bin_indices[i,4]],\
                                                     signal_models[self.bin_indices[i,0]],ml_model,\
                                                     self.p0_bead[self.bin_indices[i,2]],\
                                                     self.mass_bead[self.bin_indices[i,1]],\
                                                     harms,max_freq,downsample,wiener,time_domain,\
                                                     no_tf,force_cal_factors,window,lightweight,het_phase) \
                                                     for i,file_path in enumerate(tqdm(self.file_list)))
        # record which files are bad and save the error logs
        error_logs = []
        bad_files = []
        for i,file_data_obj in enumerate(file_data_objs):
            if file_data_obj.is_bad == True:
                self.bin_indices[i,-1] = 1
                bad_files.append(file_data_obj.file_name)
                error_logs.append(file_data_obj.error_log)
            elif file_data_obj.is_bad == False and file_data_obj.error_log != '':
                error_logs.append(file_data_obj.error_log)
        self.file_data_objs = file_data_objs
        self.bad_files = np.array(bad_files)
        self.error_logs = np.array(error_logs)
        self.file_suffixes = np.array([[]*len(self.data_dirs)])
        # edit the number of files per directory to account for bad files
        for i in range(len(self.bad_files)):
            root_dir = '/'.join(self.bad_files[i].split('/')[:-1])
            dir_ind = np.argwhere(root_dir==self.data_dirs)[0][0]
            self.num_files[dir_ind] -= 1
        # delete the reference to the data or else the data will all be kept in memory
        # later after being copied to the dictionary as numpy arrays. this is
        # necessary to avoid memory errors when loading large AggregateData objects
        del file_data_objs
        # remove the bad files from all relevant class variables
        self.__purge_bad_files()
        if len(self.bad_files):
            print('Warning: {} files could not be loaded.'.format(len(self.bad_files)))
        if len(self.error_logs):
            print('Warning: {} files raised warnings during loading. First error log:'.format(len(self.error_logs)))
            print(self.error_logs[0])
        # if no files are loaded correctly, print an error log to indicate why
        if len(self.file_list)==0:
            print('Error: no file data loaded correctly! First error log:')
            print(self.error_logs[0])
            return
        print('Successfully loaded {} files.'.format(len(self.file_list)))
        self.__build_dict(lightweight=lightweight)
        self.lightweight = lightweight


    def load_yukawa_model(self, lambda_range=[1e-6,1e-4], num_lambdas=None, attractor='gold', signal_path=None):
        """Loads functions used to make Yukawa-modified gravity templates

        :param lambda_range: Range of lambda values in meters, defaults to [1e-6,1e-4]
        :type lambda_range: list, optional
        :param num_lambdas: Number of lambda values to use, defaults to None
        :type num_lambdas: int, optional
        :param attractor: Which attractor (gold or pt_black) to use
        :type attractor: str, optional
        """
        self.__get_file_list()
        suffix = 'master' if attractor=='gold' else 'ptblack'
        signal_models = []
        for diam in self.diam_bead:
            if str(diam)[0]=='7':
                signal_path = '/data/new_trap_processed/signal_templates/yukawa_' + attractor + \
                              '_attractor/7_6um-gbead_1um-unit-cells_' + suffix + '/'
            elif str(diam)[0]=='1':
                signal_path = '/data/new_trap_processed/signal_templates/yukawa_' + attractor + \
                              '_attractor/10um-gbead_1um-unit-cells_' + suffix + '/'
            else:
                print('Error: no signal model availabe for bead diameter {} um and {} attractor.'.format(diam, attractor))
                break
            signal_models.append(s.GravFuncs(signal_path))
            signal_models[-1].load_force_funcs(lambda_range=lambda_range,num_lambdas=num_lambdas)
            print('Yukawa-modified gravity signal model loaded for {} um bead and {} attractor.'.format(diam, attractor))
        self.signal_models = signal_models


    def load_powerlaw_model(self, N_val=2, r0_range=[1e-6,1e-4], num_r0s=None, attractor='gold'):
        """Loads functions used to make power-law-modified gravity templates. Since this
        parameterization is so similar to the Yukawa model, the same functions to build those
        force templates are used for this model as well.

        :param r0_range: Range of r0 values in meters, defaults to [1e-6,1e-4]
        :type r0_range: list, optional
        :param num_r0s: Number of r_0 values to use, defaults to None
        :type num_r_0s: int, optional
        """
        self.__get_file_list()
        signal_models = []
        for diam in self.diam_bead:
            if str(diam)[0]=='7':
                signal_path = '/data/new_trap_processed/signal_templates/powerlaw_' + attractor + \
                              '_attractor/7_6um-gbead/'
            elif str(diam)[0]=='1':
                signal_path = '/data/new_trap_processed/signal_templates/powerlaw_' + attractor + \
                              '_attractor/10um-gbead/'
            else:
                print('Error: no signal model availabe for bead diameter {} um and {} attractor.'.format(diam, attractor))
                break
            signal_models.append(s.GravFuncs(signal_path))
            signal_models[-1].load_force_funcs(lambda_range=r0_range, num_lambdas=num_r0s, N_val=N_val)
            print('Power-law-modified gravity signal model loaded for {} um bead and {} attractor.'.format(diam, attractor))
        self.signal_models = signal_models


    def process_file(self,file_path,qpd_diag_mat=None,signal_model=None,ml_model=None,p0_bead=None,\
                     mass_bead=0,harms=[],max_freq=2500.,downsample=True,wiener=[False,True,False,False,False],\
                     time_domain=False,no_tf=False,force_cal_factors=[],window=None,lightweight=True,het_phase=False):
        """Processes data for an individual file and returns the FileData object.

        :param file_path: Path to the raw HDF5 file to be loaded
        :type file_path: str
        :param qpd_diag_mat: Matrix used to transform QPD quadrants into x and y motion, defaults to None
        :type qpd_diag_mat: numpy.ndarray, optional
        :param signal_model: Signal model to be tested, defaults to None
        :type signal_model: SignalModel, optional
        :param ml_model: Motion-likeness model to be applied to the QPD x and y data, defaults to None
        :type ml_model: callable, optional
        :param p0_bead: Position of the bead, defaults to None
        :type p0_bead: list, optional
        :param harms: Harmonics of the cantilever drive to use, defaults to []
        :type harms: list, optional
        :param max_freq: The maximum frequency to keep in the data, defaults to 2500. 
        :type max_freq: float, optional
        :param downsample: Downsample the data, defaults to True
        :type downsample: bool, optional
        :param wiener: a list that specifies which signal streams the noise data should be subtracted from, 
        defaults to [False, True, False, False, False]
        :type wiener: list of bools, optional
        :param no_tf: Don't apply the transfer function calibration, defaults to False
        :type no_tf: bool, optional
        :param force_cal_factors: List of calibration factors for x, y, and z to use if `no_tf` is True, defaults to []
        :type force_cal_factors: list, optional
        :param lightweight: Drop some data after loading, defaults to False
        :type lightweight: bool, optional
        :return: The FileData object containing the processed data
        :rtype: optlevanalysis.data_processing.FileData
        """
        this_file = FileData(file_path)
        try:
            this_file.load_data(qpd_diag_mat=qpd_diag_mat,signal_model=signal_model,ml_model=ml_model,\
                                p0_bead=p0_bead,mass_bead=mass_bead,harms=harms,downsample=downsample,wiener=wiener,\
                                time_domain=time_domain,max_freq=max_freq,no_tf=no_tf,force_cal_factors=force_cal_factors,\
                                window=window,lightweight=lightweight,het_phase=het_phase)
        except Exception as e:
            this_file.is_bad = True
            this_file.error_log = repr(e)
        return this_file
    

    def __build_dict(self,lightweight=True):
        """Builds a dict containing the relevant data from each FileData object to
        make indexing the data easier.

        :param lightweight: Drop some data after loading, defaults to False
        :type lightweight: bool, optional
        """

        print('Building dictionary of file data...')
        agg_dict = {}

        # data that is common to all datasets is added only once
        agg_dict['freqs'] = np.array(self.file_data_objs[0].freqs)
        agg_dict['fsamp'] = int(self.file_data_objs[0].fsamp)
        agg_dict['nsamp'] = int(self.file_data_objs[0].nsamp)
        agg_dict['window_s1'] = self.file_data_objs[0].window_s1
        agg_dict['window_s2'] = self.file_data_objs[0].window_s2
        agg_dict['fund_ind'] = int(self.file_data_objs[0].fund_ind)
        agg_dict['good_inds'] = np.array(self.file_data_objs[0].good_inds)

        # data that is unique to each file goes in agg_dict
        times = []
        timestamp = []
        accelerometer = []
        microphones = []
        bead_height = []
        mean_laser_power = []
        laser_power_full = []
        mean_p_trans = []
        p_trans_full = []
        cant_raw_data = []
        quad_raw_data = []
        quad_null = []
        mean_cant_pos = []
        qpd_ffts = []
        qpd_ffts_full = []
        qpd_sb_ffts = []
        qpd_dc_offsets = []
        qpd_sums = []
        xypd_ffts = []
        xypd_ffts_full = []
        xypd_dc_offsets = []
        xypd_sb_ffts = []
        template_ffts = []
        template_params = []
        cant_fft = []
        quad_amps = []
        quad_phases = []
        mot_likes = []
        axis_angles = []
        camera_status = []
        is_noise = []

        for i,f in enumerate(self.file_data_objs):
            timestamp.append(f.times[0]*1e-9)
            times.append(f.times)
            accelerometer.append(f.accelerometer)
            microphones.append(f.microphone)
            bead_height.append(f.bead_height)
            mean_laser_power.append(f.mean_laser_power)
            laser_power_full.append(f.laser_power_full)
            mean_p_trans.append(f.mean_p_trans)
            p_trans_full.append(f.p_trans_full)
            cant_raw_data.append(f.cant_raw_data)
            quad_raw_data.append(f.quad_raw_data)
            quad_null.append(f.quad_null)
            mean_cant_pos.append(f.mean_cant_pos)
            qpd_ffts.append(f.qpd_ffts)
            qpd_ffts_full.append(f.qpd_ffts_full)
            qpd_sb_ffts.append(f.qpd_sb_ffts)
            qpd_dc_offsets.append(f.qpd_dc_offsets)
            xypd_ffts.append(f.xypd_ffts)
            xypd_ffts_full.append(f.xypd_ffts_full)
            xypd_sb_ffts.append(f.xypd_sb_ffts)
            xypd_dc_offsets.append(f.xypd_dc_offsets)
            qpd_sums.append(f.qpd_sum)
            template_ffts.append(f.template_ffts)
            template_params.append(f.template_params)
            cant_fft.append(f.cant_fft)
            quad_amps.append(f.quad_amps)
            quad_phases.append(f.quad_phases)
            mot_likes.append(f.motion_likeness)
            axis_angles.append(np.arccos(np.dot(f.qpd_diag_mat[0]*f.force_cal_factors_qpd[0],\
                                                f.qpd_diag_mat[1]*f.force_cal_factors_qpd[1])\
                                                /(np.linalg.norm(f.qpd_diag_mat[0]*f.force_cal_factors_qpd[0])\
                                                *np.linalg.norm(f.qpd_diag_mat[1]*f.force_cal_factors_qpd[1]))))
            camera_status.append(f.camera_status)
            is_noise.append(f.is_noise)
            if lightweight:
                # delete the object once data has been extracted
                self.file_data_objs[i] = FileData()

        if lightweight:
            del self.file_data_objs
            self.file_data_objs = []

        # convert lists to numpy arrays
        times = np.array(times)
        timestamp = np.array(timestamp)
        accelerometer = np.array(accelerometer)
        microphones = np.array(microphones)
        bead_height = np.array(bead_height)
        mean_laser_power = np.array(mean_laser_power)
        laser_power_full = np.array(laser_power_full)
        mean_p_trans = np.array(mean_p_trans)
        p_trans_full = np.array(p_trans_full)
        cant_raw_data = np.array(cant_raw_data)
        quad_raw_data = np.array(quad_raw_data)
        quad_null = np.array(quad_null)
        mean_cant_pos = np.array(mean_cant_pos)
        qpd_ffts = np.array(qpd_ffts)
        qpd_ffts_full = np.array(qpd_ffts_full)
        qpd_sb_ffts = np.array(qpd_sb_ffts)
        qpd_dc_offsets = np.array(qpd_dc_offsets)
        xypd_ffts = np.array(xypd_ffts)
        xypd_ffts_full = np.array(xypd_ffts_full)
        xypd_sb_ffts = np.array(xypd_sb_ffts)
        xypd_dc_offsets = np.array(xypd_dc_offsets)
        qpd_sums = np.array(qpd_sums)
        template_ffts = np.array(template_ffts)
        template_params = np.array(template_params)
        cant_fft = np.array(cant_fft)
        quad_amps = np.array(quad_amps)
        quad_phases = np.array(quad_phases)
        mot_likes = np.array(mot_likes)
        axis_angles = np.array(axis_angles)
        camera_status = np.array(camera_status)
        is_noise = np.array(is_noise)

        # add numpy arrays to the dictionary
        agg_dict['times'] = times
        agg_dict['timestamp'] = timestamp
        agg_dict['accelerometer'] = accelerometer
        agg_dict['microphones'] = microphones
        agg_dict['bead_height'] = bead_height
        agg_dict['mean_laser_power'] = mean_laser_power
        agg_dict['laser_power_full'] = laser_power_full
        agg_dict['mean_p_trans'] = mean_p_trans
        agg_dict['p_trans_full'] = p_trans_full
        agg_dict['cant_raw_data'] = cant_raw_data
        agg_dict['quad_raw_data'] = quad_raw_data
        agg_dict['quad_null'] = quad_null
        agg_dict['mean_cant_pos'] = mean_cant_pos
        agg_dict['qpd_ffts'] = qpd_ffts
        agg_dict['qpd_ffts_full'] = qpd_ffts_full
        agg_dict['qpd_sb_ffts'] = qpd_sb_ffts
        agg_dict['qpd_dc_offsets'] = qpd_dc_offsets
        agg_dict['xypd_ffts'] = xypd_ffts
        agg_dict['xypd_ffts_full'] = xypd_ffts_full
        agg_dict['xypd_sb_ffts'] = xypd_sb_ffts
        agg_dict['xypd_dc_offsets'] = xypd_dc_offsets
        agg_dict['qpd_sums'] = qpd_sums
        agg_dict['template_ffts'] = template_ffts
        agg_dict['template_params'] = template_params
        agg_dict['cant_fft'] = cant_fft
        agg_dict['quad_amps'] = quad_amps
        agg_dict['quad_phases'] = quad_phases
        agg_dict['mot_likes'] = mot_likes
        agg_dict['axis_angles'] = axis_angles
        agg_dict['camera_status'] = camera_status
        agg_dict['is_noise'] = is_noise

        self.agg_dict = agg_dict
        print('Done building dictionary.')


    def __bin_by_config_data(self):
        """Matches the data from the config file (p0_bead, diam_bead) to the data by assigning the index
        of the correct value in the p0_bead and diam_bead arrays. Should be called automatically when
        files are first loaded, but never by the user.
        """
        # initialize the list of bin indices
        self.bin_indices = np.zeros((len(self.file_list),11)).astype(np.int32)

        # first bin by diam_bead, p0_bead, and descrips, basically already done when the data was read in
        for i in range(len(self.num_files)):
            lower_ind = sum(self.num_files[0:i])
            upper_ind = lower_ind + self.num_files[i]
            self.bin_indices[lower_ind:upper_ind,0] = i
            self.bin_indices[lower_ind:upper_ind,1] = i
            self.bin_indices[lower_ind:upper_ind,2] = i
            self.bin_indices[lower_ind:upper_ind,3] = i
            self.bin_indices[lower_ind:upper_ind,4] = i

        # now remove any duplicates and fix the corresponding entries in the bin_indices array
        self.__remove_duplicate_bead_params()
    

    def __remove_duplicate_bead_params(self):
        """Removes duplicate p0_bead and diam_bead entries that may have resulted from loading
        in files from multiple directories corresponding to the same bead parameters. Fixes
        the corresponding values in the bin_indices array. This way all 7um data can be called
        with a single index, rather than finding all indices for the multiple instances of 7um 
        data loaded in. Should only be called after loading or merging objects, not by the user.
        """

        # get unique diam_bead elements and their indices
        diam_beads,diam_bead_inds = np.unique(self.diam_bead,axis=0,return_index=True)
        # sort the array of unique elements to be in the same order as the original array
        diam_beads = diam_beads[diam_bead_inds.argsort()]
        # then make a new array of indices corresponding to the sorted array of unique elements
        diam_bead_inds = list(range(len(diam_beads)))
        # now go through the unique elements sequentially and set the correct index wherever that
        # element appears
        for diam_bead,diam_bead_ind in zip(diam_beads,diam_bead_inds):
            # reset the values in the indices array to the index of the first unique element
            self.bin_indices[:,0][self.diam_bead[self.bin_indices[:,0]]==diam_bead] = diam_bead_ind

        # then set the object attribute to the duplicate-free version
        self.diam_bead = diam_beads

        # same thing for mass_bead
        mass_beads,mass_bead_inds = np.unique(self.mass_bead,axis=0,return_index=True)
        mass_beads = mass_beads[mass_bead_inds.argsort()]
        mass_bead_inds = list(range(len(mass_beads)))
        for mass_bead,mass_bead_ind in zip(mass_beads,mass_bead_inds):
            self.bin_indices[:,1][self.mass_bead[self.bin_indices[:,1]]==mass_bead] = mass_bead_ind
        self.mass_bead = mass_beads
        
        # same thing for p0_bead, but comparison is now across rows rather than single elements since p0 is 3d
        p0_beads,p0_bead_inds = np.unique(self.p0_bead,axis=0,return_index=True)
        p0_beads = p0_beads[p0_bead_inds.argsort()]
        p0_bead_inds = list(range(len(p0_beads)))
        for p0_bead,p0_bead_ind in zip(p0_beads,p0_bead_inds):
            self.bin_indices[:,2][np.all(self.p0_bead[self.bin_indices[:,2]]==p0_bead,axis=1)] = p0_bead_ind

        # again, set to duplicate-free version
        self.p0_bead = p0_beads

        # same thing for descrips
        descrips,descrip_inds = np.unique(self.descrips,axis=0,return_index=True)
        descrips = descrips[descrip_inds.argsort()]
        descrip_inds = list(range(len(descrips)))
        for descrip,descrip_ind in zip(descrips,descrip_inds):
            # reset the values in the indices array to the index of the first unique element
            self.bin_indices[:,3][self.descrips[self.bin_indices[:,3]]==descrip] = descrip_ind

        # then set the object attribute to the duplicate-free version
        self.descrips = descrips

        # same thing for qpd_diag_mats
        mats,mat_inds = np.unique(np.array(self.qpd_diag_mats),axis=0,return_index=True)
        mats = mats[mat_inds.argsort()]
        mat_inds = list(range(len(mats)))
        for mat,mat_ind in zip(mats,mat_inds):
            # reset the values in the indices array to the index of the first unique element
            self.bin_indices[:,4][np.all(self.qpd_diag_mats[self.bin_indices[:,4]]==mat,axis=(1,2))] = mat_ind

        # then set the object attribute to the duplicate-free version
        self.qpd_diag_mats = mats


    def __purge_bad_files(self):
        """Makes a list of the file names that couldn't be loaded, then removes the FileData objects
        and other relevant object attributes.
        """
        bad_file_indices = np.copy(self.bin_indices[:,-1]).astype(bool)
        self.file_list = np.delete(self.file_list,bad_file_indices,axis=0)
        self.file_data_objs = list(np.delete(self.file_data_objs,bad_file_indices,axis=0))
        self.bin_indices = np.delete(self.bin_indices,bad_file_indices,axis=0)


    def bin_by_aux_data(self,cant_bin_widths=[1.,1.],accel_thresh=0.1,bias_bins=0):
        """Bins the data by some auxiliary data along a number of axes. Sets an object attribute
        containing a list of indices that can be used to specify into which bin of each parameter
        a file falls.
        bin_widths = [x_width_microns, z_width_microns]

        :param cant_bin_widths: Width of the bins of cantilever position data in x and y in microns, defaults to [1.,1.]
        :type cant_bin_widths: list, optional
        :param accel_thresh: Accelerometer threshold above which a file is discarded, defaults to 0.1
        :type accel_thresh: float, optional
        :param bias_bins: Number of attractor bias bins to use. Currently not implemented, defaults to 0
        :type bias_bins: int, optional
        """
        print('Binning data by mean cantilever position and accelerometer data...')

        # add accelerometer and bias
        
        # p0_bead and diam_bead are done when data is loaded. Here, binning is done in
        # cantilever x, cantilever z, and bias. Each row of bin_indices is of the format
        # [diam_ind, mass_ind, p0_bead_ind, descrips, diag_mat_ind, cant_x_ind, cant_z_ind, \
        # accelerometer_ind, bias_ind, spec_ind, is_bad]

        # first create bins in x and z given the bin widths provided
        x_lower_edge = min(self.agg_dict['mean_cant_pos'][:,0]) - cant_bin_widths[0]/2.
        x_upper_edge = max(self.agg_dict['mean_cant_pos'][:,0]) + cant_bin_widths[0]/2.
        x_bins = np.arange(x_lower_edge, x_upper_edge + cant_bin_widths[0], cant_bin_widths[0])

        z_lower_edge = min(self.agg_dict['mean_cant_pos'][:,2]) - cant_bin_widths[1]/2.
        z_upper_edge = max(self.agg_dict['mean_cant_pos'][:,2]) + cant_bin_widths[1]/2.
        z_bins = np.arange(z_lower_edge, z_upper_edge + cant_bin_widths[1], cant_bin_widths[1])

        # position associated to each bin will be the bin center
        bin_centers_x = (x_bins[:-1] + x_bins[1:])/2.
        bin_centers_z = (z_bins[:-1] + z_bins[1:])/2.

        # now for each file, record which bin it falls into
        x_index = np.zeros(self.bin_indices.shape[0])
        z_index = np.zeros(self.bin_indices.shape[0])
        for i in range(len(x_bins)-1):
            x_index[(self.agg_dict['mean_cant_pos'][:,0] > x_bins[i]) &\
                     (self.agg_dict['mean_cant_pos'][:,0] <= x_bins[i+1])] = i
        
        for i in range(len(z_bins)-1):
            z_index[(self.agg_dict['mean_cant_pos'][:,2] > z_bins[i]) &\
                    (self.agg_dict['mean_cant_pos'][:,2] <= z_bins[i+1])] = i

        # find which bin indices actually appear in x_index, indicating that at
        # least one file falls into the corresponding bin. Get the corresponding
        # indices. Reorder the unique bins based on the indices so that new ascending
        # indices can be created without messing up ordering.
        nonempty_x_bins,nonempty_x_inds = np.unique(x_index,return_index=True)
        nonempty_x_bins = nonempty_x_bins[nonempty_x_inds.argsort()]
        # now set cant bin positions to the bin centers
        cant_bins_x = bin_centers_x[nonempty_x_bins.astype(np.int32)]
        # new index array corresponding to array above
        nonempty_x_inds = list(range(len(nonempty_x_bins)))
        bin_inds_x = np.zeros(self.bin_indices.shape[0])
        # populate a new array with the new index saying which of the bins
        # the file falls into
        for nonempty_x_bin,nonempty_x_ind in zip(nonempty_x_bins,nonempty_x_inds):
            bin_inds_x[x_index==nonempty_x_bin] = nonempty_x_ind
        
        # repeat for z
        nonempty_z_bins,nonempty_z_inds = np.unique(z_index,return_index=True)
        nonempty_z_bins = nonempty_z_bins[nonempty_z_inds.argsort()]
        cant_bins_z = bin_centers_z[nonempty_z_bins.astype(np.int32)]
        nonempty_z_inds = list(range(len(nonempty_z_bins)))
        bin_inds_z = np.zeros(self.bin_indices.shape[0])
        for nonempty_z_bin,nonempty_z_ind in zip(nonempty_z_bins,nonempty_z_inds):
            bin_inds_z[x_index==nonempty_z_bin] = nonempty_z_ind

        # pass the newly found cant bin indices to the bin_indices array
        self.bin_indices[:,5] = bin_inds_x
        self.bin_indices[:,6] = bin_inds_z
        
        # set centers of nonzero bins as a class attribute
        self.cant_bins_x = cant_bins_x
        self.cant_bins_z = cant_bins_z

        # if any accelerometer axis is over the threshold, set the bin index for that file to 1
        self.bin_indices[:,7] = np.array([np.max(np.abs(np.mean(self.agg_dict['accelerometer'],\
                                                        axis=2)),axis=1)>accel_thresh]).astype(np.int32)
        print('Done binning data.')


    def get_parameter_arrays(self):
        """Returns arrays of the config data and auxiliary data for use in indexing files within the
        AggregateData object. This should be updated as more bins are added to bin_indices.

        :return: The arrays of parameters with the same length as the number of files in the object
        :rtype: tuple of numpy.ndarray
        """

        # define all parameters to be returned here
        labels = ['diam_bead', 'mass_bead','p0_bead', 'descrips', 'qpd_diag_mats', 'cant_bins_x', 'cant_bins_z']
        print('Returning a tuple of the following arrays:')
        print('('+', '.join(labels)+')')

        # then make sure they are indexed properly here
        diams = self.diam_bead[self.bin_indices[:,0]]
        masses = self.mass_bead[self.bin_indices[:,1]]
        p0s = self.p0_bead[self.bin_indices[:,2]]
        descrips = self.descrips[self.bin_indices[:,3]]
        diag_mats = self.qpd_diag_mats[self.bin_indices[:,4]]
        cant_xs = self.cant_bins_x[self.bin_indices[:,5]]
        cant_zs = self.cant_bins_z[self.bin_indices[:,6]]

        # return the results
        return diams,masses,p0s,descrips,diag_mats,cant_xs,cant_zs
        

    def get_slice_indices(self,diam_bead=-1.,descrip='',cant_x=[-1e4,1e4],cant_z=[-1e4,1e4],accel_veto=False):
        """Returns a single list of indices corresponding to the positions of files that pass the
        cuts given by the index array.

        :param diam_bead: Bead diameter in microns. All diameters are accepted if not given, defaults to -1.
        :type diam_bead: float, optional
        :param descrip: Description assigned to datasets when loaded, defaults to ''
        :type descrip: str, optional
        :param cant_x: Cantilever x position range in microns, defaults to [-1e4,1e4]
        :type cant_x: list, optional
        :param cant_z: Cantilever z position range in microns, defaults to [-1e4,1e4]
        :type cant_z: list, optional
        :param accel_veto: Whether to veto based on accelerometer readings, defaults to False
        :type accel_veto: bool, optional
        :raises Exception: If no data is found for the bead diameter specified
        :raises Exception: If no data is found for the description specified
        :raises Exception: If the cantilever x position range is not given in the correct format
        :raises Exception: If no bins are found for the given range of cantilever x position
        :raises Exception: If the cantilever z position range is not given in the correct format
        :raises Exception: If no bins are found for the given range of cantilever z position
        :raises Exception: If no data are found matching the specified cuts
        :return: The list of indices corresponding to the files that pass the cuts
        :rtype: list
        """
        # p0_bead shoudln't need to be specified since it can be identified by descrips if necessary
        # same with mass_bead
        # still need to add binning for bias

        # create a list containing the indices corresponding to the bead diameters selected,
        # handling inputs as either a list or a int/float
        if type(diam_bead) is list:
            diam_bead_inds = []
            for diam in diam_bead:
                diam_bead_ind = np.where(np.isclose(self.diam_bead,diam))[0]
                if not len(diam_bead_ind):
                    raise Exception('Error: no data for bead diameter {} um!'.format(diam))
                diam_bead_inds.append(diam_bead_ind[0])
        elif diam_bead<0:
            diam_bead_inds = list(range(len(self.diam_bead)))
        else:
            diam_bead_inds = np.where(np.isclose(self.diam_bead,diam_bead))[0]
        if not len(diam_bead_inds):
            raise Exception('Error: no data for bead diameter {} um!'.format(diam_bead))

        # create a list containing the indices corresponding to the descrips selected, again
        # taking either a list or a string
        if type(descrip) is list:
            descrip_inds = []
            for des in descrip:
                descrip_ind = np.where(des==self.descrips)[0]
                if not len(descrip_ind):
                    raise Exception('Error: no data for description "'+des+'"!')
                descrip_inds.append(descrip_ind[0])
        elif descrip=='':
            descrip_inds = list(range(len(self.descrips)))
        else:
            descrip_inds = np.where(descrip==self.descrips)[0]
        if not len(descrip_inds):
            raise Exception('Error: no data for description "'+descrip+'"!')
        
        # create a list containing the indices corresponding to the range of cantilever x
        # positions specified. Input should be a list of the form [lower_val, upper_val]
        if type(cant_x) is list:
            if (len(cant_x)==2):
                if (cant_x[1]>cant_x[0]):
                    bad_format = False
                    cant_x_inds = np.where((self.cant_bins_x > cant_x[0]) & (self.cant_bins_x < cant_x[1]))[0]
                else:
                    bad_format = True
            else:
                bad_format = True
        else:
            bad_format = True
        if bad_format:
            raise Exception('Error: cantilever x position range must be given in the format [lower, upper]!')
        if not len(cant_x_inds):
            raise Exception('Error: no bins found for the given range of cantilever x position!')

        # same for cantilever z
        if type(cant_z) is list:
            if (len(cant_z)==2):
                if (cant_z[1]>cant_z[0]):
                    bad_format = False
                    cant_z_inds = np.where((self.cant_bins_z > cant_z[0]) & (self.cant_bins_z < cant_z[1]))[0]
                else:
                    bad_format = True
            else:
                bad_format = True
        else:
            bad_format = True
        if bad_format:
            raise Exception('Error: cantilever z position range must be given in the format [lower, upper]!')
        if not len(cant_z_inds):
            raise Exception('Error: no bins found for the given range of cantilever z position!')
        
        if accel_veto:
            accel_inds = [0]
        else:
            accel_inds = [0,1]

        # get inds for anything else
        p0_bead_inds = list(range(len(self.p0_bead)))
        mass_inds = list(range(len(self.mass_bead)))
        mat_inds = list(range(len(self.qpd_diag_mats)))
        # amplitude spectral density indices. should be degenerate with all others
        asd_inds = list(range(max(self.bin_indices[:,9]+1)))
        
        # get all possible combinations of the indices found above
        index_arrays = np.array([i for i in product(diam_bead_inds,mass_inds,p0_bead_inds,descrip_inds,mat_inds,\
                                                    cant_x_inds,cant_z_inds,accel_inds,[0],asd_inds,[0])])

        # needs to be updated to take a range of acceptable indices for any given dimension
        # then we will get a list of index_arrays and loop through doing this step to append
        # the indices for that iteration to the complete indices list. Then the indices list
        # should be sorted and then returned.
        slice_indices = []
        for index_array in index_arrays:
            slice_inds = np.where([np.all(self.bin_indices[i,:]==index_array) \
                                      for i in range(self.bin_indices.shape[0])])[0]
            if len(slice_inds):
                slice_indices += list(slice_inds)

        if not len(slice_indices):
            print('Warning: no data matching specified cuts!')
            
        return list(np.sort(slice_indices))
    

    def estimate_spectral_densities(self):
        """Computes the amplitude spectral density for each axis and sensor, averaging over datasets
        with the same conditions as defined by bin_indices.
        """

        print('Estimating RMS amplitude spectral density for each set of run conditions...')

        if self.agg_dict['qpd_ffts_full'].shape[-1]==1:
            print('Error: full FFTs already dropped from the AggregateData object!')
            return

        # get all unique combinations of run conditions
        index_arrays = np.unique(self.bin_indices,axis=0)

        # get sampling parameters in order to compute ASD from the DFTs
        fft_to_asd = self.agg_dict['window_s1']/np.sqrt(2.*self.agg_dict['fsamp']*self.agg_dict['window_s2'])

        # initialize lists to store the spectra
        qpd_asds = []
        xypd_asds = []

        # get a list of indices for each unique set of run conditions
        for ind,index_array in enumerate(index_arrays):
            unique_inds = np.where([np.all(self.bin_indices[i,:]==index_array) \
                                      for i in range(self.bin_indices.shape[0])])[0]

            # average the PSDs to get the final ASD
            qpd_asd_x = np.sqrt(np.mean(np.abs(self.agg_dict['qpd_ffts_full'][unique_inds][:,0,:]*fft_to_asd)**2,axis=0))
            qpd_asd_y = np.sqrt(np.mean(np.abs(self.agg_dict['qpd_ffts_full'][unique_inds][:,1,:]*fft_to_asd)**2,axis=0))
            qpd_asd_z = np.sqrt(np.mean(np.abs(self.agg_dict['qpd_ffts_full'][unique_inds][:,2,:]*fft_to_asd)**2,axis=0))
            xypd_asd_x = np.sqrt(np.mean(np.abs(self.agg_dict['xypd_ffts_full'][unique_inds][:,0,:]*fft_to_asd)**2,axis=0))
            xypd_asd_y = np.sqrt(np.mean(np.abs(self.agg_dict['xypd_ffts_full'][unique_inds][:,1,:]*fft_to_asd)**2,axis=0))

            qpd_asds.append(np.array((qpd_asd_x,qpd_asd_y,qpd_asd_z)))
            xypd_asds.append(np.array((xypd_asd_x,xypd_asd_y)))

            # update the bin indices
            self.bin_indices[:,9][unique_inds] = ind

        # add as class attributes
        self.qpd_asds = np.array(qpd_asds)
        self.xypd_asds = np.array(xypd_asds)

        print('Amplitude spectral densities estimated for {} sets of run conditions.'.format(self.qpd_asds.shape[0]))


    def subtract_coherent_noise(self, file_inds=None, acc_channels=[2], plot=False):
        """Subtracts noise measured in the accelerometer from the QPD data streams, down to the
        coherence limit. This should only be called if the `wiener` argument was set to False
        for all channels when the data was loaded.
        """

        print('Subtracting coherent portion of accelerometer noise from QPD data...')

        if self.noise_subtracted:
            print('Error: coherent noise already subtracted from the QPD data!')
            return
        
        if file_inds is None:
            file_inds = list(range(self.bin_indices.shape[0]))
        
        # get the accelerometer data and take the DFT
        accels = self.agg_dict['accelerometer'][file_inds]
        if np.all(accels==0):
            print('Error: no accelerometer data found!')
            return
        accels = (accels - np.mean(accels,axis=-1,keepdims=True))
        win = sig.get_window(('tukey',0.05),accels.shape[-1])
        accel_ffts = np.fft.rfft(win[np.newaxis,np.newaxis,:]*accels,axis=-1)[:,:,:len(self.agg_dict['freqs'])]

        # ffts already calibrated and scaled to the correct units
        qpd_ffts = self.agg_dict['qpd_ffts_full'][file_inds]
        qpd_data = np.fft.irfft(qpd_ffts, accels.shape[-1], axis=-1)[:,:3,:]
        qpd_null = np.fft.irfft(qpd_ffts, accels.shape[-1], axis=-1)[:,3,:]

        # separate the QPD data into x, y, and null channels
        qpd_ffts_x = qpd_ffts[:,0,:]
        qpd_ffts_y = qpd_ffts[:,1,:]
        qpd_ffts_z = qpd_ffts[:,2,:]
        qpd_ffts_n = qpd_ffts[:,3,:]

        # loop through all accelerometer channels
        for i in range(3):
            if i not in acc_channels:
                continue
            print('Subtracting noise from accelerometer channel {}...'.format(i))
            # estimate the cross spectral density between the accelerometer and each QPD channel using all datasets
            _, pxa = sig.csd(qpd_data[:,0,:].flatten(),accels[:,i,:].flatten(),fs=self.agg_dict['fsamp'],\
                             window=win,nperseg=accels.shape[-1],noverlap=None)
            _, pya = sig.csd(qpd_data[:,1,:].flatten(),accels[:,i,:].flatten(),fs=self.agg_dict['fsamp'],\
                             window=win,nperseg=accels.shape[-1],noverlap=None)
            _, pza = sig.csd(qpd_data[:,2,:].flatten(),accels[:,i,:].flatten(),fs=self.agg_dict['fsamp'],\
                             window=win,nperseg=accels.shape[-1],noverlap=None)
            _, pna = sig.csd(qpd_null.flatten(),accels[:,i,:].flatten(),fs=self.agg_dict['fsamp'],\
                             window=win,nperseg=accels.shape[-1],noverlap=None)
            
            # estimate the accelerometer power spectral density using all datasets
            _,paa = sig.welch(accels[:,i,:].flatten(),fs=self.agg_dict['fsamp'],window=win,\
                              nperseg=accels.shape[-1],noverlap=None)
            # avoid division by zero
            paa[paa==0] = np.inf

            # build the transfer functions
            tf_x = pxa/paa
            tf_y = pya/paa
            tf_z = pza/paa
            tf_n = pna/paa

            if plot:
                fig, ax = plt.subplots(2, figsize=(7,9), sharex=True)
                ax[0].loglog(_, np.abs(tf_z), lw=1)
                ax[1].plot(_, np.angle(tf_z)*180./np.pi, lw=1, ls='none', marker='o', ms=2, fillstyle='none')
                ax[0].set_xlim([0,500])
                ax[1].set_ylim([-180,180])
                ax[0].grid(which='both')
                ax[1].grid(which='both')
                ax[1].set_xlabel('Frequency [Hz]')
                ax[0].set_ylabel('Magnitude [$z$ counts/$a$ count]')
                ax[1].set_ylabel('Phase [$^\circ$]')
                fig.suptitle('Transfer function from accelerometer ${}$ to QPD $z$'.format(['x', 'y', 'z'][i]))

            # subtract off the coherent noise from the ffts
            qpd_ffts_x -= tf_x[np.newaxis,:len(self.agg_dict['freqs'])]*accel_ffts[:,i,:]
            qpd_ffts_y -= tf_y[np.newaxis,:len(self.agg_dict['freqs'])]*accel_ffts[:,i,:]
            qpd_ffts_z -= tf_z[np.newaxis,:len(self.agg_dict['freqs'])]*accel_ffts[:,i,:]
            qpd_ffts_n -= tf_n[np.newaxis,:len(self.agg_dict['freqs'])]*accel_ffts[:,i,:]

        # update the ffts in the dictionary
        qpd_ffts[:,0,:] = qpd_ffts_x
        qpd_ffts[:,1,:] = qpd_ffts_y
        qpd_ffts[:,2,:] = qpd_ffts_z
        qpd_ffts[:,3,:] = qpd_ffts_n
        self.agg_dict['qpd_ffts_full'][file_inds] = qpd_ffts
        self.agg_dict['qpd_ffts'][file_inds] = qpd_ffts[:,:,self.agg_dict['good_inds']]

        # change the flag to indicate that coherent noise has been subtracted
        self.noise_subtracted = True

        print('Finished subtracting noise.')
    

    def drop_full_ffts(self):
        """Drops the complete spectra for each file from the dictionary to save memory. To be called
        after the data has been used for basic plotting.
        """

        # keep the first two dimensions the same to avoid screwing up indexing
        self.agg_dict['qpd_ffts_full'] = np.empty_like(self.agg_dict['qpd_ffts_full'][:,:,0])[...,np.newaxis]
        self.agg_dict['xypd_ffts_full'] = np.empty_like(self.agg_dict['xypd_ffts_full'][:,:,0])[...,np.newaxis]


    def diagonalize_qpd(self,fit_inds=None,peak_guess=[400.,370.],width_guess=[10.,10.],plot=False):
        """Fits the two resonant peaks in x and y and diagonalizes the QPD position sensing
        to remove cross-coupling from one into the other. Returns a matrix that can be used
        by the FileData class to extract x and y from the raw data.

        :param fit_inds: Indices of the files to use for the fit, defaults to None
        :type fit_inds: numpy.ndarray , optional
        :param peak_guess: Estimate of the resonant frequencies in x and y in Hz, defaults to [400.,370.]
        :type peak_guess: list, optional
        :param width_guess: Estimates of the resonant peak widths in x and y in Hz, defaults to [10.,10.]
        :type width_guess: list, optional
        :param plot: Whether to plot the result, defaults to False
        :type plot: bool, optional
        :return: Matrix that can be used to diagonalize the position sensing data
        :rtype: numpy.ndarray
        """

        # if no argument is provided, just use all the files
        if fit_inds is None:
            fit_inds = np.array(range(len(self.file_list)))

        # use the guess of the peak location as a starting point for fitting
        freq_ranges = [peak_guess[0]-2*width_guess[0],peak_guess[0]+2*width_guess[0],\
                       peak_guess[1]-2*width_guess[1],peak_guess[1]+2*width_guess[1]]

        # get corresponding indices
        freqs = self.agg_dict['freqs']
        freq_inds = []
        for f in freq_ranges:
            freq_inds.append(np.argmin(np.abs(freqs-f)))

        fft_to_asd = self.agg_dict['window_s1']/np.sqrt(2.*self.agg_dict['fsamp']*self.agg_dict['window_s2'])

        # get RMS of the ffts for the data used for the fit
        mean_ffts_x = np.sqrt(np.mean(np.abs(np.fft.rfft(self.agg_dict['quad_raw_data'][fit_inds][:,0,:])\
                                             *fft_to_asd)**2,axis=0))
        mean_ffts_y = np.sqrt(np.mean(np.abs(np.fft.rfft(self.agg_dict['quad_raw_data'][fit_inds][:,1,:])\
                                             *fft_to_asd)**2,axis=0))

        # resonant frequency in y is lower
        max_ind_x = freq_inds[0] + np.argmax(mean_ffts_x[freq_inds[0]:freq_inds[1]])
        max_ind_y = freq_inds[2] + np.argmax(mean_ffts_y[freq_inds[2]:freq_inds[3]])

        # fit a lorentzian to the peaks for better peak location finding
        try:
            p_x,_ = curve_fit(lor,freqs[freq_inds[0]:freq_inds[1]],mean_ffts_x[freq_inds[0]:freq_inds[1]],\
                            p0=[freqs[max_ind_x],width_guess[0],1e-3])
            max_ind_x = np.argmin(np.abs(freqs-p_x[0]))
            p_y,_ = curve_fit(lor,freqs[freq_inds[2]:freq_inds[3]],mean_ffts_y[freq_inds[2]:freq_inds[3]],\
                            p0=[freqs[max_ind_y],width_guess[1],1e-3])
            max_ind_y = np.argmin(np.abs(freqs-p_y[0]))
            failed = False
        except RuntimeError:
            print('Error: peak fitting failed!')
            max_ind_x = np.argmin(np.abs(freqs-peak_guess[0]))
            max_ind_y = np.argmin(np.abs(freqs-peak_guess[1]))
            p_x = [freqs[max_ind_x],width_guess[0],1e-3]
            p_y = [freqs[max_ind_y],width_guess[1],1e-3]
            failed = True
        max_ind_n = np.argmin(np.abs(freqs-2000.))

        # get the raw data from each quadrant
        raw_qpd_1 = np.array(self.agg_dict['quad_amps'][fit_inds][:,0,:],dtype=np.float64)
        raw_qpd_2 = np.array(self.agg_dict['quad_amps'][fit_inds][:,1,:],dtype=np.float64)
        raw_qpd_3 = np.array(self.agg_dict['quad_amps'][fit_inds][:,2,:],dtype=np.float64)
        raw_qpd_4 = np.array(self.agg_dict['quad_amps'][fit_inds][:,3,:],dtype=np.float64)

        # normalize by the total at each timestep
        tot_at_time = np.sum(self.agg_dict['quad_amps'][fit_inds][:,:4,:],axis=1)
        if (tot_at_time<0).any():
            print('Warning: potential overflow in sum of quad amps!')
        raw_qpd_1 = raw_qpd_1/tot_at_time
        raw_qpd_2 = raw_qpd_2/tot_at_time
        raw_qpd_3 = raw_qpd_3/tot_at_time
        raw_qpd_4 = raw_qpd_4/tot_at_time

        # get the ffts and phase shift them relative to quadrant 1
        fft_qpd_1_all = np.fft.rfft(raw_qpd_1)
        phase_qpd_1 = np.angle(fft_qpd_1_all)
        fft_qpd_1 = np.mean(fft_qpd_1_all*np.exp(-1j*phase_qpd_1)*2./self.agg_dict['window_s1'],axis=0)[:len(freqs)]
        fft_qpd_2 = np.mean(np.fft.rfft(raw_qpd_2)*np.exp(-1j*phase_qpd_1)*2./self.agg_dict['window_s1'],axis=0)[:len(freqs)]
        fft_qpd_3 = np.mean(np.fft.rfft(raw_qpd_3)*np.exp(-1j*phase_qpd_1)*2./self.agg_dict['window_s1'],axis=0)[:len(freqs)]
        fft_qpd_4 = np.mean(np.fft.rfft(raw_qpd_4)*np.exp(-1j*phase_qpd_1)*2./self.agg_dict['window_s1'],axis=0)[:len(freqs)]

        if plot:
            fig,ax = plt.subplots(2,1,sharex=True)
            ax[0].semilogy(freqs,np.abs(fft_qpd_1)*fft_to_asd,alpha=0.65,label='Q1')
            ax[0].semilogy(freqs,np.abs(fft_qpd_2)*fft_to_asd,alpha=0.65,label='Q2')
            ax[0].semilogy(freqs,np.abs(fft_qpd_3)*fft_to_asd,alpha=0.65,label='Q3')
            ax[0].semilogy(freqs,np.abs(fft_qpd_4)*fft_to_asd,alpha=0.65,label='Q4')
            ax[0].axvline(freqs[max_ind_x],color='k',alpha=0.8,ls='--',lw=1)
            ax[0].axvline(freqs[max_ind_y],color='k',alpha=0.8,ls='--',lw=1)
            ax[1].plot(freqs,np.angle(fft_qpd_1)*180./np.pi,'.',ms='2',label='Q1')
            ax[1].plot(freqs,np.angle(fft_qpd_2)*180./np.pi,'.',ms='2',label='Q2')
            ax[1].plot(freqs,np.angle(fft_qpd_3)*180./np.pi,'.',ms='2',label='Q3')
            ax[1].plot(freqs,np.angle(fft_qpd_4)*180./np.pi,'.',ms='2',label='Q4')
            ax[1].axvline(freqs[max_ind_x],color='k',alpha=0.8,ls='--',lw=1)
            ax[1].axvline(freqs[max_ind_y],color='k',alpha=0.8,ls='--',lw=1)
            ax[0].set_title('Response of individual quadrants')
            ax[1].set_xlabel('Frequency [Hz]')
            ax[1].set_xlim([min(freq_ranges)-40,max(freq_ranges)+40])
            ax[0].set_ylabel('ASD [arb/$\sqrt{\mathrm{Hz}}$]')
            ax[0].set_ylim([1e-6,1e-3 ])
            ax[1].set_ylabel('Phase [degrees]')
            ax[1].set_ylim([-200,200])
            ax[1].set_yticks([-180,0,180])
            ax[0].legend(ncol=2)
            ax[0].grid(which='both')
            ax[1].legend(ncol=2)
            ax[1].grid(which='both')

        # bins on each side to average over for smoothing
        df = 0
        qpd_1x = np.mean(fft_qpd_1[max_ind_x-df:max_ind_x+df+1])
        qpd_2x = np.mean(fft_qpd_2[max_ind_x-df:max_ind_x+df+1])
        qpd_3x = np.mean(fft_qpd_3[max_ind_x-df:max_ind_x+df+1])
        qpd_4x = np.mean(fft_qpd_4[max_ind_x-df:max_ind_x+df+1])
        qpd_1y = np.mean(fft_qpd_1[max_ind_y-df:max_ind_y+df+1])
        qpd_2y = np.mean(fft_qpd_2[max_ind_y-df:max_ind_y+df+1])
        qpd_3y = np.mean(fft_qpd_3[max_ind_y-df:max_ind_y+df+1])
        qpd_4y = np.mean(fft_qpd_4[max_ind_y-df:max_ind_y+df+1])
        qpd_1n = np.mean(fft_qpd_1[max_ind_n-df:max_ind_n+df+1])
        qpd_2n = np.mean(fft_qpd_2[max_ind_n-df:max_ind_n+df+1])
        qpd_3n = np.mean(fft_qpd_3[max_ind_n-df:max_ind_n+df+1])
        qpd_4n = np.mean(fft_qpd_4[max_ind_n-df:max_ind_n+df+1])

        # 4x2 matrix, rows are quadrants and columns are f_x and f_y
        signal_mat = np.array(((qpd_1x,qpd_1y),
                               (qpd_2x,qpd_2y),
                               (qpd_3x,qpd_3y),
                               (qpd_4x,qpd_4y)))

        # for simplicity drop the imaginary parts. Power should be negligible, but check anyway
        print('Fraction of power in imaginary parts to be discarded:')
        print('Q1: {:.3f}, Q2: {:.3f}, Q3: {:.3f}, Q4: {:.3f}'\
              .format(*(np.sum(np.imag(signal_mat)**2,axis=1)/np.sum(np.abs(signal_mat)**2,axis=1)).flatten()))
        signal_mat = np.real(signal_mat)

        # compute the left inverse, which gives the transformation to pure x and y, then normalize
        mode_vecs = np.linalg.solve(signal_mat.T.dot(signal_mat),signal_mat.T)
        mode_vecs /= np.linalg.norm(mode_vecs)

        # find the null vectors, which we expect to span the scattered-light-only space
        light_vecs = null_space(mode_vecs).T

        # make the full transformation matrix
        diag_mat = np.vstack([mode_vecs,light_vecs])

        # scale the result to try to keep the amplitude comparable before and after
        naive_mat = np.array(((1.,1.,-1.,-1.),\
                              (1.,-1.,1.,-1.),\
                              (1.,-1.,-1.,1.),\
                              (0,0,0,0)))
        signal_null = np.array(((qpd_1n,qpd_1n),
                                (qpd_2n,qpd_2n),
                                (qpd_3n,qpd_3n),
                                (qpd_4n,qpd_4n)))
        signal_null = np.real(signal_null)
        signal_full = np.hstack((signal_mat,signal_null))
        total_naive = np.sqrt(np.sum(np.abs(np.matmul(naive_mat,signal_full))**2,axis=0))
        total_diag = np.sqrt(np.sum(np.abs(np.matmul(diag_mat,signal_full))**2,axis=0))
        scale_fac = np.mean(total_naive/total_diag)
        diag_mat = scale_fac*diag_mat

        if plot:
            cross_coupling(self.agg_dict,diag_mat,p_x=p_x,p_y=p_y,plot_inds=fit_inds,plot_null=True)
            visual_diag_mat(diag_mat)

        if not failed:
            print('Diagonalization complete!')
            print('Angle between x and y: {:.1f} degrees'\
                  .format(np.arccos(np.dot(diag_mat[0],diag_mat[1])/(np.linalg.norm(diag_mat[0])*np.linalg.norm(diag_mat[1])))*180./np.pi))
            print('Copy the following into the config.yaml files for the relevant datasets:')
            print('qpd_diag_mat: [[{:.8f},{:.8f},{:.8f},{:.8f}],[{:.8f},{:.8f},{:.8f},{:.8f}],[{:.8f},{:.8f},{:.8f},{:.8f}],[{:.8f},{:.8f},{:.8f},{:.8f}]]'\
                .format(*[i for i in diag_mat.flatten()]))
        return diag_mat


    def merge_objects(self,object_list):
        """Merges two AggregateData objects. First create a new object with no input arguments,
        then immediately call this function, passing in a list of the objects to be merged.
        The binning should then be done again to ensure the indices are set correctlly.

        :param object_list: List of AggregateData objects to merge
        :type object_list: list
        """
        print('Merging {} objects...'.format(len(object_list)))

        # consistency checks before merging
        ffts = []
        asds = []
        fsamps = []
        nsamps = []
        fund_inds = []
        good_inds = []
        lightweights = []
        diagonalize = []
        subtracted = []
        het_phases = []
        for object in object_list:
            ffts.append(object.agg_dict['qpd_ffts_full'].shape[-1]>2)
            asds.append(object.qpd_asds.shape[-1]>2)
            fsamps.append(object.agg_dict['fsamp'])
            nsamps.append(object.agg_dict['nsamp'])
            fund_inds.append(object.agg_dict['fund_ind'])
            good_inds.append(object.agg_dict['good_inds'])
            lightweights.append(object.lightweight)
            diagonalize.append(object.diagonalize)
            subtracted.append(object.noise_subtracted)
            het_phases.append(object.het_phase)
        if not (all(np.isclose(np.array(fsamps)-fsamps[0],0)) or \
                all(np.isclose(np.array(nsamps)-nsamps[0],0))):
            print('Error: inconsistent number of samples or sampling frequency between objects!')
            return
        if not (all(np.isclose(np.mean(good_inds,axis=0)-np.array(good_inds[0]),0)) or \
                all(np.isclose(np.array(fund_inds)-fund_inds[0],0))):
            print('Error: inconsistent fundamental frequency or harmonics between objects!')
            return
        if not all(np.array(lightweights)==lightweights[0]):
            print('Error: cannot merge a lightweight object with a full object!')
            return
        if not all(np.array(diagonalize)==diagonalize[0]):
            print('Error: cannot merge under different diagonalization conditions!')
            return
        if not all(np.array(subtracted)==subtracted[0]):
            print('Error: cannot merge under different noise subtraction conditions!')
            return
        if not all(np.array(het_phases)==het_phases[0]):
            print('Error: cannot merge heterodyne amplitude data with phase data!')
        merge_asds = True
        if not all(asds):
            print('Warning: not all objects have spectral density estimates. Skipping this attribute...')
            merge_asds = False
        if not all(ffts):
            print('Warning: not all objects have the full FFTs. Skipping this dictionary entry...')
            [object.drop_full_ffts() for object in object_list]
        
        # loop through and add in pairs
        while(len(object_list)>1):
            object1 = object_list[0]
            object2 = object_list[1]
            # just add them all together
            self.lightweight = object1.lightweight
            self.diagonalize = object1.diagonalize
            self.noise_subtracted = object1.noise_subtracted
            self.het_phase = object1.het_phase
            self.data_dirs = np.array(list(object1.data_dirs) + list(object2.data_dirs))
            self.file_prefixes = np.array(list(object1.file_prefixes) + list(object2.file_prefixes))
            self.file_suffixes = np.array(list(object1.file_suffixes) + list(object2.file_suffixes))
            self.descrips = np.array(list(object1.descrips) + list(object2.descrips))
            self.num_to_load = np.array(list(object1.num_to_load) + list(object2.num_to_load))
            self.num_files = np.array(list(object1.num_files) + list(object2.num_files))
            self.file_list = np.array(list(object1.file_list) + list(object2.file_list))
            self.p0_bead = np.array(list(object1.p0_bead) + list(object2.p0_bead))
            self.diam_bead = np.array(list(object1.diam_bead) + list(object2.diam_bead))
            self.mass_bead = np.array(list(object1.mass_bead) + list(object2.mass_bead))
            self.qpd_diag_mats = np.array(list(object1.qpd_diag_mats) + list(object2.qpd_diag_mats))
            self.configs = np.array(list(object1.configs) + list(object2.configs))
            self.bad_files = np.array(list(object1.bad_files) + list(object2.bad_files))
            self.error_logs = np.array(list(object1.error_logs) + list(object2.error_logs))
            self.file_data_objs = object1.file_data_objs + object2.file_data_objs
            # ensure that the indices in the second object are offset by the length
            # of the relevant columns in the first object. The remaining indices will be
            # recalculated and duplicates will be removed when bin_by_aux_data is next called
            row0 = np.zeros(object1.bin_indices.shape[1])
            row0[0] = len(list(object1.diam_bead))
            row0[1] = len(list(object1.mass_bead))
            row0[2] = len(list(object1.p0_bead))
            row0[3] = len(list(object1.descrips))
            offset = np.tile(row0,(object2.bin_indices.shape[0],1))
            self.bin_indices = np.concatenate((object1.bin_indices,object2.bin_indices+offset),axis=0).astype(np.int32)
            # the indices in the remaining columns are wrong once objects are added together. Just remove
            # the indices to avoid confusion. They'll be set again once bin_by_aux_data is called.
            self.bin_indices[:,5:] = 0
            # merge amplitude spectral densities if they are all present
            if merge_asds:
                self.qpd_asds = np.concatenate((object1.qpd_asds,object2.qpd_asds),axis=0)
                self.xypd_asds = np.concatenate((object1.xypd_asds,object2.xypd_asds),axis=0)
            # merge the agg_dicts
            agg_dict = {}
            keys_to_skip = ['freqs','good_inds']
            for k in object1.agg_dict.keys():
                if (len(np.shape(object1.agg_dict[k]))>0) and (k not in keys_to_skip):
                    # concatentate the arrays
                    agg_dict[k] = np.concatenate((object1.agg_dict[k],object2.agg_dict[k]))
                else:
                    # and just take the first of the scalars
                    agg_dict[k] = object1.agg_dict[k]
            self.agg_dict = agg_dict
            self.__remove_duplicate_bead_params()
            object_list = [deepcopy(self)] + object_list[2:]

        print('Objects merged successfully. Note that the binning by cantilever position and '\
              +'bias must be redone after merging.')


    def save_to_hdf5(self,path=''):
        """Saves the data in the AggregateData object to an HDF5 file.

        :param path: Path to the HDF5 file where the AggregateData object should be saved, defaults to ''
        :type path: str, optional
        """
        print('Saving AggregateData object...')

        # if no path is given, construct the default from the filename
        if path=='':
            path = '/data/new_trap_processed/aggdat/'+'/'.join(self.file_list[0].split('/')[3:-1]) + 'aggdat.h5'

        # make the folders if they don't already exist
        if not os.path.exists('/'.join(path.split('/')[:-1])):
            os.makedirs('/'.join(path.split('/')[:-1]))

        if len(self.file_data_objs):
            print('Warning: FileData objects cannot be saved to HDF5. Only saving AggregateData attributes.')
        
        with h5py.File(path, 'w') as f:
            # add the commit hash, creation date, and user as attributes
            short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
            f.attrs['git_revision_short_hash'] = short_hash
            f.attrs['creation_timestamp'] = time.time()
            f.attrs['creation_user'] = str(os.environ.get('USER'))
            # everything other than agg_dict goes in run_params
            f.create_group('run_params')
            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, dict):
                    # agg_dict gets its own group
                    dict_group = f.create_group(attr_name)
                    for key, value in attr_value.items():
                        # add a dataset for each column of the dictionary
                        dict_group.create_dataset(key, data=value)
                elif isinstance(attr_value, (float,int)):
                    # add floats and ints
                    f.create_dataset('run_params/'+attr_name, data=attr_value)
                elif isinstance(attr_value, np.ndarray):
                    # add numpy arrays of strings or anything else
                    if (attr_value.dtype.type is np.str_) or (attr_value.dtype.type is np.object_):
                        f.create_dataset('run_params/'+attr_name, data=np.array(attr_value,dtype='S'))
                    else:
                        f.create_dataset('run_params/'+attr_name, data=attr_value)

            print('AggregateData object saved to '+path)


    def load_from_hdf5(self,path):
        """Loads the AggregateData object from an HDF5 file.

        :param path: Path to the HDF5 file to be loaded
        :type path: str
        """
        print('Loading AggregateData object...')

        # add the current hash to the file attributes
        this_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

        with h5py.File(path, 'r') as f:

            # check for code version compatibility
            short_hash = f.attrs['git_revision_short_hash']
            if short_hash!=this_hash:
                print('Warning: attempting to load a file created with an earlier version of OptLevAnalysis.')
                print('To ensure compatibility, run "git checkout {}" prior to loading.'.format(short_hash))

            # load the dictionary data
            agg_dict = {}
            for ds_name in f['agg_dict'].keys():
                agg_dict[ds_name] = f['agg_dict/'+ds_name][()]
            self.agg_dict = agg_dict

            # load the run parameters
            self.lightweight = bool(np.array(f['run_params/lightweight']))
            self.diagonalize = bool(np.array(f['run_params/diagonalize']))
            self.noise_subtracted = bool(np.array(f['run_params/noise_subtracted']))
            self.het_phase = bool(np.array(f['run_params/het_phase']))
            confs = np.array(f['run_params/configs'],dtype=np.str_)
            self.configs = np.array([ast.literal_eval(conf) for conf in confs])
            self.data_dirs = np.array(f['run_params/data_dirs'],dtype=np.str_)
            self.file_prefixes = np.array(f['run_params/file_prefixes'],dtype=np.str_)
            self.file_suffixes = np.array(f['run_params/file_suffixes'],dtype=np.str_)
            self.descrips = np.array(f['run_params/descrips'],dtype=np.str_)
            self.file_list = np.array(f['run_params/file_list'],dtype=np.str_)
            self.bad_files = np.array(f['run_params/bad_files'],dtype=np.str_)
            self.error_logs = np.array(f['run_params/error_logs'],dtype=np.str_)
            self.p0_bead = np.array(f['run_params/p0_bead'])
            self.diam_bead = np.array(f['run_params/diam_bead'])
            self.mass_bead = np.array(f['run_params/mass_bead'])
            self.qpd_diag_mats = np.array(f['run_params/qpd_diag_mats'])
            self.num_to_load = np.array(f['run_params/num_to_load'])
            self.num_files = np.array(f['run_params/num_files'])
            self.bin_indices = np.array(f['run_params/bin_indices'])
            self.cant_bins_x = np.array(f['run_params/cant_bins_x'])
            self.cant_bins_z = np.array(f['run_params/cant_bins_z'])
            self.qpd_asds = np.array(f['run_params/qpd_asds'])
            self.xypd_asds = np.array(f['run_params/xypd_asds'])

            # fill empty attributes
            self.file_data_objs = []
            self.signal_models = []

        print('Loaded AggregateData object from '+path)


    def _print_attributes(self):
        """Debugging tool to ensure that all attributes and types are preserved
        through saving/loading/merging.
        """
        print('ATTRIBUTE :    TYPE')
        print('---------------------')
        for attr_name, attr_value in vars(self).items():
            print(attr_name,':    ',type(attr_value))

    
    def _print_errors(self):
        """Debugging tool to print error logs for any files that could not be loaded.
        """
        for fil,err in zip(self.bad_files,self.error_logs):
            print(fil)
            print(err)
            print()
