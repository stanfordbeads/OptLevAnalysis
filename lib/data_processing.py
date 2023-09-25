import numpy as np
import h5py
import os
import re
import pickle
import yaml
import scipy.interpolate as interp
import scipy.signal as signal
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
from copy import deepcopy
from funcs import *


class FileData:

    def __init__(self,path):
        '''
        Initializes a RawData object with some metadata attributes and a dict containing
        the raw data, while setting other attributes to default values.
        '''
        self.file_name = path
        self.data_dict = {}
        self.date = re.search(r"\d{8,}", self.file_name)[0]
        self.times = np.array(())
        self.amps = np.array(())
        self.phases = np.array(())
        self.seismometer = np.array(())
        self.fsamp = 0
        self.nsamp = 0
        self.freqs = np.array(())
        self.cant_raw_data = np.array(((),(),()))
        self.quad_raw_data = np.array(((),(),()))
        self.cant_pos_calibrated = np.array(((),(),()))
        self.mean_cant_pos = np.array(())
        self.bead_force_calibrated = np.array(((),(),()))
        self.good_inds = np.array(())
        self.drive_ind = 0
        self.bead_ffts = np.array(((),(),()))
        self.bead_ffts_full = np.array(((),(),()))
        self.sideband_ffts = np.array(((),(),()))
        self.drive_fft = np.array(())
        self.force_cal_factors = np.array([0,0,0])
        self.is_bad = False


    def load_data(self,step_cal_drive_freq=71.0,max_freq=2500.,num_harmonics=10,\
                     harms=[],width=0,noise_bins=10,lightweight=False):
        '''
        Applies calibrations to the cantilever data and QPD data, then gets FFTs for both.
        '''
        self.read_hdf5()
        self.fsamp = self.data_dict['fsamp']
        self.seismometer = self.data_dict['seismometer']
        self.get_xyz_from_quad()
        self.nsamp = len(self.times)
        self.freqs = np.fft.rfftfreq(self.nsamp, d=1.0/self.fsamp)
        self.calibrate_stage_position()
        if int(self.date) < 20230101:
            self.calibrate_bead_response('/data/new_trap_processed/calibrations/transfer_funcs/20200320.trans',\
                                         step_cal_drive_freq=step_cal_drive_freq,max_freq=max_freq)
        else:
            self.calibrate_bead_response(step_cal_drive_freq=step_cal_drive_freq,max_freq=max_freq)
        self.get_boolean_cant_filter(num_harmonics=num_harmonics,harms=harms,width=width,max_freq=max_freq)
        self.get_ffts_and_noise(noise_bins=noise_bins)
        # for use with AggregateData, don't carry around all the raw data
        if lightweight:
            self.data_dict = {}
            self.cant_raw_data = np.array(((),(),()))
            self.quad_raw_data = np.array(((),(),()))
            self.cant_pos_calibrated = np.array(((),(),()))
            self.bead_force_calibrated = np.array(((),(),()))


    def read_hdf5(self):
        '''
        Reads raw data and metadata from an hdf5 file directly into a dict.
        '''
        f = h5py.File(self.file_name, 'r')
        dd = {}
        dd['cant_data'] = np.array(f['cant_data'])
        dd['quad_data'] = np.array(f['quad_data'])
        try:
            dd['seismometer'] = np.array(f['seismometer'])
            dd['laser_power'] = np.array(f['laser_power'])
            dd['PSPD'] = np.array(f['PSPD'])
        except:
            dd['seismometer'] = np.zeros_like(dd['cant_data'][0])
            dd['laser_power'] = np.zeros_like(dd['cant_data'])
            dd['PSPD'] = np.zeros_like(dd['cant_data'])
        dd['timestamp_ns'] = os.stat(self.file_name).st_mtime*1e9
        dd['fsamp'] = f.attrs['Fsamp']/f.attrs['downsamp']
        dd['cantilever_axis'] = f.attrs['cantilever_axis']
        dd['cantilever_freq'] = f.attrs['cantilever_freq']
        cant_voltages = list(f['cantilever_settings'])
        dd['cantilever_DC'] = [cant_voltages[i] for i in [0,2,4]]
        dd['cantilever_amp'] = [cant_voltages[i] for i in [1,3,5]]
        self.data_dict = dd
    

    def extract_quad(self):
        '''
        De-interleave the quad_data to extract timestamp, amplitude, and phase data.
        Since the read request is asynchronous, the timestamp may not be the first entry.
        First step is to identify it, then all other values can be extracted based on the index
        of the first timestamp.
        '''

        # get the data and timestamp from the data_dict
        quad_data = self.data_dict['quad_data']
        timestamp_ns = self.data_dict['timestamp_ns']

        # within an hour should be good enough for now, but in future it could be calculated dynamically
        diff_thresh = 3600.*1e9

        # first timestamp should be in first 12 elements
        for ind in range(12):
            # try reconstructing a timestamp by making a 64-bit object from consecutive 32-bit objects
            ts_attempt = (np.uint32(quad_data[ind]).astype(np.uint64) << np.uint64(32)) + np.uint32(quad_data[ind+1])

            # if it is close to the timestamp from the hdf5 file metadata, we've found the first timestamp
            if(abs(ts_attempt - timestamp_ns) < diff_thresh):
                tind = ind
                break

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

        # return numpy arrays
        return quad_times, np.array(amps), np.array(phases)
    
    
    def calibrate_stage_position(self):
        '''
        Convert voltage in cant_data into microns. Returns a tuple of x,y,z
        '''

        # get the cantilever data from the data dict
        self.cant_raw_data = self.data_dict['cant_data']

        # could eventually move this elsewhere and reference a config file instead of hard-coded params
        cant_cal_func = lambda x: list(np.repeat([[50.418,50.418,10.0]],self.cant_raw_data.shape[1],axis=0).T*x\
              + np.repeat([[0.0766,0.0766,0]],self.cant_raw_data.shape[1],axis=0).T)

        # save calibrated data as attributes of FileData object
        self.cant_pos_calibrated = cant_cal_func(self.cant_raw_data)

        # also save the mean position, used for rough binning of datasets later
        self.mean_cant_pos = np.mean(self.cant_pos_calibrated,axis=1)
    
    
    def get_xyz_from_quad(self):
        '''
        Calculates x, y, and z from the quadrant photodiode amplitude and phase data.
        '''
        
        self.times,amps,phases = self.extract_quad()

        # mapping from cartesian plane to QPD indices:
        # I -> 0
        # II -> 2
        # III -> 3
        # IV -> 1

        # get top, bottom, left, right from raw quad data
        right = amps[0] + amps[1]
        left = amps[2] + amps[3]
        top = amps[0] + amps[2]
        bottom = amps[1] + amps[3]

        # get x and y from left/right and top/bottom
        x = right - left
        y = top - bottom

        # total light to normalize by
        quad_sum = right + left

        # set object attribute with a numpy array of x,y,z
        self.quad_raw_data = np.array([x.astype(np.float64)/quad_sum,y.astype(np.float64)/quad_sum,phases[4]])
    

    def calibrate_bead_response(self,tf_path=None,step_cal_drive_freq=71.0,max_freq=2500.):
        '''
        Apply correction using the transfer function to calibrate the
        x, y, and z responses.
        '''

        # for data from 2023 and later, the code will automatically find the transfer
        # function in the right format. For old data, specify the path manually
        if tf_path is None:
            Harr = self.tf_array_fitted(self.freqs,max_freq=max_freq)
        else:
            Harr = self.tf_array_interpolated(self.freqs,tf_path)

        # force calibration factor at driven frequency
        drive_freq_ind = np.argmin(np.abs(self.freqs - step_cal_drive_freq))
        response_matrix = Harr[drive_freq_ind,:,:]
        force_cal_factors = [0,0,0]
        for i in [0,1,2]:
            # assume the response is purely diagonal, and take from it the x, y, and z
            # force calibration factors
            force_cal_factors[i] = np.abs(response_matrix[i,i])
        self.force_cal_factors = force_cal_factors
        
        # calculate the DFT of the data, then correct using the transfer function matrix
        data_fft = np.fft.rfft(self.quad_raw_data)
        # matrix multiplication with index contraction made explicit
        # 'kj,ki' = matrix multiplication along second two indices (the 3x3 part)
        # output has one free index (j). '->ji' = output uncontracted indices in this order
        calibrated_fft = np.einsum('ikj,ki->ji', Harr, data_fft)

        # take care of nans from inverting transfer functions
        nan_inds = np.isnan(calibrated_fft)
        calibrated_fft[nan_inds] = 0.0+0.0j

        # inverse DFT to get the now-calibrated position data
        self.bead_force_calibrated = np.fft.irfft(calibrated_fft)


    def tf_array_fitted(self,freqs,max_freq=2500.):
        '''
        Get the transfer function array from the hdf5 file containing the fitted poles,
        zeros, and gain from the measured transfer functions along x, y, and z, and returns it.
        '''

        # transfer function data should be stored here in a folder named by the date
        tf_path = '/data/new_trap_processed/calibrations/transfer_funcs/'+str(self.date)+'/TF.h5'
        tf_file = h5py.File(tf_path,'r')

        ### Compute TF at frequencies of interest. Appropriately inverts
        ### so we can map response -> drive
        Harr = np.zeros((len(freqs), 3, 3), dtype=complex)
        Harr[:,0,0] = 1/signal.freqs_zpk(tf_file['fits/zXX'], 
            tf_file['fits/pXX'], tf_file['fits/kXX']/tf_file.attrs['scaleFactors'][0], 2*np.pi*freqs)[1]
        Harr[:,1,1] = 1/signal.freqs_zpk(tf_file['fits/zYY'], 
            tf_file['fits/pYY'], tf_file['fits/kYY']/tf_file.attrs['scaleFactors'][1], 2*np.pi*freqs)[1]
        Harr[:,2,2] = 1/signal.freqs_zpk(tf_file['fits/zZZ'], 
            tf_file['fits/pZZ'], tf_file['fits/kZZ']/tf_file.attrs['scaleFactors'][2], 2*np.pi*freqs)[1]
        tf_file.close()

        # Only apply the TF to frequencies below some frequency
        max_freq_ind = np.argmin( np.abs(freqs - max_freq) )
        Harr[max_freq_ind+1:,:,:] = 0.0+0.0j

        return Harr
    

    def tf_array_interpolated(self,freqs,tf_path=None,suppress_off_diag=False):
        '''
        Extracts the interpolated transfer function array from a .trans file and returns it.
        '''

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

        return Hout
    

    def build_drive_filter(self, drive_fft, freqs, num_harmonics=10, width=0, harms=[], max_freq=2500.):
        '''
        Identify the fundamental drive frequency and make an array of harmonics specified
        by the function arguments, then make a notch filter of the width specified around these harmonics.
        '''

        # index of maximum frequency
        max_ind = np.argmin(np.abs(freqs - max_freq))

        # find the drive frequency, ignoring the DC bin
        fund_ind = np.argmax(np.abs(drive_fft[1:max_ind])) + 1
        drive_freq = freqs[fund_ind]

        # filter is  initialized with 1 at the drive frequency and 0 elsewhere
        drive_filter = np.zeros(len(drive_fft))
        drive_filter[fund_ind] = 1.0

        # can make the notch filter wider than 1 bin with the 'width' argument
        if width:
            lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
            upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
            drive_filter[lower_ind:upper_ind+1] = 1.0

        # create default array of harmonics if an input is not provided
        if len(harms) == 0:
            harms = np.array([x+1 for x in range(num_harmonics)])
        # remove the fundamental frequency if 1 is not in the provided list of harmonics
        elif 1 not in harms:
            drive_filter[fund_ind] = 0.0
            if width:
                drive_filter[lower_ind:upper_ind+1] = 0.0

        # loop over harmonics and add them to the filter
        for n in harms:
            harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
            drive_filter[harm_ind] = 1.0 
            if width:
                h_lower_ind = np.argmin(np.abs(n * drive_freq - 0.5 * width - freqs))
                h_upper_ind = np.argmin(np.abs(n * drive_freq + 0.5 * width - freqs))
                drive_filter[h_lower_ind:h_upper_ind+1] = 1.0
        
        # a boolean array is ultimately needed so do that conversion here
        drive_filter = drive_filter > 0

        return drive_filter

    def get_boolean_cant_filter(self, num_harmonics=10, harms=[], width=0, max_freq=2500.):
        '''
        Build a boolean filter of a given width for the cantilever drive for the specified harnonics
        '''

        # driven axis is the one with the maximum amplitude of driving voltage
        drive_ind = np.argmax(self.data_dict['cantilever_amp'])

        # vector of calibrated cantilever positions along the driven axis
        drive_vec = self.cant_pos_calibrated[drive_ind]

        # fft of the cantilever position vector
        drive_fft = np.fft.rfft(drive_vec)

        # get the notch filter for the given harmonics, as well as the index of the fundamental
        # frequency and the drive frequency
        drive_filter = self.build_drive_filter(drive_fft, self.freqs, num_harmonics=num_harmonics, \
                                               harms=harms, width=width, max_freq=max_freq)

        # create array containing the indices of the values that survive the filter
        good_inds = np.arange(len(drive_filter)).astype(int)[drive_filter]

        # set indices as class attributes
        self.good_inds = good_inds
        self.drive_ind = drive_ind


    def get_ffts_and_noise(self, noise_bins=10):   
        '''
        Compute the fft of the x, y, and z data at each of the harmonics, and of the noise,
        and of some side-band frequencies.
        ''' 
        
        # frequency values at the specified harmonics
        harm_freqs = self.freqs[self.good_inds]

        # make sure it's an array even if theres only one harmonic given
        if type(harm_freqs) == np.float64:
            harm_freqs = np.array([harm_freqs])

        # initialize an array for the ffts of x, y, and z
        bead_ffts_full = np.zeros((3, len(self.freqs)), dtype=np.complex128)
        bead_ffts = np.zeros((3, len(self.good_inds)), dtype=np.complex128)
        sideband_ffts = np.zeros((3, len(self.good_inds)*noise_bins), dtype=np.complex128)

        # get the fft for the cantilever data along the driven axis
        drive_fft_full = np.fft.rfft(self.cant_pos_calibrated[self.drive_ind])

        # now select only the indices for the chosen harmonics
        drive_fft = drive_fft_full[self.good_inds]

        # get the QPD position data
        data = self.bead_force_calibrated

        # loop through the axes
        for resp in [0,1,2]:

            # get the fft for the given axis and multiply by the calibration factor
            bead_fft = np.fft.rfft(data[resp])

            # add the fft to the existing array, which was initialized with zeros
            bead_ffts_full[resp] += bead_fft
            bead_ffts[resp] += bead_fft[self.good_inds]

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
            sideband_ffts[resp] += bead_fft[sideband_inds]

        # normalize the ffts
        norm_factor = 2./self.nsamp
        bead_ffts_full *= norm_factor
        bead_ffts *= norm_factor
        drive_fft *= norm_factor
        sideband_ffts *= norm_factor

        # save the ffts as class attributes
        self.bead_ffts_full = bead_ffts_full
        self.bead_ffts = bead_ffts
        self.drive_fft = drive_fft
        self.sideband_ffts = sideband_ffts



class AggregateData:

    def __init__(self,data_dirs=[],file_prefixes=[],descrips=[],num_to_load=1e6):
        '''
        Takes a list of directories containing the files to be aggregated, and optionally
        a list of file prefixes. If given, the list of file prefixes should be the same length as
        the list of data directories. If files with multiple prefixes are required from the same
        directory, add the directory to the list multiple times with the corresponding prefixes
        in the file_prefixes argument.
        '''
        self.data_dirs = data_dirs
        if len(file_prefixes):
            if len(file_prefixes) != len(data_dirs):
                raise Exception('Error: length of data_dirs and file_prefixes do not match.')
        else:
            file_prefixes = ['']*len(data_dirs)
        if len(descrips):
            if len(descrips) != len(data_dirs):
                raise Exception('Error: length of data_dirs and descrips do not match.')
        else:
            descrips = ['']*len(data_dirs)
        self.file_prefixes = file_prefixes
        self.descrips = np.array(descrips)
        if type(num_to_load) is not list:
            self.num_to_load = [num_to_load]*len(data_dirs)
        else:
            if len(num_to_load) != len(data_dirs):
                raise Exception('Error: length of data_dirs and num_to_load do not match.')
            else:
                self.num_to_load = num_to_load
        self.num_files = []
        self.file_list = []
        self.p0_bead = np.array(())
        self.diam_bead = np.array(())
        self.seis_thresh = 1e3
        self.file_data_objs = []
        self.bin_indices = np.array(())
        self.agg_dict = {}
        self.cant_bins_x = np.array((0,))
        self.cant_bins_z = np.array((0,))
        self.bad_files = []


    def get_file_list(self):
        '''
        Get a list of all file paths given the directories and prefixes specified
        when the object was created and set it as an object attribute.
        '''
        file_list = []
        num_files = []
        p0_bead = []
        diam_bead = []
        for i,dir in enumerate(self.data_dirs):
            # get the bead position wrt the stage for each directory
            try:
                with open(dir+'/config.yaml','r') as infile:
                    config = yaml.safe_load(infile)
                    p0_bead.append(config['p0_bead'])
                    diam_bead.append(config['diam_bead'])
            except FileNotFoundError:
                raise Exception('Error: config file not found in directory.')
            files = os.listdir(dir)
            # only add files, not folders, and ensure they end with .h5 and have the correct prefix
            files = [dir+'/'+f for f in files if (os.path.isfile(dir+'/'+f) and (self.file_prefixes[i] in f and f.endswith('.h5')))]
            files.sort(key=get_file_number)
            num_to_load = min(self.num_to_load[i],len(files))
            file_list += files[:num_to_load]
            # keep track of the number of files loaded for each directory
            num_files.append(num_to_load)
        self.file_list = file_list
        self.p0_bead = np.array(p0_bead)
        self.diam_bead = np.array(diam_bead)
        self.num_files = num_files
        self.__bin_by_config_data()


    def load_file_data(self,num_cores=1,lightweight=True):
        '''
        Create a FileData object for each of the files in the file list and load
        in the relevant data for physics analysis.
        '''
        self.get_file_list()
        file_data_objs = Parallel(n_jobs=num_cores)(delayed(self.process_file)(file_path,lightweight) \
                                                      for file_path in tqdm(self.file_list))
        # record which files are bad in the self.bin_indices array
        for i,file_data_obj in enumerate(file_data_objs):
            if file_data_obj.is_bad == True:
                self.bin_indices[i,-1] = 1
        self.file_data_objs = file_data_objs
        # remove the bad files from all relevant class variables
        self.__purge_bad_files()
        

    def process_file(self,file_path,lightweight=True):
        '''
        Process data for an individual file and return the FileData object
        '''
        this_file = FileData(file_path)
        try:
            this_file.load_data(lightweight=lightweight)
        except:
            this_file.is_bad = True
        return this_file
    

    def build_dict(self):
        '''
        Build a dict containing the relevant data from each FileData object to
        make indexing the data easier.
        '''

        agg_dict = {}
        dates = []
        times = []
        seismometer = []
        cant_raw_data = []
        quad_raw_data = []
        mean_cant_pos = []
        freqs = []
        good_inds = []
        bead_ffts = []
        bead_ffts_full = []
        sideband_ffts = []
        drive_fft = []
        for f in self.file_data_objs:
            dates.append(f.date)
            times.append(f.times)
            seismometer.append(f.seismometer)
            cant_raw_data.append(f.cant_raw_data)
            quad_raw_data.append(f.quad_raw_data)
            mean_cant_pos.append(f.mean_cant_pos)
            good_inds.append(f.good_inds)
            freqs.append(f.freqs)
            bead_ffts.append(f.bead_ffts)
            bead_ffts_full.append(f.bead_ffts_full)
            sideband_ffts.append(f.sideband_ffts)
            drive_fft.append(f.drive_fft)
        agg_dict['dates'] = np.array(dates)
        agg_dict['times'] = np.array(times)
        agg_dict['seismometer'] = np.array(seismometer)
        agg_dict['cant_raw_data'] = np.array(cant_raw_data)
        agg_dict['quad_raw_data'] = np.array(quad_raw_data)
        agg_dict['mean_cant_pos'] = np.array(mean_cant_pos)
        agg_dict['freqs'] = np.array(freqs)
        agg_dict['good_inds'] = np.array(good_inds)
        agg_dict['bead_ffts'] = np.array(bead_ffts)
        agg_dict['bead_ffts_full'] = np.array(bead_ffts_full)
        agg_dict['sideband_ffts'] = np.array(sideband_ffts)
        agg_dict['drive_fft'] = np.array(drive_fft)
        self.agg_dict = agg_dict


    def __bin_by_config_data(self):
        '''
        Match the data from the config file (p0_bead, diam_bead) to the data by assigning the index
        of the correct value in the p0_bead and diam_bead arrays. Should be called automatically when
        files are first loaded, but never by the user.
        '''

        # initialize the list of bin indices
        self.bin_indices = np.zeros((len(self.file_list),8)).astype(np.int32)

        # first bin by diam_bead, p0_bead, and descrips, basically already done when the data was read in
        for i in range(len(self.num_files)):
            lower_ind = sum(self.num_files[0:i])
            upper_ind = lower_ind + self.num_files[i]
            self.bin_indices[lower_ind:upper_ind,0] = i
            self.bin_indices[lower_ind:upper_ind,1] = i
            self.bin_indices[lower_ind:upper_ind,2] = i

        # now remove any duplicates and fix the corresponding entries in the bin_indices array
        self.__remove_duplicate_bead_params()
    

    def __remove_duplicate_bead_params(self):
        '''
        Removes duplicate p0_bead and diam_bead entries that may have resulted from loading
        in files from multiple directories corresponding to the same bead parameters. Fixes
        the corresponding values in the bin_indices array. This way all 7um data can be called
        with a single index, rather than finding all indices for the multiple instances of 7um 
        data loaded in. Should only be called after loading or merging objects, not by the user.
        '''

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
        
        # same thing for p0_bead, but comparison is now across rows rather than single elements since p0 is 3d
        p0_beads,p0_bead_inds = np.unique(self.p0_bead,axis=0,return_index=True)
        p0_beads = p0_beads[p0_bead_inds.argsort()]
        p0_bead_inds = list(range(len(p0_beads)))
        for p0_bead,p0_bead_ind in zip(p0_beads,p0_bead_inds):
            self.bin_indices[:,1][np.all(self.p0_bead[self.bin_indices[:,1]]==p0_bead,axis=1)] = p0_bead_ind

        # again, set to duplicate-free version
        self.p0_bead = p0_beads

        # same thing for descrips
        descrips,descrip_inds = np.unique(self.descrips,axis=0,return_index=True)
        descrips = descrips[descrip_inds.argsort()]
        descrip_inds = list(range(len(descrips)))
        for descrip,descrip_ind in zip(descrips,descrip_inds):
            # reset the values in the indices array to the index of the first unique element
            self.bin_indices[:,2][self.descrips[self.bin_indices[:,2]]==descrip] = descrip_ind

        # then set the object attribute to the duplicate-free version
        self.descrips = descrips


    def __purge_bad_files(self):
        '''
        Make a list of the file names that couldn't be loaded, then remove the FileData objects
        and other relevant object attributes.
        '''
        bad_file_indices = np.copy(self.bin_indices[:,-1]).astype(bool)
        self.bad_files = list(np.array(self.file_list)[bad_file_indices])
        self.file_list = list(np.delete(self.file_list,bad_file_indices,axis=0))
        self.file_data_objs = list(np.delete(self.file_data_objs,bad_file_indices,axis=0))
        self.bin_indices = np.delete(self.bin_indices,bad_file_indices,axis=0)
        # success_fraction


    def bin_by_aux_data(self,cant_bin_widths=[1.,1.],seis_thresh=0.1,bias_bins=0):
        '''
        Bins the data by some auxiliary data along a number of axes. Sets an object attribute
        containing a list of indices that can be used to specify into which bin of each parameter
        a file falls.
        bin_widths = [x_width_microns, z_width_microns]
        '''
        # add seismometer and bias
        
        # p0_bead and diam_bead are done when data is loaded. Here, binning is done in
        # cantilever x, cantilever z, and bias. Each row of bin_indices is of the format
        # [diam_ind, p0_bead_ind, descrips, cant_x_ind, cant_z_ind, seismometer_ind, \
        # bias_ind, is_bad]

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
        self.bin_indices[:,3] = bin_inds_x
        self.bin_indices[:,4] = bin_inds_z
        
        # set centers of nonzero bins as a class attribute
        self.cant_bins_x = cant_bins_x
        self.cant_bins_z = cant_bins_z

        # add seismometer threshold as a class attribute, then update bin indices
        self.seis_thresh = seis_thresh
        self.bin_indices[:,5] = np.array([np.abs(np.mean(self.agg_dict['seismometer'],\
                                                         axis=1))>seis_thresh]).astype(np.int32)
        

    def get_slice_indices(self,diam_bead=-1.,descrip='',cant_x=[0.,1e4],cant_z=[0.,1e4],seis_veto=False):
        '''
        Returns a single list of indices corresponding to the positions of files that pass the
        cuts given by the index array.
        '''
        # p0_bead shoudln't need to be specified since it can be identified by descrips if necessary
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
            raise Exception('Error: cantilever x position range must be given in the format [lower, upper]!')
        if not len(cant_z_inds):
            raise Exception('Error: no bins found for the given range of cantilever x position!')
        
        if seis_veto:
            seis_inds = [0]
        else:
            seis_inds = [0,1]
        
        # get all possible combinations of the indices found above
        p0_bead_inds = list(range(len(self.p0_bead)))
        index_arrays = np.array([i for i in product(p0_bead_inds,diam_bead_inds,descrip_inds,\
                                                    cant_x_inds,cant_z_inds,seis_inds,[0],[0])])

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
    

    def merge_objects(self,object_list):
        '''
        Merge two AggregateData objects. First create a new object with no input arguments,
        then immediately call this function, passing in a list of the objects to be merged.
        The binning should then be done again to ensure the indices are set correctlly.
        '''
        while(len(object_list)>1):
            object1 = object_list[0]
            object2 = object_list[1]
            # just add them all together
            self.data_dirs = object1.data_dirs + object2.data_dirs
            self.file_prefixes = object1.file_prefixes + object2.file_prefixes
            self.descrips = np.array(list(object1.descrips) + list(object2.descrips))
            self.num_to_load = object1.num_to_load + object2.num_to_load
            self.num_files = object1.num_files + object2.num_files
            self.file_list = object1.file_list + object2.file_list
            self.p0_bead = np.array(list(object1.p0_bead) + list(object2.p0_bead))
            self.diam_bead = np.array(list(object1.diam_bead) + list(object2.diam_bead))
            self.file_data_objs = object1.file_data_objs + object2.file_data_objs
            # ensure that the indices in the second object are offset by the length
            # of the relevant columns in the first object. The remaining indices will be
            # recalculated and duplicates will be removed when bin_by_aux_data is next called
            row0 = np.zeros(object1.bin_indices.shape[1])
            row0[0] = len(list(object1.diam_bead))
            row0[1] = len(list(object1.p0_bead))
            row0[2] = len(list(object1.descrips))
            offset = np.tile(row0,(object2.bin_indices.shape[0],1))
            self.bin_indices = np.concatenate((object1.bin_indices,object2.bin_indices+offset),axis=0).astype(np.int32)
            # the indices in the remaining columns are wrong once objects are added together. Just remove
            # the indices to avoid confusion. They'll be set again once bin_by_aux_data is called.
            self.bin_indices[:,3:] = 0
            agg_dict = {}
            for k in object1.agg_dict.keys():
                agg_dict[k] = np.concatenate((object1.agg_dict[k],object2.agg_dict[k]))
            self.agg_dict = agg_dict
            self.__remove_duplicate_bead_params()
            object_list = [deepcopy(self)] + object_list[2:]

        print('Objects merged successfully. Note that the binning by cantilever position and'\
              +'bias must be redone after merging.')


    def save(self,path):
        '''
        Save the AggregateData object to the path specified.
        '''
        outfile = open(path,'wb')
        pickle.dump(self,outfile)
        outfile.close()
        print('AggregateData object saved to '+path)


    def load(self,path):
        '''
        Load an AggregateData object that was saved previously.
        '''
        infile = open(path,'rb')
        self.__dict__.update(pickle.load(infile).__dict__)
        infile.close()

