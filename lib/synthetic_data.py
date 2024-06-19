import numpy as np
from data_processing import FileData,AggregateData

class SynthFile(FileData):
    '''
    This child class of FileData inherits most of the methods but overrides a couple to allow for a time series
    of forces from a Yukawa-like interaction to be added to the time series. By default, it assumes that data
    being loaded is noise and therefore adds a sine wave to the cantilever y motion. If the synthetic signal is
    to be made for an already-moving cantilever, use the argument noise_only=False when loading data.
    '''

    def load_and_inject(self,alpha=1e7,lamb=10.,noise_only=True,cant_stroke=170.,**kwargs):
        '''
        Loads the data in the same way as the usual FileData class with a couple steps changed to allow for the
        injection of synthetic signals.
        '''

        # set aside the lightweight argument for later and remove it from the arguments passed to load_data()
        lightweight = kwargs.pop('lightweight',False)
        p0_bead = kwargs.pop('p0_bead',None)
        signal_model = kwargs.pop('signal_model',None)
        mass_bead = kwargs.pop('mass_bead',0)
        self.load_data(p0_bead=p0_bead,signal_model=None,mass_bead=mass_bead,lightweight=False,**kwargs)

        # add synthetic cantilever motion to the noise data so we get a non-trivial signal model
        if noise_only:
            cant_y = (cant_stroke/2.)*np.sin(2.*np.pi*3.*np.linspace(0,10,self.nsamp))
            self.cant_pos_calibrated = self.cant_pos_calibrated + np.array((np.zeros_like(cant_y),cant_y,np.zeros_like(cant_y)))

        # make the synthetic signal using the modified cantilever data and the given alpha and lambda
        self.make_synthetic_signal(signal_model,p0_bead,mass_bead=mass_bead,alpha=alpha,lamb=lamb)
        self.calibrate_bead_response(sensor='QPD',no_tf=True)
        self.calibrate_bead_response(sensor='PSPD',no_tf=True)
        self.xypd_force_calibrated[2,:] = np.copy(self.qpd_force_calibrated[2,:])
        self.get_ffts_and_noise(noise_bins=10)
        self.make_templates(signal_model,p0_bead,mass_bead=mass_bead)

        # for use with AggregateData, don't carry around all the raw data
        if lightweight:
            self.drop_raw_data()


    def make_synthetic_signal(self,signal_model,p0_bead,mass_bead=0,cant_vec=None,num_harms=10,alpha=1e7,lamb=10.):
        '''
        Get a signal template, scale it by the alpha provided, and add it to the raw time series.
        '''

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
        params,_ = signal_model.get_params_and_inds()

        positions = np.ones((len(bead_y),3))*1e-6
        positions[:,0] *= bead_x
        positions[:,1] *= bead_y
        positions[:,2] *= bead_z

        # for each value of the model parameter, append the array of force for all
        # cantilever positions
        forces = []
        ind = np.argmin(np.abs(params - lamb))

        forces = signal_model.get_force_at_pos(positions,[ind])*mass_fac*alpha

        # detrend the force data before windowing
        forces = forces - np.mean(forces,axis=0,keepdims=True)

        # apply the same window to the force data as the quad data
        forces = forces*self.window[None,:,None]

        # add the synthetic force time series as a class attribute
        self.quad_raw_data = self.qpd_force_calibrated + forces[0,...].swapaxes(0,1)
        self.xypd_raw_data = self.xypd_force_calibrated + forces[0,...].swapaxes(0,1)


class SynthAggregate(AggregateData):
    '''
    This child class of AggregateData inherits most of the methods but override the file loading
    to allow for synthetic signals to be added as the files are loaded.
    '''

    def __init__(self,*args,**kwargs):
        '''
        Usual __init__ function but also defines the alpha and lambda used to create the synthetic signal.
        '''
        keys = ['data_dirs','file_prefixes','descrips']
        for i,arg in enumerate(args):
            kwargs[keys[i]] = arg
        data_dirs = kwargs.pop('data_dirs',[])
        file_prefixes = kwargs.pop('file_prefixes',[])
        descrips = kwargs.pop('descrips',[])
        alpha = kwargs.pop('alpha',1e7)
        lamb = kwargs.pop('lamb',10.)
        noise_only = kwargs.pop('noise_only',True)
        super().__init__(data_dirs=data_dirs,file_prefixes=file_prefixes,descrips=descrips,**kwargs)
        self.alpha = alpha
        self.lamb = lamb
        self.noise_only = noise_only


    def process_file(self,file_path,qpd_diag_mat=None,signal_model=None,ml_model=None,p0_bead=None,\
                     mass_bead=0,harms=[],max_freq=500.,downsample=True,wiener=[False,True,False,False,False],\
                     no_tf=False,lightweight=True):
        '''
        Process data for an individual file and return the SynthFile object.
        '''
        this_file = SynthFile(file_path)
        try:
            this_file.load_and_inject(qpd_diag_mat=qpd_diag_mat,signal_model=signal_model,ml_model=ml_model,\
                                      p0_bead=p0_bead,mass_bead=mass_bead,harms=harms,downsample=downsample,wiener=wiener,\
                                      max_freq=max_freq,no_tf=no_tf,lightweight=lightweight,alpha=self.alpha,lamb=self.lamb,\
                                      noise_only=self.noise_only)
        except Exception as e:
            this_file.is_bad = True
            this_file.error_log = repr(e)
        return this_file