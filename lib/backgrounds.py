import csv
import numpy as np
from data_processing import AggregateData


class BackgroundData:
    '''
    This class can be used to study the relationship between the observed backgrounds and
    numerous experimental parameters. To initialize an object of this class, pass the path
    to a csv file to which the corresponding sheet of the backgrounds spreadsheet has been
    exported. Ensure that the backgrounds spreadsheet has been filled out using the usual
    template.

    You can then load the data into a separate AggregateData object and populate the
    corresponding BackgroundData object with the relevant parameters using the set_params()
    method. In this way, the BackgroundData object can be used to index rows within the
    dictionary of the AggregateData object to select individual files.
    '''

    def __init__(self,path,segments):
        '''
        Initialize the background object with all of the parameters that will be studied
        in relation to the background prominence.
        '''
        self.spreadsheet_path = path
        headers,lines = self.load_bkg_spreadsheet(self.spreadsheet_path)
        self.background_dict = self.make_bkg_dict(headers,lines)
        self.num_lines = len(self.background_dict['Directory name'])
        self.inds = self.get_directory_inds(segments)
        self.num_dirs = len(self.inds)
        self.make_descrips()


    def load_bkg_spreadsheet(self,path):
        '''
        Load the backgrounds spreadsheet and extract the relevant info for each dataset
        needed to make an AggregateData object.
        '''
        lines = []
        with open(path,'r') as f:
            reader = csv.reader(f,delimiter=',')
            segment = 0
            # extract headers and lines while keeping track of individual measurement segments
            for line in reader:
                if all([l=='' for l in line]):
                    segment += 1
                    continue
                elif line[0]=='':
                    line.append('Segment')
                    headers = line
                else:
                    line.append(str(segment))
                    lines.append(line)
        headers[0] = lines[0][0]
        lines = lines[1:]
        return headers,lines


    def make_bkg_dict(self,headers,lines):
        '''
        Construct a dictionary from the headers and content of the backgrounds spreadsheet.
        '''
        dict = {}
        for i,header in enumerate(headers):
            dict[header] = np.array([line[i] for line in lines])
        return dict


    def make_descrips(self):
        '''
        Concatenate the spreadsheet parameters together to produce a string that fully defines
        the experimental parameters for each directory so the information can be retrieved from
        within the AggregateData object if necessary.
        '''
        descrips = []
        for i in range(self.num_dirs):
            descrip = self.background_dict['Aperture'][i] + '_' + \
                      self.background_dict['Shield sep [um]'][i] + '_' + \
                      self.background_dict['Attractor sep [nm]'][i] + '_' + \
                      self.background_dict['Attractor stroke [um]'][i] + '_' + \
                      self.background_dict['Attractor bias [mV]'][i] + '_' + \
                      self.background_dict['Shield bias [mV]'][i] + '_' + \
                      self.background_dict['y-offset [um]'][i] + '_' + \
                      self.background_dict['z-offset [um]'][i] + '_' + \
                      self.background_dict['Spin freq [kHz]'][i] + '_' + \
                      self.background_dict['Spin plane'][i]
            descrips.append(descrip)
        self.descrips = np.array(descrips)


    def get_directory_inds(self,segments):
        '''
        Get the indices of which directories should be loaded from those in the spreadsheet, specified
        by the spreadsheet segments to include (given as a list).
        '''
        segments = [str(s) for s in segments]
        inds = [s in segments for s in self.background_dict['Segment']]
        inds = np.arange(self.num_lines)[inds]
        return inds


    def load_aggregate_data(self,num_to_load=100,**kwargs):
        '''
        Load the background data in an AggregateData object.
        '''
        bkgdat = AggregateData(self.background_dict['Directory name'][self.inds],file_prefixes='',\
                               descrips=self.descrips,num_to_load=num_to_load)
        bkgdat.load_file_data(**kwargs)
        return bkgdat


    def set_params(self,bkgdat):
        '''
        Return an array of indices for the corresponding subset of the aggregated data
        based on the row of the backgrounds spreadsheet to be collected.
        '''
        self.aperture = []
        self.attractor_stroke = []
        self.shield_sep = []
        self.attractor_sep = []
        self.attractor_bias = []
        self.shield_bias = []
        self.y_offset = []
        self.z_offset = []
        self.spin_freq = []
        self.spin_plane = []
        for i in range(len(bkgdat.num_files)):
            self.aperture += [self.background_dict['Aperture'][self.inds][i]]*bkgdat.num_files[i]
            self.attractor_stroke += [float(self.background_dict['Attractor stroke [um]'][self.inds][i])]*bkgdat.num_files[i]
            self.shield_sep += [float(self.background_dict['Shield sep [um]'][self.inds][i])]*bkgdat.num_files[i]
            self.attractor_sep += [float(self.background_dict['Attractor sep [nm]'][self.inds][i])]*bkgdat.num_files[i]
            self.attractor_bias += [float(self.background_dict['Attractor bias [mV]'][self.inds][i])]*bkgdat.num_files[i]
            self.shield_bias += [float(self.background_dict['Shield bias [mV]'][self.inds][i])]*bkgdat.num_files[i]
            self.y_offset += [float(self.background_dict['y-offset [um]'][self.inds][i])]*bkgdat.num_files[i]
            self.z_offset += [float(self.background_dict['z-offset [um]'][self.inds][i])]*bkgdat.num_files[i]
            self.spin_freq += [float(self.background_dict['Spin freq [kHz]'][self.inds][i])]*bkgdat.num_files[i]
            self.spin_plane += [self.background_dict['Spin plane'][self.inds][i]]*bkgdat.num_files[i]
        self.aperture = np.array(self.aperture)
        self.attractor_stroke = np.array(self.attractor_stroke)
        self.shield_sep = np.array(self.shield_sep)
        self.attractor_sep = np.array(self.attractor_sep)
        self.attractor_bias = np.array(self.attractor_bias)
        self.shield_bias = np.array(self.shield_bias)
        self.y_offset = np.array(self.y_offset)
        self.z_offset = np.array(self.z_offset)
        self.spin_freq = np.array(self.spin_freq)
        self.spin_plane = np.array(self.spin_plane)


# ----------------------------------------------------------
# USEFUL FUNCTIONS FOR ANALYSIS OF BACKGROUNDS GO BELOW HERE
# ----------------------------------------------------------


def prominence(ffts,sidebands,n_datasets=351,n_samp=100,bootstrap=False,combine=False):
    '''
    Compute the prominence at all axes and harmonics for a set of measurements.

    :ffts:         the array of ffts from an AggregateData agg_dict entry
    :sidebands:    the array of sidebands from an AggregateData agg_dict entry
    :n_datasets:   the number of datasets to sample when bootstrapping the uncertainty
    :n_samp:       the number of bootstrap samples to use when bootstrapping the uncertainty
    :bootstrap:    whether or not to boostrap the uncertainty. If false, propagates the variances
    :combine:      whether or not to compute prominences over all harmonics. If false, keeps them separate
    '''
    num_sb = int(sidebands.shape[2]/ffts.shape[2])
    if not bootstrap:
        n_samp = ffts.shape[0]
    if combine:
        ffts = ffts.reshape(ffts.shape[0],-1)
        sidebands = sidebands.reshape(ffts.shape[0],-1,num_sb)
        proms = np.zeros_like(np.real(ffts[0,...]))    
        errs = np.zeros_like(proms)
        m1s = np.zeros_like(proms)
        p1s = np.zeros_like(proms)
        med = np.zeros_like(proms)
        dists = np.zeros(np.array([n_samp]+list(proms.shape)))
        for i in range(ffts.shape[1]):
            proms[i],errs[i] = prom_func(ffts[:,i],sidebands[:,i,:])
            inds = range(ffts.shape[0])
            if bootstrap:
                inds = np.random.randint(0,ffts.shape[0],(n_samp,n_datasets))
            vals = np.array([prom_func(np.array([ffts[k,i]]),np.array([sidebands[k,i]]))
                            for k in inds])
            m1s[i] = np.quantile(vals[:,0],0.16)
            p1s[i] = np.quantile(vals[:,0],0.84)
            med[i] = np.quantile(vals[:,0],0.50)
            dists[:,i] = vals[:,0]
    else:
        proms = np.zeros_like(np.real(ffts[0,...]))    
        errs = np.zeros_like(proms)
        m1s = np.zeros_like(proms)
        p1s = np.zeros_like(proms)
        med = np.zeros_like(proms)
        if not bootstrap:
            n_samp = ffts.shape[0]
        dists = np.zeros(np.array([n_samp]+list(proms.shape)))
        for i in range(ffts.shape[1]):
            for j in range(ffts.shape[2]):
                proms[i,j],errs[i,j] = prom_func(ffts[:,i,j],sidebands[:,i,j*num_sb:(j+1)*num_sb])
                inds = range(ffts.shape[0])
                if bootstrap:
                    inds = np.random.randint(0,ffts.shape[0],(n_samp,n_datasets))
                vals = np.array([prom_func(np.array([ffts[k,i,j]]),np.array([sidebands[k,i,j*num_sb:(j+1)*num_sb]]))
                                for k in inds])
                m1s[i,j] = np.quantile(vals[:,0],0.16)
                p1s[i,j] = np.quantile(vals[:,0],0.84)
                med[i,j] = np.quantile(vals[:,0],0.50)
                dists[:,i,j] = vals[:,0]
    return proms,errs,dists,m1s,p1s,med


def prom_func(ffts,sidebands):
    '''
    Function called by prominence() to get the prominence with error for a
    single harmonic and axis.
    '''
    power = np.abs(ffts)**2
    num = np.mean(power)
    std_num = np.std(power,ddof=1)
    power_sb = np.mean(np.abs(sidebands)**2,axis=1)
    denom = np.mean(power_sb)
    std_denom = np.std(power_sb,ddof=1)
    prom = np.sqrt(num/denom)
    std_prom = (prom/2.)*np.sqrt((std_num/num)**2 + (std_denom/denom)**2)
    return prom,std_prom

        



        