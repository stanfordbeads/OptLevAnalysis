import os
import glob
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
plt.style.use('../src/optlevanalysis/optlevstyle.mplstyle')

import h5py
import subprocess
import time

### Path to save the best current experimental constraints as a single array
save = False
savepath = '/home/clarkeh/gravity_sens_plot/prev_meas/master_all_2025.txt'

### Some arrays for sorting out various mesurements
min_lambdas = []
max_lambdas = []

files = glob.glob('/home/clarkeh/gravity_sens_plot/prev_meas/*.txt')

rawfiles = []
rawdata = []
for file in files:
    ### Ignore previously compiled limits
    if 'master' in file:
        continue
    if 'tot' in file:
        continue

    ### Load the raw data for reach limit and append it to 
    ### an array. Do the same for the file name for labeling
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    rawdata.append(data)
    rawfiles.append(file)

### Determine the extent in lambda of the published constraint 
for data in rawdata:
    min_lambdas.append(np.min(data[:,0]))
    max_lambdas.append(np.max(data[:,0]))

### Sort the data based on the min value of lambda and build
### a sorted array of the rawdata
sorter = np.argsort(min_lambdas)
alldata = []
allfiles = []
for ind in sorter:
    alldata.append(rawdata[ind])
    allfiles.append(rawfiles[ind])

### Construct a master array of logrithmically-spaced lambdas from 
### the largest and smallest values of lambdas over all of the data
master_lambda = np.logspace(np.log10(np.min(min_lambdas)), \
                            np.log10(np.max(max_lambdas)), 100000)
master_alpha = []

### Each constraint will be resampled at values contained within 
### the master_lambda array, and then the tightest constraint will be 
### selected. To do this algorithmically, we build a matrix of the 
### overlap between each measurement and the master_lambda array
overlap_matrix = np.zeros((len(master_lambda), len(allfiles)))

resamp_data = []
for i, data in enumerate(alldata):
    ### Sort each constraint so that lambda is monotonically increasing
    sorter = np.argsort(data[:,0])
    clambda = data[sorter,0]
    calpha = data[sorter,1]

    ### Determine the overlap and add it to the matrix
    overlap = (master_lambda >= clambda[0]) * (master_lambda <= clambda[-1])
    overlap_matrix[:,i] = overlap

    ### Get the values of master_lamda covered by the overlap. The
    ### constraint will be resampled over these values
    rlambda = master_lambda[overlap]

    ### Build a linear interpolating function in loglog space and 
    ### determine the resampled values of alpha for the constraint
    func = interpolate.interp1d(np.log(clambda), np.log(calpha))
    ralpha = np.exp(func(np.log(rlambda)))

    ### append the resampled data
    resamp_data.append(np.array([rlambda, ralpha]))


### Loop over all values of master_lambda an select the best constraint for
### the master array. 
for i, lamb_val in enumerate(master_lambda):
    ### For each lambda, find all measurements with a valid alpha
    valid = overlap_matrix[i]

    ### Loop over all the datasets and append valid values
    vals = []
    for j, data in enumerate(resamp_data):
        ### Check if valid
        if not valid[j]:
            continue

        ### Find the in index of the resamp_data corresponding to the 
        ### current value of master_lambda
        ind = np.argmin(np.abs(data[0] - lamb_val))

        ### Append the valid value
        vals.append(data[1,ind])

    ### Append the best constraint to the master_alpha array. For some reason
    ### the last value of master_lambda sometimes has no valid measurement, so
    ### there's a little try/except block here to try to deal with that
    try:
        master_alpha.append(np.min(vals))
    except:
        master_alpha.append(master_alpha[-1])



### Plot all the constraints individually and the extracted best curve
colors = [plt.get_cmap('plasma',len(alldata))(i) for i in range(len(alldata))]

# save the data so it can be opened later and plotted for my thesis
with h5py.File('/home/clarkeh/thesis_plot_data/all_alpha_lambda.h5', 'w') as d:
    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    git_origin = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('ascii').strip()
    
    d.attrs['git_revision_short_hash'] = short_hash
    d.attrs['git_origin'] = git_origin
    d.attrs['creation_timestamp'] = time.time()
    d.attrs['creation_user'] = str(os.environ.get('USER'))

    for i, data in enumerate(alldata):
        label = allfiles[i].split('/')[-1].split('.')[0]
        plt.loglog(data[:,0], data[:,1], color=colors[i], label=label)
        
        d.create_dataset(label + '/' + 'lambdas', data=data[:,0])
        d.create_dataset(label + '/' + 'alphas', data[:,1])

    d.create_dataset('master/lambdas', data=master_lambda)
    d.create_dataset('master/alphas', data=master_alpha)

plt.legend(fontsize=8, ncol=2)
plt.loglog(master_lambda, master_alpha, lw=2, ls='--', color='r')
plt.savefig('/home/clarkeh/gravity_sens_plot/prev_meas/all_grav_limits.png')

if save:
    np.savetxt(savepath, np.array([master_lambda, master_alpha]).T, delimiter=',')
    print('Saved best limits to: {:s}'.format(os.path.abspath(savepath)))




