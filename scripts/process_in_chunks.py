import sys
import os
from data_processing import AggregateData
import argparse

import time

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
parser.add_argument('prefix',type=str)
parser.add_argument('-num_cores',type=int,default=20)
parser.add_argument('-no_templates',action='store_true',default=False)
parser.add_argument('-descrip',type=str,default='')
parser.add_argument('-num_to_load',type=int,default=1000000)
parser.add_argument('-chunk_size',type=int,default=100)
args = parser.parse_args()
path = args.path
prefix = args.prefix
num_cores = args.num_cores
load_templates = ~args.no_templates
descrip = args.descrip
num_to_load = args.num_to_load
chunk_size = args.chunk_size
if descrip=='':
    descrip = ''.join(path.split('/data/new_trap/')[-1].split('/'))

# define functions
def get_loaded_indices(aggdat):
    loaded_files = aggdat.file_list
    failed_files = aggdat.bad_files
    loaded_indices = [int(loaded_file.split('_')[-1][:-3]) for loaded_file in loaded_files]
    failed_indices = [int(failed_file.split('_')[-1][:-3]) for failed_file in failed_files]
    indices = loaded_indices + failed_indices
    # pick the next file to start from
    if len(indices)==0:
        first_index = 0
    else:
        first_index = max(indices) + 1
    return indices, first_index

# determine how many files are to be loaded
all_files = os.listdir(path)
num_raw_files = len([path+'/'+f for f in all_files if (os.path.isfile(path+'/'+f) and \
                     (prefix in f and f.endswith('.h5')))])
print('Found {} files in '.format(num_raw_files)+path)

# object to hold the existing aggdat.h5 data
aggdat = AggregateData()

# first check if there is an existing aggdat.h5 file and load the data if it exists
agg_path = '/data/new_trap_processed/aggdat/'+'/'.join(path.split('/')[3:-1]) + '/aggdat.h5'
if os.path.exists(agg_path):
    aggdat.load_from_hdf5(agg_path)

# determine which raw files have been processed
indices,first_index = get_loaded_indices(aggdat)

# loop through and load data until all raw files have been processed
while(len(indices)<min(num_raw_files,num_to_load)):
    print('Loading starting from file '+str(first_index))

    # data not yet loaded
    aggdat_temp = AggregateData([path],[prefix],[descrip],num_to_load=chunk_size,first_index=first_index)
    aggdat_temp.load_yukawa_model(lambda_range=[1e-6,1e-4],num_lambdas=25)
    aggdat_temp.load_file_data(num_cores=num_cores,load_templates=load_templates,harms=[3,5,6,8,9,12,14])
    
    # newly loaded + previously loaded data
    aggdat_new = AggregateData()
    if len(aggdat.file_list)>0:
        aggdat_new.merge_objects([aggdat,aggdat_temp])
    else:
        aggdat_new = aggdat_temp

    # determine which files have been loaded again 
    indices,first_index = get_loaded_indices(aggdat_new)

    # set variables for the next iteration
    aggdat = aggdat_new
    del aggdat_temp

# redo the binning and save the merged object
aggdat.bin_by_aux_data()
aggdat.save_to_hdf5()