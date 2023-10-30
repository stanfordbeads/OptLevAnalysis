import sys
sys.path.insert(0,'../lib/')
from data_processing import AggregateData
import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
parser.add_argument('prefix',type=str)
parser.add_argument('-num_cores',type=int,default=39)
parser.add_argument('-no_templates',action='store_true',default=False)
parser.add_argument('-descrip',type=str,default='')
parser.add_argument('-num_to_load',type=int,default=1000000)
parser.add_argument('-output_file',type=str,default='')
args = parser.parse_args()
path = args.path
prefix = args.prefix
num_cores = args.num_cores
load_templates = ~args.no_templates
output_file = args.output_file
descrip = args.descrip
num_to_load = args.num_to_load
if output_file == '':
    output_file = ''.join(path.split('/')[-5:])+'.h5'
if descrip=='':
    descrip = output_file[:-3]

# create and save AggregateData object
aggdat = AggregateData([path],[prefix],[descrip],num_to_load=num_to_load)
aggdat.load_yukawa_model(lambda_range=[1e-6,1e-4],num_lambdas=25)
aggdat.load_file_data(num_cores=num_cores,load_templates=load_templates,harms=[3,5,6,8,9,12,14])
aggdat.bin_by_aux_data()
aggdat.save_to_hdf5(output_file)