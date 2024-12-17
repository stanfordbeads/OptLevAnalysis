import sys
from optlevanalysis.data_processing import AggregateData
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
parser.add_argument('-attractor',type=str,default='pt_black')
parser.add_argument('-time_domain',action='store_true',default=False)
parser.add_argument('-diagonalize_qpd',action='store_true',default=False)
args = parser.parse_args()
path = args.path
prefix = args.prefix
num_cores = args.num_cores
load_templates = ~args.no_templates
output_file = args.output_file
descrip = args.descrip
num_to_load = args.num_to_load
attractor = args.attractor
time_domain = args.time_domain
diagonalize_qpd = args.diagonalize_qpd
if descrip=='':
    descrip = ''.join(path.split('/data/new_trap/')[-1].split('/'))

# create and save AggregateData object
aggdat = AggregateData([path],[prefix],[descrip],num_to_load=num_to_load)
aggdat.load_yukawa_model(lambda_range=[1e-6,1e-4],num_lambdas=25,attractor=attractor)
aggdat.load_file_data(num_cores=num_cores,load_templates=True,harms=[2,3,4,5,6,7,8,9,10,11,12,13,14],wiener=[True]*5,\
                      time_domain=time_domain,diagonalize_qpd=diagonalize_qpd)
aggdat.bin_by_aux_data()
aggdat.save_to_hdf5(output_file)