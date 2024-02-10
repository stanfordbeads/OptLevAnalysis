import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
parser.add_argument('scale_factors',nargs=3,type=float)
args = parser.parse_args()
path = args.path
scale_factors = args.scale_factors

with h5py.File(path,'a') as tf_file:
    tf_file.attrs['scaleFactors_QPD_diag'] = scale_factors

print('\nWrote new scale factors to file {}'.format(path))