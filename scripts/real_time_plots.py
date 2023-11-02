import os
import sys
import subprocess

# directory where the data is currently being written
current_dir = os.getenv('CURRENT_DATA_DIR')
current_prefix = 'shaking_'
if current_dir is None:
    print('No current data folder specified. Exiting.')
    sys.exit()
if current_dir[-1]!='/':
    current_dir = ''.join([current_dir,'/'])

# path where the AggregateData object will be saved by default
agg_path = '/data/new_trap_processed/aggdat/'+'/'.join(current_dir.split('/')[3:-1]) + '/aggdat.h5'

# process the data up to the current point and make the standard figures
subprocess.run(['python','process_in_chunks.py',current_dir,current_prefix])
subprocess.run(['python','make_figures.py',agg_path])