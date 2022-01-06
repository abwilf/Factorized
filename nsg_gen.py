import pandas as pd
import pathlib
from datetime import datetime
import itertools
import json
import hashlib
import os
import pathlib
from os.path import join
import shutil
import time
import json
import numpy as np
from glob import glob

HASH_LEN = 15
# HASH_LEN = -1
RESULTS_PATH = '/work/awilf/MTAG/results/'
OVERWRITE = 1
MAX_SBATCH_OPS = 8
SLEEP_SECS = 2
NUM_CHARS_SQUEUE = 3 # assume it only shows three characters if running over 100 tests

MEM_GB = 56

class Runtime():
    def __init__(self):
        self.start_time = datetime.now()
    def get(self):
        end_time = datetime.now()
        sec = (end_time - self.start_time).seconds
        days = int(sec/(3600*24))
        hrs = int(sec/3600)
        mins = int((sec % 3600)/60)
        
        days_str = f'{days} days, ' if days > 0 else ''
        hrs_str = f'{hrs} hrs, ' if hrs > 0 else ''
        # print(f'\nEnd time: {end_time}')
        print(f'Runtime: {days_str}{hrs_str}{mins} mins')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(file_stub, obj):
    filename = file_stub
    for k,v in obj.items():
        if isinstance(v, list):
            obj[k] = np.array(v)
            if obj[k].dtype==np.float32:
                obj[k] = obj[k].astype(np.float64)
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)
    
def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def rmtree(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

def load_json(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    with open(filename) as json_file:
        return json.load(json_file)

# hp = {
#     'a': [1,2],
#     'b': [2,3]
# }
from hp import hp

rt = Runtime()

hash = hashlib.sha1(json.dumps(hp, sort_keys=True).encode('utf-8')).hexdigest()[:HASH_LEN]

if 'subsets' in hp.keys():
    all_grids = []
    subsets = hp['subsets']
    del hp['subsets']

    for sub in subsets:
        sub_hp = {**hp, **sub}
        keys, vals = zip(*list(sub_hp.items()))
        grid = [{k:v for k,v in zip(keys,elt)} for elt in list(itertools.product(*vals))]
        all_grids.extend(grid)

    grid = all_grids
else:
    keys, vals = zip(*list(hp.items()))
    grid = [{k:v for k,v in zip(keys,elt)} for elt in list(itertools.product(*vals))]



print('Length of grid:', len(grid))
exit()

# create directory structure: within results/, looks like
'''
hash1
├── 1
│   ├── err
│   ├── out
│   └── results.json
├── 2
│   ├── err
│   ├── out
│   └── results.json
└── run_scripts
    ├── run1.sh
    └── run2.sh
'''

hash_path = join(RESULTS_PATH, hash)

if os.path.isdir(hash_path):
    if not OVERWRITE:
        print(f'Hash path {hash_path} exists and overwrite is not specified. Exiting now.')
        exit()
    else:
        print(f'Removing and rewriting hash path {hash_path}')
        rmtree(hash_path)

run_scripts_dir = join(RESULTS_PATH, hash, 'run_scripts')
mkdirp(run_scripts_dir)

exclude_list = 'compute-2-9,compute-0-19'
to_run = []
for i,comb in enumerate(grid):
    out_dir = join(RESULTS_PATH, hash, str(i))
    mkdirp(out_dir)

    to_add = ' '.join([f'--{k} {v}' for k,v in comb.items()]) + f' --out_dir {out_dir}'
    to_write = f'''\
#!/bin/bash
#
#SBATCH -p gpu_low
#SBATCH --gres=gpu:1  # Use GPU
#SBATCH --mem {MEM_GB}GB   # memory pool for all cores
#SBATCH -t 1-00:00    # time (D-HH:MM)
#SBATCH -o {join(out_dir,'%N-out.txt')}        # STDOUT. %j specifies JOB_ID.
#SBATCH -e {join(out_dir,'%N-err.txt')}        # STDERR. See the first link for more options.
#SBATCH --job-name {i}_{hash}        # STDERR. See the first link for more options.
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dummyblah123@gmail.com
#SBATCH --exclude={exclude_list}

# run this with: sbatch -p gpu_low ./blah.sh
# echo "hi"
# sleep 2
# source activate fairseq

cd /work/awilf/MTAG

ulimit -v unlimited
singularity exec --nv -B /work/awilf/awilf/.local/python3.7/site-packages:/home/awilf/.local/lib/python3.7/site-packages,/work/awilf/MTAG,/work/awilf/Standard-Grid,/work/awilf/CMU-MultimodalSDK,/work/awilf/Social-IQ blah4.sif python main.py {to_add}
# python hi.py {to_add}

'''

    # write to run_{i}.sh file
    run_script = join(run_scripts_dir, f'{i}.sh')
    to_run.append(run_script)
    file_path = run_script
    with open(file_path, 'w') as f:
        f.write(to_write)

print('\n###\n# Scripts generated...to run one as a trial: ')
print(f'srun -p gpu_low --exclude {exclude_list} --gres=gpu:1 --mem={MEM_GB}GB --pty bash')
print('\n'.join([elt for elt in to_write.split('\n') if len(elt)>0 and elt[0]!='#']))
print('###\n')

in_progress = []
finished = []
failed = []

tot_num = len(to_run)

program_complete = False

def get_ops(hash):
    return os.popen(f'squeue | grep awilf | grep {hash[:NUM_CHARS_SQUEUE]}').read().count('\n')

def submit_scripts(to_run):
    num_sbatch_ops = get_ops(hash)
    while num_sbatch_ops <= MAX_SBATCH_OPS and len(to_run) > 0:
        # print('num_sbatch_ops', num_sbatch_ops)
        run_script = to_run.pop()
        in_progress.append(run_script)
        os.popen(f'sbatch {run_script}').read()
        # print(f'sbatch {run_script}')
        num_sbatch_ops = get_ops(hash)
    
print(f'\n\nhash=\'{hash}\'\n')
print(f'\n## Status ## \nRunning {len(to_run)} scripts total (max {MAX_SBATCH_OPS} at a time) from {run_scripts_dir}\n')

def get_id(path):
    return path.split('/')[-1].replace('.sh', '')

def monitor(sleep_secs):
    global in_progress, program_complete
    
    submit_scripts(to_run)

    if len(to_run) == 0 and len(in_progress)==0:
        program_complete = True
        return

    sbatch_ops = os.popen(f'squeue | grep awilf | grep {hash[:NUM_CHARS_SQUEUE]}').read()
    finished.extend([elt for elt in in_progress if f"{get_id(elt)}_{hash[:NUM_CHARS_SQUEUE]}" not in sbatch_ops])
    in_progress = [elt for elt in in_progress if f"{get_id(elt)}_{hash[:NUM_CHARS_SQUEUE]}" in sbatch_ops]

    num_sbatch_ops = get_ops(hash)
    print(f'To run: {100*len(to_run) / tot_num:.1f}%\tIn progress: {100*len(in_progress)/tot_num:.1f}%\tFinished: {100*len(finished)/tot_num:.1f}% \tnum_sbatch_ops: {num_sbatch_ops}', end='\r')

    time.sleep(sleep_secs)

while not program_complete:
    monitor(SLEEP_SECS)

print('\n\nProgram Complete!')
print(f'\n\nhash=\'{hash}\'\n')

# Consolidate json files into a single csv
csv_path = join(hash_path, 'csv_results.csv')
print(f'Writing csv to \n{csv_path}')

ld = {} # list of dicts
for path in pathlib.Path(f'results/{hash}/').rglob('*.json'):
    id = int(str(path).split('/')[-2])
    hp_comb = grid[id]
    ld[id] = {**load_json(path), **{'_'+k:v for k,v in hp_comb.items()}}

df = pd.DataFrame(ld).transpose()
df.to_csv(csv_path)

# email me once completed
os.popen('sbatch --mail-type=END --mail-user=dummyblah123@gmail.com --wrap "python hi.py"')

rt.get()

hp_path = join(hash_path, 'hp.json')
save_json(hp_path, hp)

shutil.copyfile('main.py', join(hash_path, 'main.py'))

# compile error report
report = {
    'num_combs': len(grid),
    'num_successful': 0,
    'num_failed': 0,
    'errors': {}
}

for i in range(len(grid)):
    if 'success.txt' not in [elt.split('/')[-1] for elt in glob(join(hash_path, f'{i}', '*'))]:
        report['num_failed'] += 1
        report['errors'][i] = {
            'hp': grid[i],
            'err': open([elt for elt in glob(join(hash_path, f'{i}', '*')) if 'err.txt' in elt][0]).read()
        }
    else:
        report['num_successful'] += 1

save_json(join(hash_path, 'report.json'), report)
