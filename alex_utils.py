# Don't bother reading this!  Just utility functions.
import shutil, os, pathlib, pickle, sys, math, importlib, json.tool, argparse, requests, atexit, builtins
import pandas as pd
import numpy as np
from glob import glob
from os.path import join, exists, isdir
from tqdm import tqdm
from itertools import product
from datetime import datetime
from sklearn.utils import class_weight
from os.path import join

from itertools import product
import collections
import functools

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   src: https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


def get_sample_weight(labels,class_weights=None):
    labels = labels.astype('int32')
    if class_weights is None:
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels).astype('int32'), y=labels)
    sample_weight = lvmap(lambda elt: class_weights[elt], labels)
    return sample_weight

def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features

def csd_to_pk(ds, key, path=None):
    new_text = {}
    for k in ds[key].keys():
        new_text[k] = {
            'features': ar(ds[key][k]['features']),
            'intervals': ar(ds[key][k]['intervals']),
        }
    if path is not None:
        save_pk(path, new_text)
    return new_text

def lvmap(f, arr, axis=None):
    if axis is None:
        f = np.vectorize(f)
        return f(arr)
    else:
        return np.apply_along_axis(f,axis=axis,arr=arr)
    
def init_except_hook(gpu_id=None, filename=None, test=False):
    def my_except_hook(exctype, value, traceback):
        print('\n\n########### ERROR ###########')
        print('Emailing you that an error has occurred...')
        # update_gpu_log(gpu_id, 'open')
        sys.__excepthook__(exctype, value, traceback)
        send_email()
            # t.send(f'ERROR: {os.path.basename(__file__) if filename is None else filename} failed')
    sys.excepthook = my_except_hook

def init_exit_hook(gpu_id=None, test=False):
    def my_exit_hook():
        # update_gpu_log(gpu_id, 'open')
        if not test:
            send_email()
            # t.send('Finished!')
    atexit.register(my_exit_hook)

def send_email(subject='Hi there', text='Hello!', secrets_path='/z/abwilf/dw/mailgun_secrets.json'):
    secrets = load_json(secrets_path)
    return requests.post(
        secrets['url'],
        auth=("api", secrets['api_key']),
        data={"from": secrets['from_addr'],
            "to": secrets['to_addr'],
            "subject": subject,
            "text": text})

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

def update_gpu_log(gpu_id, status):
    if gpu_id is None:
        return
    gpu_log = load_json(gpu_log_path)
    gpu_log[str(gpu_id)] = status
    save_json(gpu_log_path, gpu_log)

def init_except_hook(gpu_id=None, filename=None, test=False):
    def my_except_hook(exctype, value, traceback):
        print('\n\n########### ERROR ###########')
        print('Emailing you that an error has occurred...')
        update_gpu_log(gpu_id, 'open')
        sys.__excepthook__(exctype, value, traceback)
        if not test:
            send_email()
            # t.send(f'ERROR: {os.path.basename(__file__) if filename is None else filename} failed')
    sys.excepthook = my_except_hook

def init_exit_hook(gpu_id=None, test=False):
    def my_exit_hook():
        update_gpu_log(gpu_id, 'open')
        if not test:
            send_email()
            # t.send('Finished!')
    atexit.register(my_exit_hook)

def send_email(subject='Hi there', text='Hello!', secrets_path='./mailgun_secrets.json'):
    secrets = load_json(secrets_path)
    return requests.post(
        secrets['url'],
        auth=("api", secrets['api_key']),
        data={"from": secrets['from_addr'],
            "to": secrets['to_addr'],
            "subject": subject,
            "text": text})

def obj_to_grid(a):
    '''get all objects corresponding to hyperparamter grid search
    a = {'b': [1,2], 'c': [3,4], 'd': 5}
    ->
    [{'b': 1, 'c': 3, 'd': 5}, {'b': 1, 'c': 4, 'd': 5}, {'b': 2, 'c': 3, 'd': 5}, {'b': 2, 'c': 4, 'd': 5}]
    '''

    for k,v in list(a.items()):
        if type(v) != list:
            a[k] = [v]

    to_ret = []
    for values in list(product(*list(a.values()))):
        to_ret.append({k:v for k,v in zip(a.keys(), values)})
    return to_ret

def ar(a):
    return np.array(a)

def rmtree(dir_path):
    print(f'Removing {dir_path}')
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    # else:
        # print(f'{dir_path} is not a directory, so cannot remove')

def remove_inf(x):
    x[x==-np.inf] = 0
    x[x==np.inf] = 0
    return np.array(x, dtype='float32')

def npr(x, decimals=4):
    '''Round'''
    return np.round(x, decimals=decimals)

def nprs(x, decimals=2, scale=100):
    '''Round & scale'''
    return np.round(x*scale, decimals=decimals)

def int_to_str(*keys):
    return [list(map(lambda elt: str(elt), key)) for key in keys]
    
def rm_mkdirp(dir_path, overwrite, quiet=False):
    if os.path.isdir(dir_path):
        if overwrite:
            if not quiet:
                print('Removing ' + dir_path)
            shutil.rmtree(dir_path, ignore_errors=True)

        else:
            print('Directory ' + dir_path + ' exists and overwrite flag not set to true.  Exiting.')
            exit(1)
    if not quiet:
        print('Creating ' + dir_path)
    pathlib.Path(dir_path).mkdir(parents=True)

def lists_to_2d_arr(list_in, max_len=None):
    '''2d list in, but where sub lists may have differing lengths, one big padded 2d arr out'''
    max_len = max([len(elt) for elt in list_in]) if max_len is None else max_len
    new_arr = np.zeros((len(list_in), max_len))
    for i,elt in enumerate(list_in):
        if len(elt) < max_len:
            new_arr[i,:len(elt)] = elt
        else:
            new_arr[i,:] = elt[:max_len]
    return new_arr


def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def rmfile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    
def rglob(dir_path, pattern):
    return list(map(lambda elt: str(elt), pathlib.Path(dir_path).rglob(pattern)))

def move_matching_files(dir_path, pattern, new_dir, overwrite):
    rm_mkdirp(new_dir, True, overwrite)
    for elt in rglob(dir_path, pattern):
        shutil.move(elt, new_dir)
    
def df_sample():
    return pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], columns=['one', 'two', 'three'])

def subset(a, b):
    return np.min([elt in b for elt in a]) > 0

def subsets_eq(a,b):
    return subset(a,b) and subset(b,a)

def dict_at(d):
    k = lkeys(d)[0]
    return k, d[k]

def list_gpus():
    return tf.config.experimental.list_physical_devices('GPU')

def save_pk(file_stub, pk, protocol=4):
    filename = file_stub if '.pk' in file_stub else f'{file_stub}.pk'
    rmfile(filename)
    with open(filename, 'wb') as f:
        pickle.dump(pk, f, protocol=protocol)
    
def load_pk(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        return load_pk_old(filename)

def load_pk_old(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

# def ds(d, *keys):
#     '''Destructure dict. e.g.
#     a = {'hey': 1, 'you': 2}
#     hey, you = ds(a, 'hey', 'you')
#     '''
#     return [ d[k] if k in d else None for k in keys ]

def get_dir(path, silent=True):
    if '.' not in path and not silent:
        print(f'NOTE: {path} is not a file, creating dir with just {path}')
    else:
        path = '/'.join(path.split('/')[:-1])
    return path

def get_ints(*keys):
    return [int(key) for key in keys]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(file_stub, obj):
    filename = file_stub
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)

def load_json(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    with open(filename) as json_file:
        return json.load(json_file)

def lfilter(fn, iterable):
    return list(filter(fn, iterable))

def lkeys(obj):
    return list(obj.keys())

def lvals(obj):
    return list(obj.values())

def lmap(fn, iterable):
    return list(map(fn, iterable))

def arlmap(fn, iterable):
    return ar(list(map(fn, iterable)))

def arlist(x):
    return ar(list(x))
    
def llmap(fn, iterable):
    return list(map(lambda elt: fn(elt), iterable))
    
def sort_dict(d, reverse=False):
    return {k: v for k,v in sorted(d.items(), key=lambda elt: elt[1], reverse=reverse)}

def csv_path(sym):
    return join('csvs', f'{sym}.csv')

def is_unique(a):
    return len(np.unique(a)) == len(a)

def lists_equal(a,b):
    return np.all([elt in b for elt in a]) and np.all([elt in a for elt in b])
    
def split_arr(cond, arr):
    return lfilter(cond, arr), lfilter(lambda elt: not cond(elt), arr)

def lzip(*keys):
    return list(zip(*keys))

def dilation_pad(max_len, max_dilation_rate):
    to_ret = math.ceil(max_len/max_dilation_rate)*max_dilation_rate
    assert (to_ret % max_dilation_rate) == 0
    return to_ret

def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)

def onehot_initialization(a):
    '''a is 2d with values ranging from 0 to num_labels-1.  this turns a into a 3d matrix with same first two dimensions, one hot encoded'''
    ncols = a.max()+1
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out

def zero_pad_to_length(data, length):
    padAm = length - data.shape[0]
    if padAm == 0:
        return data
    else:
        return np.pad(data, ((0,padAm), (0,0)), 'constant')

def get_batch(arr, batch_idx, batch_size):
    return arr[batch_idx * batch_size:(batch_idx + 1) * batch_size]

def sample_batch(arrs, batch_size):
    start = np.random.randint(arrs[0].shape[0]-batch_size)
    return [arr[start:(start+batch_size)] for arr in arrs]

def shuffle_data(*arrs):
    rnd_state = np.random.get_state()
    for arr in arrs:
        np.random.shuffle(arr)
        np.random.set_state(rnd_state)

def get_class_weights(arr):
    '''pass in dummies'''
    class_weights = np.nansum(arr, axis=0)
    return np.sum(class_weights) / (class_weights*len(class_weights))

def get_class_weights_ds(arr):
    '''do not pass in dummies'''
    arr = np.stack(np.unique(np.array(arr), return_counts=True), axis=1)
    return (np.sum(arr[:,1]) - arr[:,1]) / arr[:,1]

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

'''
mapping from original preprocessed mosei data to form used in rest of .pickle files
def map_13_to_7(mode):
    a = load_pk(f'/z/abwilf/hffn/multimodal-sentiment-analysis/dataset/mosei/{mode}_3way.pickle')
    new_obj = [None]*7
    new_obj[0] = np.concatenate([a[0], a[2]])
    new_obj[1] = np.concatenate([a[1], a[3]])
    new_obj[2] = a[4]
    new_obj[3] = a[5]
    new_obj[4] = a[6]
    new_obj[5] = np.concatenate([a[7], a[8]])
    new_obj[6] = a[9]
    save_pk(f'/z/abwilf/hffn/reshaped_theirs/{mode}.pk', new_obj)
[map_13_to_7(mode) for mode in ['audio', 'text', 'video']]
'''

def rreplace(s, old, new, occurrence):
    '''replace the last occurrence # of of old characters with new characters in a str'''
    li = s.rsplit(old, occurrence)
    return new.join(li)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LD(dict):
    '''Dict mod that allows multi indexing'''
    def __init__(self, dict_in):
        self.__dict__ = dict_in
    def lcg(self,key,val=[]):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            self.__dict__[key] = val
            return self.__dict__[key]
    def __getitem__(self, key):
        if type(key) == list:
            return [self.__dict__[k] for k in key]
        else:
            return self.__dict__[key]
    def __setitem__(self, key, item):
        self.__dict__[key] = item
    def __repr__(self):
        return repr(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __delitem__(self, key):
        del self.__dict__[key]
    def clear(self):
        return self.__dict__.clear()
    def copy(self):
        return self.__dict__.copy()
    def has_key(self, k):
        return k in self.__dict__
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def pop(self, *args):
        return self.__dict__.pop(*args)
    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)
    def __contains__(self, item):
        return item in self.__dict__
    def __iter__(self):
        return iter(self.__dict__)
    def __unicode__(self):
        return unicode(repr(self.__dict__))

def z_norm(arr):
    mean, std = arr.mean(), arr.std()
    return (arr-arr.mean()) / (arr.std() + 1e-5), mean, std

def un_z_norm(arr, mean, std):
    return arr*std + mean