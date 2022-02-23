import io
import os
import json
import pickle
import distutils.dir_util
import numpy as np

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)

def write_pickle(fpath, obj):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)

def load_json(fname):
    with open(fname, encoding='utf-8') as f:
        json_obj = json.load(f)

    return json_obj

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath('./output/'+ parent) # making directory
    with io.open('./output/' + fname, 'w', encoding = 'utf-8') as f:
        json_str = json.dumps(data, ensure_ascii=False, default = _conv)
        f.write(json_str)