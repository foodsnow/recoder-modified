from pathlib import Path
import os
import pickle, json
import numpy as np


cwd = Path(os.getcwd())
first_n = 1000

for pklfile in cwd.glob('*.pkl'):
    print(pklfile)
    fname = pklfile.name[:-4]
    pkldata = pickle.load(open(pklfile, 'rb'))
    with open(fname + '.json', 'w') as fout:
        if isinstance(pkldata, dict):
            sample = {k: pkldata[k] for k in list(pkldata.keys())[:first_n]}
        elif isinstance(pkldata, list):
            sample = pkldata[:first_n]
        elif isinstance(pkldata, np.ndarray):
            sample = pkldata.tolist()
        else:
            assert 0

        try:
            fout.write(json.dumps(sample))
        except:
            print('some error for', fname)
