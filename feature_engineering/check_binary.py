import os
import glob
import pandas as pd
import numpy as np

path = os.path.join(os.path.expanduser('~'), 'development', 'data', 'smallHybridUnnorm', '*')
#path = os.path.join(os.path.expanduser('~'), 'data', 'smallHybrid', '*')

files = glob.glob(path)
files.sort()
print(files)


for file in files:

    data = np.fromfile(file)
    n_row = int(data[0])
    n_col = int(data[1])
    data = np.delete(data, [0,1])
    data = np.reshape(data, (n_row, n_col))

    print("name", file)
    print("data.shape", data.shape)

    print(data[:20, :3])
