import numpy as np
import os
import datetime


# Find all the files of data
folders = ['LINER']

files_in = []

for i in range(len(folders)):
    files_in.append([f for f in os.listdir(folders[i]) if os.path.isfile(os.path.join(folders[i], f)) and f[-4:] == '.txt' and ('confusion' or 'lines' or 'fitting' or 'all') not in f])
    


# Put them together
data_out = np.array([])


for i in range(len(files_in)):
    for j in range(len(files_in[i])):
        data_out = np.append(data_out, np.loadtxt(f'{folders[i]}/{files_in[i][j]}'))

data_out = np.reshape(data_out, (int(len(data_out)/16), 16))


if len(folders) == 1:
    fold = folders[0]
    files_present = [f for f in os.listdir(fold) if os.path.isfile(os.path.join(fold, f)) and f[-4:] == ".txt" and f'{fold}_all' in f]
    np.savetxt(f'{fold}/{fold}_all_{len(files_present)}.txt', data_out, header=f'# All {fold} region lines')
elif len(folders) == 4:
    files_present = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f[-4:] == ".txt" and 'spectra_all' in f and '_0' or '_1' in f]
    np.savetxt(f'spectra_all_{len(files_present)}.txt', data_out, header=f'# All lines from all regions generated till {datetime.datetime.now()}')
else:
    files_present = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f[-4:] == ".txt" and 'spectra' in f and f'{"_".join(folders)}' in f]
    np.savetxt(f'spectra_{"_".join(folders)}_{len(files_present)}', data_out, header=f'# All lines from {" ".join(folders)}')
    
    
    
    
    
    
    # TEST THIS