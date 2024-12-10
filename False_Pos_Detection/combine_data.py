import numpy as np
import os
import datetime


# Find all the files of data
folders = ['planetary_nebula', 'seyfert', 'star_forming', 'LINER']

# Number of files from each folder to combine, takes from start_file to end_file
start_file = 5
end_file = 37

files_in = []

for i in range(len(folders)):
    files_in.append([f for f in os.listdir(folders[i]) if os.path.isfile(os.path.join(folders[i], f)) and f[-4:] == '.txt' and ('confusion' and 'lines' and 'fitting' and 'to') not in f and int(f[-6:-4]) <= end_file and int(f[-6:-4]) >= start_file])
    


# Put them together
data_out = np.array([])


for i in range(len(files_in)):
    for j in range(len(files_in[i])):
        data_out = np.append(data_out, np.loadtxt(f'{folders[i]}/{files_in[i][j]}'))

data_out = np.reshape(data_out, (int(len(data_out)/16), 16))


if len(folders) == 1:
    fold = folders[0]
    files_present = [f for f in os.listdir(fold) if os.path.isfile(os.path.join(fold, f)) and f[-4:] == ".txt" and f'{fold}_all' in f]
    np.savetxt(f'{fold}/{fold}_{start_file}_to_{end_file}.txt', data_out, header=f'# {fold} region lines from files {start_file} to {end_file}')
elif len(folders) == 4:
    files_present = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f[-4:] == ".txt" and 'spectra_all' in f and '_0' or '_1' in f]
    np.savetxt(f'spectra_{start_file}_to_{end_file}.txt', data_out, header=f'# Lines from files {start_file} to {end_file} for all regions generated till {datetime.datetime.now()}')
else:
    files_present = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f[-4:] == ".txt" and 'spectra' in f and f'{"_".join(folders)}' in f]
    np.savetxt(f'spectra_{start_file}_to_{end_file}_{"_".join(folders)}_{len(files_present)}.txt', data_out, header=f'# Lines from files {start_file} to {end_file} from {" ".join(folders)}')
