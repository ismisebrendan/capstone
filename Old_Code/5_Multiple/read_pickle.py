import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('peak_data_out_vels_11.pickle', 'rb') as pickle_file:
    data = pickle.load(pickle_file)
    
Niter = data[-2]

# for i in range(11):
#     plt.scatter(data[i][3], (data[i][10][0] - data[i][9][0]) / data[i][9][0], s=1, label=f'v_in={data[i][12][0][0]}')
#     plt.legend()
#     plt.show()


stds = np.array([])
vels_in = np.array([])

for i in range(Niter):
    stds = np.append(stds, np.std(data[i][10]))
    vels_in = np.append(vels_in, data[i][12][0][0])
    
    
plt.scatter(vels_in, stds)