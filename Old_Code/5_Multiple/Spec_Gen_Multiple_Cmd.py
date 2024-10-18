from spectrum_obj import Spectrum
import numpy as np
import pickle
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and fit synthetic spectra using the Spectrum object.')
    parser.add_argument('-i', '--file_in', type=str, help='The file to take the input from.')
    parser.add_argument('-f', '--file_fit', type=str, help='The file to take the fitting data from.')
    parser.add_argument('--var', type=str, choices=['vel', 'sig'], help='Which of the sigma or velocity of the line to vary.')
    parser.add_argument('--min', type=float, help='The minimum for the value to be varied. default = 1.0', default=1.0)
    parser.add_argument('--max', type=float, help='The maximum for the value to be varied. default = 1000.0', default=1000.0)
    parser.add_argument('-t', '--iter', type=int, help='The numbe of different values of the varied quantity to simulate for. default = 1000', default=1000)
    parser.add_argument('-s', '--sim', type=int, help='The number of simulations to carry out. default = 1000', default=1000)
    args = parser.parse_args()

    file_in = args.file_in
    file_fit = args.file_fit
    variant = args.var
    min_val = args.min
    max_val = args.max
    Niter = args.iter
    Nsim = args.sim
        
    # Output data
    data_out = []
    
    # Which parameter is varied
    if variant == 'vel':
        param = 3
    elif variant == 'sig':
        param = 4
    else:
        raise ValueError('variant must be sig or vel')
    
    # Range of values
    vals = np.linspace(min_val, max_val, Niter)
    
    # Generate Spectrum object
    spec = Spectrum(file_in, file_fit, Nsim=Nsim)
    
    # Import data
    spec.get_data()

    # Loop
    for i in range(Niter):
        print(f'Simulation {i+1}')
        spec.overwrite(param, vals[i])
        spec.simulation(plotting=False)
        
        # Save the results
        data_out.append(spec.dump())
    
    # Save the data
    data_out.append(vals)
    data_out.append(Niter)
    
    with open('output_info.txt') as f:
        data_out.append(f.readlines())
    
    date = datetime.now()
    file_in = file_in.split('/')[1]
    file_in = file_in.split('.')[0]
    filename = f'output_files/{file_in}_{variant}_{Niter}-{date.strftime("%Y%m%dT%H%M%S")}.pickle'
    outfile = open(filename, 'wb')
    pickle.dump(data_out, outfile)
    print(f'Saving {filename}')
    outfile.close()