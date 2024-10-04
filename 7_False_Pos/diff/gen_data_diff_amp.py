# 
# Generate amplitude data for the ANN to learn from
# 

from spectrum_obj import Spectrum
import numpy as np
import sys
import os

spec_path = os.path.abspath('../..') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and fit synthetic spectra using the Spectrum object for both actual amplitudes and false positives for use when training an ANN.')
    parser.add_argument('-n', '--Nsim', type=int, help='The number of real spectra to generate.', default=10000)
    parser.add_argument('-f', '--false', type=int, help='The number of false positives to generate (default is the same as Nsim).', default=None)
    args = parser.parse_args()
    
    Nsim = args.Nsim
    if args.false == None:
        false = Nsim
    else:
        false = args.false
    
    Nsim = args.Nsim

    # Generate Spectrum object
    spec = Spectrum('../lines_in.txt', '../fitting.txt', Nsim=Nsim, AoN_min=3)

    # Import data
    print('Importing data')
    spec.get_data()

    # Generate actual amplitudes
    print("Generating full spectra")
    spec.simulation()
    AoN_out = spec.AoNs_out
    data_out = np.transpose(np.concatenate((AoN_out, np.ones((8, Nsim)))))

    
    print('Generating spectra with missing lines')
    spec.simulation_false()
    AoN_out_missing = spec.AoNs_out
    lines_present = spec.keep_lines

    # Save data
    missing_lines = np.transpose(np.concatenate((AoN_out_missing, lines_present)))
    
    data_out = np.concatenate((data_out, missing_lines))

    # Generate false positives
    print('Generating false positives')
    spec.AoN_min = 0
    spec.AoN_max = 0
    spec.Nsim = false
    spec.simulation()
    AoN_out = spec.AoNs_out

    # Save data
    empty_data = np.transpose(np.concatenate((AoN_out, np.zeros((8, false)))))
    data_out = np.concatenate((data_out, empty_data))
    

    files_present = [f for f in os.listdir() if os.path.isfile(os.path.join(".", f)) and f[:10] == "false_diff" and f[-4:] == ".txt"]
    file_out = f'false_diff{len(files_present)}.txt'
    
    np.savetxt(file_out, data_out)