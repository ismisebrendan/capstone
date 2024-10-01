# 
# Generate amplitude data for the ANN to learn from
# 

from spectrum_obj import Spectrum
import numpy as np
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and fit synthetic spectra using the Spectrum object for both actual amplitudes and false positives for use when training an ANN.')
    parser.add_argument('-n', '--Nsim', type=int, help='The number of real spectra to generate.', default=10000)
    parser.add_argument('-o', '--file_out', type=str, help='The filename to save this data to.', default=None)
    args = parser.parse_args()
    
    Nsim = args.Nsim

    # Generate Spectrum object
    spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=Nsim, AoN_min=3)

    # Import data
    print('Importing data')
    spec.get_data()

    # Generate actual amplitudes
    print('Generating spectral data')
    spec.simulation_false()
    AoN_out = spec.AoNs_out
    lines_present = spec.keep_lines

    # Save data
    data_out = np.transpose(np.concatenate((AoN_out, lines_present)))


    if args.file_out == None:
        date = datetime.now()
        file_out = f'false_pos_diff-{date.strftime("%Y%m%dT%H%M%S")}.txt'
    else:
        file_out = args.file_out
    
    np.savetxt(file_out, data_out)