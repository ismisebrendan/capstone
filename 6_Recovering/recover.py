from spectrum_obj import Spectrum

spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=100)

# Import data
spec.get_data()

spec.simulation(plotting=True)

spec.plot_results(param='sig')
spec.plot_results_err(param='sig')
spec.plot_results(param='vel')
spec.plot_results_err(param='vel')
spec.plot_results(param='A')
spec.plot_results_err(param='A')
spec.plot_results(param='flux')
spec.plot_results_err(param='flux')