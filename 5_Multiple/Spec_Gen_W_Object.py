from object import Spectrum

spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=100)

spec.get_data()
spec.init_model()
spec.simulation()

spec.output()

spec.plot_results()
spec.plot_results(param='vel')
spec.plot_results(param='A')
