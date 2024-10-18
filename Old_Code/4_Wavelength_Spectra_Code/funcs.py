import numpy as np

# The speed of light
c = 299792.458 #km/s

def background(x, bkg):
    """
    Generates the background level of the spectrum.

    Parameters
    ----------
    lam : array
        The wavelength range over which the background should be generated.
    bkg : float
        The level of the background spectrum.

    Returns
    -------
    array
        The background level for each of the input wavelength values.

    """
    return x*0 + bkg


def gaussian(x, A, lam_rf, vel, sig, sig_resolution):
    """
    Produces a Gaussian curve with a background level.
    
    Parameters
    ----------
    x : float
        The wavelength range over which the Gaussian is generated.
    A : float
        The amplitude of the Gaussian.
    lam_rf : float
        The wavelength of the peak in its rest frame in Angstrom.
    vel : float
        The relative velocity of the source in km/s.
    sig : float
        The sigma of the Gaussian in km/s.
    sig_resolution : float
        The resolution of the detector in Angstrom.

    Returns
    -------
    array
        The Gaussian.

    """
    lam_obs = lam_rf * (1 + vel/c)
    sig_intr = sig / c * lam_obs
    sig_obs = np.sqrt(sig_intr**2 + sig_resolution**2)
    return A * np.exp(-0.5*(x - lam_obs)**2 / sig_obs**2)