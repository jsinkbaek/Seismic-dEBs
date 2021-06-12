"""
This collection of functions is used for general work on wavelength reduced échelle spectra.
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.constants as scc
from scipy.interpolate import interp1d


def load_template_spectrum(template_spectrum_path):
    with fits.open(template_spectrum_path) as hdul:
        hdr = hdul[0].header
        flux = hdul[0].data
        wl0, delta_wl, naxis = hdr['CRVAL1'], hdr['CDELT1'], hdr['NAXIS1']
    wavelength = np.linspace(wl0, wl0+delta_wl*naxis, naxis)
    return wavelength, flux


def load_program_spectrum(program_spectrum_path):
    with fits.open(program_spectrum_path) as hdul:
        hdr = hdul[0].header
        flux = hdul[0].data
        wl0, delta_wl = hdr['CRVAL1'], hdr['CDELT1']
        RA, DEC = hdr['OBJRA'], hdr['OBJDEC']
        date = hdr['DATE-AVG']
    wavelength = np.linspace(wl0, wl0 + delta_wl * flux.size, flux.size)
    return wavelength, flux, date, RA, DEC


def resample_to_equal_velocity_steps(wavelength, delta_v, flux=None, wavelength_resampled=None, wavelength_a=None,
                                     wavelength_b=None, resampled_len_even=True):
    """
    Over-engineered, multi-use function that can be used to either: Create a new wavelength grid equi-distant in vel
    (without interpolation), create grid and resample flux to it, create mutual grid and resample flux for multiple
    spectra simultanously. It can also resample one (or more) spectra to a provided wavelength grid.
    See interpolate_to_equal_velocity_steps() for simpler illustration of what this function does.
    :param wavelength:              either a list of np.ndarray's, or a np.ndarray
    :param delta_v:                 desired spectrum resolution in velocity space in km/s
    :param flux:                    either a list of np.ndarray's, or a np.ndarray. Set to None if not used
    :param wavelength_resampled:    a np.ndarray with a provided resampled grid. Set to None if one should be calculated
    :param wavelength_a:            float, start of desired grid. If None, calculates from provided spectra
    :param wavelength_b:            float, end of desired grid. If None, calculates from provided spectra.
    :param resampled_len_even:      bool switch, sets if resampled grid should be kept on an even length
    :return:    either wavelength_resampled, (wavelength_resampled, flux_resampled_collection),
                or (wavelength_resampled, flux_resampled)
    """
    speed_of_light = scc.c / 1000  # in km/s
    if isinstance(wavelength, list):
        if wavelength_a is None and wavelength_resampled is None:
            wavelength_a = np.max([np.min(x) for x in wavelength])
        if wavelength_b is None and wavelength_resampled is None:
            wavelength_b = np.min([np.max(x) for x in wavelength])
    elif isinstance(wavelength, np.ndarray):
        if wavelength_a is None and wavelength_resampled is None:
            wavelength_a = np.min(wavelength)
        if wavelength_b is None and wavelength_resampled is None:
            wavelength_b = np.max(wavelength)
    else:
        raise ValueError("wavelength is neither a list (of arrays) or an array")

    if wavelength_resampled is None:
        step_amnt = int(np.log10(wavelength_b / wavelength_a) / np.log10(1.0 + delta_v / speed_of_light))
        wavelength_resampled = wavelength_a * (1.0 + delta_v / speed_of_light) ** (np.linspace(1, step_amnt, step_amnt))

        if resampled_len_even and np.mod(wavelength_resampled.size, 2) != 0.0:
            wavelength_resampled = wavelength_resampled[:-1]    # all but last element

    if flux is not None:
        if isinstance(flux, list):
            flux_resampled_collection = np.empty(shape=(wavelength_resampled.size, len(flux)))
            for i in range(0, len(flux)):
                flux_interpolator = interp1d(wavelength[i], flux[i], kind='linear')
                flux_resampled_collection[:, i] = flux_interpolator(wavelength_resampled)
            return wavelength_resampled, flux_resampled_collection
        elif isinstance(flux, np.ndarray):
            flux_interpolator = interp1d(wavelength, flux, kind='linear')
            flux_resampled = flux_interpolator(wavelength_resampled)
            return wavelength_resampled, flux_resampled
        else:
            raise ValueError("flux is neither a list (of arrays), an array, or None.")
    else:
        return wavelength_resampled


def interpolate_to_equal_velocity_steps(wavelength_collector_list, flux_collector_list, delta_v):
    """
    Resamples a set of spectra to the same wavelength grid equi-spaced in velocity map.
    :param wavelength_collector_list:   list of arrays, one array for each spectrum
    :param flux_collector_list:         list of arrays, one array for each spectrum
    :param delta_v:                     interpolation resolution for spectrum in km/s
    :return: wavelength, flux_collector_array
    """
    speed_of_light = scc.c / 1000       # in km/s

    # # Create unified wavelength grid equispaced in velocity # #
    wavelength_a = np.max([x[0] for x in wavelength_collector_list])
    wavelength_b = np.min([x[-1] for x in wavelength_collector_list])
    step_amnt = np.log10(wavelength_b / wavelength_a) / np.log10(1.0 + delta_v / speed_of_light) + 1
    wavelength = wavelength_a * (1.0 + delta_v / speed_of_light) ** (np.linspace(1, step_amnt, step_amnt))

    # # Interpolate to unified wavelength grid # #
    flux_collector_array = np.empty(shape=(wavelength.size, len(flux_collector_list)))
    for i in range(0, len(flux_collector_list)):
        flux_interpolator = interp1d(wavelength_collector_list[i], flux_collector_list[i], kind='linear')
        flux_collector_array[:, i] = flux_interpolator(wavelength)

    return wavelength, flux_collector_array


def moving_median_filter(flux, window=51):
    """
    Filters the data using a moving median filter. Useful to capture some continuum trends.
    :param flux:        np.ndarray size (n, ) of y-values (fluxes) for each wavelength
    :param window:      integer, size of moving median filter
    :return:            filtered_flux, np.ndarray size (n, )
    """
    from scipy.ndimage import median_filter
    return median_filter(flux, size=window, mode='reflect')


def reduce_emission_lines(wavelength, flux, mf_window=401, std_factor=1.5, plot=False, limit=None):
    """
    Reduces the effect of emission lines in a spectrum by median-filtering and cutting away all data points a certain
    point ABOVE the median variance of the data (flux - median flux).
    :param wavelength:              np.ndarray size (n, ) of wavelengths.
    :param flux:                    np.ndarray size (n, ) of fluxes for the spectrum
    :param mf_window:               int, window size of the median filter. Must be uneven.
    :param std_factor:              float, cutoff factor multiplier to the standard deviation of the variance.
    :param plot:                    bool, indicates whether illustrative plots should be shown
    :param limit:                   float, max distance from variance allowed. Default is None
    :return wavelength_reduced:     np.ndarray of wavelengths, masked to the reduced fluxes.
            flux_emission_reduced:  np.ndarray of fluxes for the spectrum, with reduced emission lines.
    """
    flux_filtered = moving_median_filter(flux, mf_window)
    variance = flux - flux_filtered
    mask = variance < std_factor * np.std(variance)
    wavelength_reduced = wavelength[mask]
    flux_emission_reduced = flux[mask]
    if limit is not None:
        mask = variance[mask] < limit
        wavelength_reduced = wavelength_reduced[mask]
        flux_emission_reduced = flux_emission_reduced[mask]

    if plot:
        plt.figure()
        plt.plot(wavelength, flux); plt.plot(wavelength, flux_filtered)
        plt.xlabel('Wavelength'); plt.ylabel('Flux'); plt.legend(['Unfiltered', 'Median Filtered'])
        plt.show(block=False)

        plt.figure()
        plt.plot(wavelength, variance); plt.plot(wavelength_reduced, variance[mask])
        plt.plot([np.min(wavelength), np.max(wavelength)], [std_factor*np.std(variance), std_factor*np.std(variance)],
                 'k')
        plt.plot([np.min(wavelength), np.max(wavelength)],
                 [-std_factor * np.std(variance), -std_factor * np.std(variance)], 'k--')
        plt.xlabel('Wavelength'); plt.ylabel('Variance'); plt.legend(['Full set', 'Set with reduced emission lines'])
        plt.show(block=False)

        plt.figure()
        plt.plot(wavelength, flux, linewidth=1)
        plt.plot(wavelength_reduced, flux_emission_reduced, '--', linewidth=0.8)
        plt.xlabel('Wavelength'); plt.ylabel('Flux'); plt.legend(['Full set', 'Set with reduced emission lines'])
        plt.show()

    return wavelength_reduced, flux_emission_reduced


def save2col(column1, column2, filename):
    save_data = np.empty((column1.size, 2))
    save_data[:, 0] = column1
    save_data[:, 1] = column2
    np.savetxt(filename, save_data)


def savefig(filename, xs, ys, xlabel, ylabel, title=None, xlim=None, ylim=None, legend=None, legend_loc='upper left',
            figsize=(17.78, 10), dpi=400):
    fig = plt.figure(figsize=figsize)
    for i in range(0, len(xs)):
        plt.plot(xs[i], ys[i], linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if title is not None:
        plt.title(title)
    if legend is not None:
        plt.legend(legend, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)




