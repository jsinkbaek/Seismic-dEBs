"""
This collection of functions is used for general work on wavelength reduced échelle spectra.
"""
import numpy as np
from astropy.io import fits
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import alphashape
from localreg import *
import warnings
import scipy.constants as scc
from scipy.interpolate import interp1d


def load_template_spectrum(template_spectrum_path):
    with fits.open(template_spectrum_path) as hdul:
        hdr = hdul[0].header
        flux = hdul[0].data
        wl0, delta_wl = hdr['CRVAL1'], hdr['CDELT1']
    wavelength = np.linspace(wl0, wl0+delta_wl*flux.size, flux.size)
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
    :param delta_v:                 desired spectrum resolution in velocity space
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
            wavelength_a = np.max([x[0] for x in wavelength])
        if wavelength_b is None and wavelength_resampled is None:
            wavelength_b = np.min([x[-1] for x in wavelength])
    elif isinstance(wavelength, np.ndarray):
        if wavelength_a is None and wavelength_resampled is None:
            wavelength_a = wavelength[0]
        if wavelength_b is None and wavelength_resampled is None:
            wavelength_b = wavelength[-1]
    else:
        raise ValueError("wavelength is neither a list (of arrays) or an array")

    if wavelength_resampled is None:
        step_amnt = np.log10(wavelength_b / wavelength_a) / np.log10(1.0 + delta_v / speed_of_light) + 1
        wavelength_resampled = wavelength_a * (1.0 + delta_v / speed_of_light) ** (np.linspace(1, step_amnt, step_amnt))
        if resampled_len_even and np.mod(wavelength_resampled.size, 2) != 0.0:
            wavelength_resampled = wavelength_resampled[:-1]    # all but last element

    if flux is not None:
        if isinstance(flux, list):
            flux_resampled_collection = np.empty(shape=(wavelength_resampled.size, len(flux)))
            for i in range(0, len(flux)):
                flux_interpolator = interp1d(wavelength[i], flux[i], kind='cubic')
                flux_resampled_collection[:, i] = flux_interpolator(wavelength_resampled)
            return wavelength_resampled, flux_resampled_collection
        elif isinstance(flux, np.ndarray):
            flux_interpolator = interp1d(wavelength, flux, kind='cubic')
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
        flux_interpolator = interp1d(wavelength_collector_list[i], flux_collector_list[i], kind='cubic')
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


def separate_polygon_boundary(polygon):
    """
    Separates a polygon to its boundary, and into an upper and lower part, for use in the AFS algorithm.
    :param polygon:     shapely.geometry.polygon.Polygon encasing the spectrum data
    :return:            upper_boundary, lower_boundary. Both np.ndarray of data points shape (xs, ys)
    """
    boundary_coords = np.array(polygon.boundary.coords)
    idx_left  = np.argmin(boundary_coords[:, 0])         # index of furthermost left point (on x-axis)
    idx_right = np.argmax(boundary_coords[:, 0])         # index of futhermost right point
    if idx_left < idx_right:
        idx_min, idx_max = idx_left, idx_right
    elif idx_right < idx_left:
        idx_min, idx_max = idx_right, idx_left
    else:
        raise ValueError("Both left and right side index are equal in polygon.")

    boundary_part1 = boundary_coords[idx_min:idx_max+1, :]
    boundary_part2 = np.concatenate([boundary_coords[idx_max:, :], boundary_coords[:idx_min+1, :]])

    if np.sum(boundary_part1[:, 1]) > np.sum(boundary_part2[:, 1]):
        upper_boundary = boundary_part1
        lower_boundary = boundary_part2
    elif np.sum(boundary_part2[:, 1]) > np.sum(boundary_part1[:, 1]):
        upper_boundary = boundary_part2
        lower_boundary = boundary_part1
    else:
        raise ValueError("Both boundary parts give same sum, which should not be possible.")

    return upper_boundary, lower_boundary


def reduce_emission_lines(wavelength, flux, mf_window=401, std_factor=1.5, plot=False):
    """
    Reduces the effect of emission lines in a spectrum by median-filtering and cutting away all data points a certain
    point ABOVE the variance of the data (flux - median flux).
    :param wavelength:              np.ndarray size (n, ) of wavelengths.
    :param flux:                    np.ndarray size (n, ) of fluxes for the spectrum
    :param mf_window:               int, window size of the median filter. Must be uneven.
    :param std_factor:              float, cutoff factor multiplier to the standard deviation of the variance.
    :param plot:                    bool, indicates whether illustrative plots should be shown
    :return wavelength_reduced:     np.ndarray of wavelengths, masked to the reduced fluxes.
            flux_emission_reduced:  np.ndarray of fluxes for the spectrum, with reduced emission lines.
    """
    flux_filtered = moving_median_filter(flux, mf_window)
    variance = flux - flux_filtered
    mask = variance < std_factor * np.std(variance)
    wavelength_reduced = wavelength[mask]
    flux_emission_reduced = flux[mask]

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


def select_areas(fig, ax):
    """
    Select areas on figure using pyplot.ginput.
    Expects figure to be created with data beforehand. Adds the option to zoom to dataset.
    Inspiration and code snippets from:
    https://scientificallysound.org/2017/12/28/moving-around-in-a-matplotlib-figure-before-selecting-point-of-interest/
    :return: selected_areas, list of tuples with (x, y) coordinates
    """
    repeat = True
    selected_areas = []
    while repeat:
        cursor = Cursor(ax, useblit=True, color='k', linewidth=1)
        zoom_ok = False
        print('\nZoom or pan to view, \npress spacebar when ready to click:\n')
        while not zoom_ok:
            zoom_ok = plt.waitforbuttonpress()
        print('Left click to select points (use 2 for each interval). '
              'Right click to remove last point. Middle click to exit. Zoom and select can be repeated after this.')
        selection = plt.ginput(mouse_add=1, mouse_pop=3, mouse_stop=2, n=-1, timeout=0, show_clicks=True)
        while True:
            inpt = input('Do you want to select more intervals? (y/n)')
            if inpt == 'y':
                break
            elif inpt == 'n':
                repeat = False
                break
            else:
                print('Wrong input, repeats prompt.')
                pass
        selected_areas = selected_areas + selection
    plt.close(fig)
    return selected_areas


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


def AFS_algorithm(wavelength, flux, alpha=None, mf_window=5001, emline_factor=1, lr_frac=0.2, save_string=None):
    """
    https://arxiv.org/pdf/1904.10065v1.pdf
    https://iopscience.iop.org/article/10.3847/1538-3881/ab1b47/pdf
    https://ui.adsabs.harvard.edu/abs/2019AJ....157..243X/abstract
    Modeling the Echelle Spectra Continuum with Alpha Shapes and Local Regression Fitting

    This is a continuum normalization algorithm for use with a merged spectrum (even though the articles mention the
    algorithm for use with unmerged spectra where blaze can be removed).
    Create alpha shape of spectrum. Then make local regression fit to upper boundary of alpha shape to estimate blaze.
    Divide spectrum with it. Then something more.

    :param wavelength:      np.ndarray of size (n, ) containing equispaced wavelengths in Ångstrom units.
    :param flux:            np.ndarray of size (n, ) containing flux values for corresponding wavelengths.
    :param alpha:           float, alpha value for alpha shape.
    :param mf_window:       int, index size of the median filter window used to reduce emission lines.
    :param emline_factor:   float, cutoff factor multipler to variance standard dev when reducing emission lines.
    :param lr_frac:         float, specifies fraction of all datapoints to include in width of kernel (varying kernel
                            width). Overrides lr_width if not None.
    :param save_string:     string. If not None, data is saved using "save_string" as the beginning name in the folder
                            RV/Data/processed/AFS_algorithm/
    :return:
    """
    # Set processed data and figure path
    data_out_path = 'RV/Data/processed/AFS_algorithm/'
    fig_path = 'figures/report/RV/Continuum_Normalization/'
    # Convert wavelength to ln(wl)
    wavelength = np.log(wavelength)

    # Reduce emission lines in data set used for alpha shape
    wavelength_uncorrected, flux_uncorrected = np.copy(wavelength), np.copy(flux)
    wavelength, flux = reduce_emission_lines(wavelength, flux, mf_window, emline_factor, plot=False)
    # Select alpha (assumes wavelength in ln units)
    if alpha is None:
        alpha = (np.exp(np.max(wavelength)) - np.exp(np.min(wavelength)))/(6*50)
        print(alpha)
    # Rescale flux
    u = (np.max(wavelength)-np.min(wavelength)) / (10 * np.max(flux))
    flux = u * flux
    flux_uncorrected = u * flux_uncorrected

    # Set up points for alpha shape
    points = np.transpose(np.array([wavelength, flux]))
    # Create alpha shape
    alpha_shape = alphashape.alphashape(points, alpha)
    # Separate boundary into upper and lower part
    upper_boundary, lower_boundary = separate_polygon_boundary(alpha_shape)
    # Cut off ends before fitting to improve results in ends
    upper_boundary = np.delete(upper_boundary, range(0, 5), axis=0)
    upper_boundary = np.delete(upper_boundary, range(upper_boundary[:, 0].size-5, upper_boundary[:, 0].size), axis=0)
    # Do local polynomial regression
    flux_localreg = localreg(upper_boundary[:, 0], upper_boundary[:, 1], wavelength, degree=3, kernel=rbf.tricube,
                             frac=lr_frac)
    flux_lrec_uncorr = localreg(upper_boundary[:, 0], upper_boundary[:, 1], wavelength_uncorrected, degree=3,
                                kernel=rbf.tricube, frac=lr_frac)       # interpolated to match wavelength_uncorrected

    # Plot result
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(wavelength_uncorrected, flux_uncorrected, s=1, c='b')
    ax.scatter(*zip(*points), s=1)
    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    ax.plot(upper_boundary[:, 0], upper_boundary[:, 1], 'g')
    ax.plot(wavelength, flux_localreg, 'k--')
    ax.plot(lower_boundary[:, 0], lower_boundary[:, 1], 'r')

    # Call ginput to manually exclude areas (space to add point, backspace to remove, enter to return)
    wavelength_reduced = wavelength
    flux_reduced = flux
    exclude_lims = select_areas(fig, ax)
    equal_len = True
    if exclude_lims:        # if not empty
        if np.floor(len(exclude_lims)/2) != np.ceil(len(exclude_lims)/2):
            warnings.warn("Uneven amount of ginput points selected. Ignoring last point.")
            equal_len = False
        for i in range(0, len(exclude_lims), 2):
            if not equal_len and i==np.floor(len(exclude_lims))-1:
                break
            x1 = exclude_lims[i][0]
            x2 = exclude_lims[i+1][0]
            exclude_mask = (wavelength_reduced > x1) & (wavelength_reduced < x2)
            wavelength_reduced = wavelength_reduced[~exclude_mask]
            flux_reduced = flux_reduced[~exclude_mask]

        # Recreate alpha shape with reduced points
        points_reduced = np.transpose(np.array([wavelength_reduced, flux_reduced]))
        alpha_shape = alphashape.alphashape(points_reduced, alpha)
        upper_boundary, lower_boundary = separate_polygon_boundary(alpha_shape)
        upper_boundary = np.delete(upper_boundary, range(0, 5), axis=0)
        upper_boundary = np.delete(upper_boundary, range(upper_boundary[:, 0].size-5, upper_boundary[:, 0].size),axis=0)
        flux_localreg = localreg(upper_boundary[:, 0], upper_boundary[:, 1], wavelength_reduced, degree=3,
                                 kernel=rbf.tricube, frac=lr_frac)
        flux_lrec_uncorr = localreg(upper_boundary[:, 0], upper_boundary[:, 1], wavelength_uncorrected, degree=3,
                                    kernel=rbf.tricube, frac=lr_frac)  # interpolated to match wavelength_uncorrected

        # Plot result
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.scatter(*zip(*points), s=1)
        ax.scatter(*zip(*points_reduced), s=1)
        ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
        ax.plot(upper_boundary[:, 0], upper_boundary[:, 1], 'g')
        ax.plot(lower_boundary[:, 0], lower_boundary[:, 1], 'r')
        ax.plot(wavelength_reduced, flux_localreg, 'k--')
        plt.legend(['Full set with reduced emission lines', 'Reduced set by additional manual selection', 'Polygon',
                    'Upper boundary', 'Lower boundary', 'Local Polynomial Regression fit'], loc='upper left')
        plt.show()

    # Correct spectrum using local regression fit as continuum
    if exclude_lims:
        flux_normalized = flux_reduced / flux_localreg
        wavelength_normalized = wavelength_reduced
    else:
        flux_normalized = flux / flux_localreg
        wavelength_normalized = wavelength

    flux_normalized_full_set = flux_uncorrected / flux_lrec_uncorr
    wavelength_full_set = wavelength_uncorrected

    # Remove negative values
    mask_negative_values = flux_normalized < 0
    mask_negative_values_full_set = flux_normalized_full_set < 0

    flux_normalized = flux_normalized[~mask_negative_values]
    wavelength_normalized = wavelength_normalized[~mask_negative_values]
    flux_localreg = flux_localreg[~mask_negative_values]

    flux_normalized_full_set = flux_normalized_full_set[~mask_negative_values_full_set]
    wavelength_full_set = wavelength_full_set[~mask_negative_values_full_set]

    # Convert wavelength back to Ångstrom units
    wavelength, wavelength_reduced, = np.exp(wavelength), np.exp(wavelength_reduced)
    wavelength_normalized, wavelength_uncorrected = np.exp(wavelength_normalized), np.exp(wavelength_uncorrected)
    wavelength_full_set = np.exp(wavelength_full_set)

    # Save data
    save2col(wavelength_normalized, flux_normalized,data_out_path+'Normalized_Spectrum/'+save_string+'_reduced_set.dat')
    save2col(wavelength_full_set, flux_normalized_full_set, data_out_path+'Normalized_Spectrum/'+save_string
                      +'_full_set.dat')
    save2col(wavelength_uncorrected, flux_uncorrected,data_out_path+'Uncorrected_Spectrum/'+save_string+'_full_set.dat')
    if exclude_lims:
        save2col(wavelength_reduced, flux_reduced, data_out_path+'Uncorrected_Spectrum/'+save_string+'_reduced_set.dat')
    else:
        save2col(wavelength, flux, data_out_path + 'Uncorrected_Spectrum/'+save_string+'_reduced_set.dat')
    save2col(wavelength_normalized, flux_localreg, data_out_path+'Continuum_Fit/'+ save_string+'.fit')
    save2col(upper_boundary[:,0], upper_boundary[:,1], data_out_path+'Polygon_Boundary/'+save_string+'.upper')
    save2col(lower_boundary[:,0], lower_boundary[:,1], data_out_path+'Polygon_Boundary/'+save_string+'.lower')

    plot_xs = [wavelength_uncorrected, wavelength_reduced, wavelength_normalized]
    plot_ys = [flux_uncorrected, flux_reduced, flux_localreg]
    savefig(fig_path+save_string, plot_xs, plot_ys, xlabel=r'ln($\lambda$)', ylabel='', legend=['Full set',
            'Set with reduced and removed emission lines', 'Local Polyinomial Regression Fit'])

    return wavelength_normalized, flux_normalized, flux_localreg


