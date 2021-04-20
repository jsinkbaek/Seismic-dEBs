"""
This collection of functions is used for general work on wavelength reduced échelle spectra.
"""
import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import alphashape
from localreg import *
import warnings


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


def AFS_algorithm(wavelength, flux, alpha=None, mf_window=5001, emline_factor=1, lr_frac=0.2):
    """
    https://arxiv.org/pdf/1904.10065v1.pdf
    https://iopscience.iop.org/article/10.3847/1538-3881/ab1b47/pdf
    https://ui.adsabs.harvard.edu/abs/2019AJ....157..243X/abstract
    Modeling the Echelle Spectra Continuum with Alpha Shapes and Local Regression Fitting

    Create alpha shape of spectrum. Then make local regression fit to upper boundary of alpha shape to estimate blaze.
    Divide spectrum with it. Then something more.

    :param wavelength:      np.ndarray of size (n, ) containing equispaced wavelengths in Ångstrom units.
    :param flux:            np.ndarray of size (n, ) containing flux values for corresponding wavelengths.
    :param alpha:           float, alpha value for alpha shape.
    :param mf_window:       int, index size of the median filter window used to reduce emission lines.
    :param emline_factor:   float, cutoff factor multipler to variance standard dev when reducing emission lines.
    :param lr_frac:         float, specifies fraction of all datapoints to include in width of kernel (varying kernel
                            width). Overrides lr_width if not None.
    :return:
    """
    # Reduce emission lines in data set used for alpha shape
    wavelength_uncorrected, flux_uncorrected = np.copy(wavelength), np.copy(flux)
    wavelength, flux = reduce_emission_lines(wavelength, flux, mf_window, emline_factor, plot=False)
    # Select alpha
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
    flux_localreg = localreg(upper_boundary[:, 0], upper_boundary[:, 1], degree=3, kernel=rbf.tricube,
                             frac=lr_frac)

    fig, ax = plt.subplots()
    ax.scatter(wavelength_uncorrected, flux_uncorrected, s=1, c='b')
    ax.scatter(*zip(*points), s=1)
    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    ax.plot(upper_boundary[:, 0], upper_boundary[:, 1], 'g')
    ax.plot(upper_boundary[:, 0], flux_localreg, 'k--')
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
        for i in range(0, int(np.floor(len(exclude_lims)/2)), 2):
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
        upper_boundary = np.delete(upper_boundary, range(upper_boundary[:, 0].size - 5, upper_boundary[:, 0].size),
                                   axis=0)
        flux_localreg = localreg(upper_boundary[:, 0], upper_boundary[:, 1], degree=3, kernel=rbf.tricube, frac=lr_frac)

        fig, ax = plt.subplots()
        ax.scatter(*zip(*points), s=1)
        ax.scatter(*zip(*points_reduced), s=1)
        ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
        ax.plot(upper_boundary[:, 0], upper_boundary[:, 1], 'g')
        ax.plot(lower_boundary[:, 0], lower_boundary[:, 1], 'r')
        ax.plot(upper_boundary[:, 0], flux_localreg, 'k--')
        plt.legend(['Full set with reduced emission lines', 'Reduced set by additional manual selection', 'Polygon',
                    'Upper boundary', 'Lower boundary', 'Local Polynomial Regression fit'])
        plt.show()

    # TODO: Make manual selection
    # TODO: Make data save
    # TODO: Remove negative values



