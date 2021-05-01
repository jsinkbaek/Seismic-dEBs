"""
First edition on May 1, 2021.
@author Jeppe Sinkbæk Thomsen, Master's student in astronomy at Aarhus University.

Collection of functions for performing continuum normalization using the AFS algorithm (or a similar process) of
https://ui.adsabs.harvard.edu/abs/2019AJ....157..243X/abstract
    Modeling the Echelle Spectra Continuum with Alpha Shapes and Local Regression Fitting

Currently, data used is merged échelle spectra instead of the separate orders, so left-overs of the blaze-function is
not removed using this process.
"""
from descartes import PolygonPatch
import alphashape
import warnings
from localreg import *
from matplotlib.widgets import Cursor
from RV.library.spectrum_processing_functions import *


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


def AFS_merged_spectrum(wavelength, flux, alpha=None, mf_window=5001, emline_factor=1, lr_frac=0.2, save_string=None):
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
             + '_full_set.dat')
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

