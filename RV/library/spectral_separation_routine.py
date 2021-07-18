"""
First edition May 02/2021.
@author Jeppe Sinkbæk Thomsen, Master's thesis studen at Aarhus University.
Supervisor: Karsten Frank Brogaard.

This is a collection of functions that form a routine to perform spectral separation of detached eclipsing binaries
with a giant component and a main sequence component. The routine is adapted from multiple scripts, an IDL script by
Karsten Frank Brogaard named "dis_real_merged_8430105_2021.pro", and a similar python script "spec_8430105_bf.py" by the
same author. Both follows the formula layout of the article:
'Separation of composite spectra: the spectroscopic detecton of an eclipsing binary star'
        by J.F. Gonzalez and H. Levato ( A&A 448, 283-292(2006) )

Other module files which this code uses are also adapted from other sources, including the shazam library for the
SONG telescope (written by Emil Knudstrup).
However, multiple people have been over shazam.py, including Karsten Frank Brogaard and Frank Grundahl, and it itself is
based on previous implementations by J. Jessen Hansen and others.
"""

from RV.library.calculate_radial_velocities import radial_velocity_single_component
from RV.library.broadening_function_svd import *
from RV.library.initial_fit_parameters import InitialFitParameters
from lmfit.minimizer import MinimizerResult
import RV.library.spectrum_processing_functions as spf
from copy import deepcopy


def shift_spectrum(flux, radial_velocity_shift, delta_v):
    shift = int(radial_velocity_shift / delta_v)
    return np.roll(flux, shift)


def separate_component_spectra(
        flux_collection, radial_velocity_collection_A, radial_velocity_collection_B, delta_v, convergence_limit,
        suppress_scs=False, max_iterations=20, rv_proximity_limit=0.0, rv_lower_limit=0.0, use_spectra=True or np.ndarray
):
    """
    Assumes that component A is the dominant component in the spectrum. Attempts to separate the two components using
    RV shifts and averaged spectra.

    :param flux_collection:               np.ndarray shape (datasize, nspectra) of all the observed spectra
    :param radial_velocity_collection_A:  np.ndarray shape (nspectra, ) of radial velocity values for component A
    :param radial_velocity_collection_B:  np.ndarray shape (nspectra, ) of radial velocity values for component B
    :param delta_v:                       float, the sampling size of the spectra in velocity space
    :param convergence_limit:             float, the precision needed to break while loop
    :param suppress_scs:                  bool, indicates if printing should be suppressed
    :param max_iterations:                int, maximum number of allowed iterations before breaking loop
    :param rv_proximity_limit:            float, RV limit in order to add separated spectrum (if closer, the
                                          components are expected to be mixed, and are not included to avoid pollution)
    :param rv_lower_limit:                float. RV limit using only component A. Overrides rv_proximity_limit if also
                                          provided. Useful if RV for B is unstable.
    :param use_spectra:                   bool, np.ndarray. If True, use all spectra for calculating the
                                            separated spectra. If np.ndarray or list, this should indicate the indices
                                            of the spectra to use from 0 to n_spectra-1 in flux_collection[:, i].
                                            Overrides use of rv_lower_limit and rv_proximity_limit

    :return separated_flux_A, separated_flux_B:   the separated and meaned total component spectra of A and B.
    """
    n_spectra = flux_collection[0, :].size
    separated_flux_B = np.zeros((flux_collection[:, 0].size,))  # Set to 0 before iteration
    separated_flux_A = np.zeros((flux_collection[:, 0].size,))

    iteration_counter = 0
    while True:
        RMS_values_A = -separated_flux_A
        RMS_values_B = -separated_flux_B
        iteration_counter += 1
        separated_flux_A = np.zeros((flux_collection[:, 0].size,))
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (isinstance(use_spectra, bool) and use_spectra is True) or (isinstance(use_spectra, np.ndarray) and
                                                                           i in use_spectra):
                rvA = radial_velocity_collection_A[i]
                rvB = radial_velocity_collection_B[i]
                if rv_lower_limit == 0.0 and not isinstance(use_spectra, np.ndarray):
                    condition = np.abs(rvA-rvB) > rv_proximity_limit
                elif not isinstance(use_spectra, np.ndarray):
                    condition = np.abs(rvA) > rv_lower_limit
                else:
                    condition = True

                if condition:
                    shifted_flux_A = shift_spectrum(flux_collection[:, i], -rvA, delta_v)
                    separated_flux_A += shifted_flux_A - shift_spectrum(separated_flux_B, rvB - rvA, delta_v)
                    n_used_spectra += 1
            elif isinstance(use_spectra, np.ndarray) and use_spectra.size != 0 and i not in use_spectra:
                pass
            else:
                raise TypeError(f'use_spectra is either of wrong type ({type(use_spectra)}), empty, or wrong value.\n' +
                                f'Expected type: {type(True)} or np.ndarray. Expected value if bool: True')
        separated_flux_A = separated_flux_A / n_used_spectra

        separated_flux_B = np.zeros((flux_collection[:, 0].size,))
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (isinstance(use_spectra, bool) and use_spectra is True) or (isinstance(use_spectra, np.ndarray) and
                                                                           i in use_spectra):
                rvA = radial_velocity_collection_A[i]
                rvB = radial_velocity_collection_B[i]
                if rv_lower_limit == 0.0 and not isinstance(use_spectra, np.ndarray):
                    condition = np.abs(rvA-rvB) > rv_proximity_limit
                elif not isinstance(use_spectra, np.ndarray):
                    condition = np.abs(rvA) > rv_lower_limit
                else:
                    condition = True

                if condition:
                    shifted_flux_B = shift_spectrum(flux_collection[:, i], -rvB, delta_v)
                    separated_flux_B += shifted_flux_B - shift_spectrum(separated_flux_A, rvA - rvB, delta_v)
                    n_used_spectra += 1
            elif isinstance(use_spectra, np.ndarray) and use_spectra.size != 0 and i not in use_spectra:
                pass
            else:
                raise TypeError(f'use_spectra is either of wrong type ({type(use_spectra)}), empty, or wrong value.\n' +
                                f'Expected type: {type(True)} or np.ndarray. Expected value if bool: True')
        separated_flux_B = separated_flux_B / n_used_spectra

        RMS_values_A += separated_flux_A
        RMS_values_B += separated_flux_B
        RMS_A = np.sum(RMS_values_A**2)/RMS_values_A.size
        RMS_B = np.sum(RMS_values_B**2)/RMS_values_B.size
        if RMS_A < convergence_limit and RMS_B < convergence_limit:
            if not suppress_scs:
                print(f'Separate Component Spectra: Convergence limit of {convergence_limit} successfully reached in '
                      f'{iteration_counter} iterations. \nReturning last separated spectra.')
            break
        elif iteration_counter >= max_iterations:
            warnings.warn(f'Warning: Iteration limit of {max_iterations} reached without reaching convergence limit'
                          f' of {convergence_limit}. \nCurrent RMS_A: {RMS_A}. RMS_B: {RMS_B} \n'
                          'Returning last separated spectra.')
            break
    if not suppress_scs:
        print('n_spectra vs n_used_spectra: ', n_spectra, ' ', n_used_spectra)
    return separated_flux_A, separated_flux_B


def update_bf_plot(plot_ax, model, index):
    fit = model[0]
    model_values = model[1]
    velocity_values = model[2]
    bf_smooth_values = model[4]
    amplitude, RV, _, _, _, _ = get_fit_parameter_values(fit.params)
    plot_ax.plot(velocity_values, 1+0.02*bf_smooth_values/np.max(bf_smooth_values)-0.05*index)
    plot_ax.plot(velocity_values, 1+0.02*model_values/np.max(bf_smooth_values)-0.05*index, 'k--')
    plot_ax.plot(np.ones(shape=(2,))*RV, [1-0.05*index-0.005,
                                          1+0.025*np.max(model_values)/np.max(bf_smooth_values)-0.05*index],
                 color='grey')


def recalculate_RVs(
        inv_flux_collection: np.ndarray, separated_flux_A: np.ndarray, separated_flux_B: np.ndarray,
        RV_collection_A: np.ndarray, RV_collection_B: np.ndarray, inv_flux_templateA: np.ndarray,
        inv_flux_templateB: np.ndarray, delta_v: float, ifitparamsA:InitialFitParameters,
        ifitparamsB:InitialFitParameters, buffer_mask: np.ndarray, iteration_limit=10, convergence_limit=1e-5,
        plot_ax_A=None, plot_ax_B=None, plot_ax_d1=None, plot_ax_d2=None, rv_lower_limit=0.0
):
    """
    This part of the spectral separation routine corrects the spectra for the separated spectra found by
    separate_component_spectra and recalculates RV values for each component using the corrected spectra (with one
    component removed).

    :param inv_flux_collection: np.ndarray shape (:, n_spectra). Collection of inverted fluxes for the program spectra.
    :param separated_flux_A:    np.ndarray shape (:, ). Meaned inverted flux from separate_component_spectra() for A.
    :param separated_flux_B:    np.ndarray shape (:, ). Meaned inverted flux from separate_component_spectra() for B.
    :param RV_collection_A:     np.ndarray shape (n_spectra, ). Current RV values used to remove A from spectrum with.
    :param RV_collection_B:     np.ndarray shape (n_spectra, ). Current RV values used to remove B from spectrum with.
    :param inv_flux_templateA:  np.ndarray shape (:, ). Template spectrum inverted flux for component A.
    :param inv_flux_templateB:  np.ndarray shape (:, ). Template spectrum inverted flux for component B.
    :param delta_v:             float. The sampling delta_v of the velocity grid (resolution). Example 1.0 km/s
    :param ifitparamsA:         InitialFitParameters. Fitting and broadening function parameters for component A.
    :param ifitparamsB:         InitialFitParameters. Fitting and broadening function parameters for component B.
    :param buffer_mask:         np.ndarray shape (:, ). Mask used to remove "buffer" (or "padding") from spectrum.
                                    See spectral_separation_routine() for more info.
    :param iteration_limit:     int, number of allowed iterations for each spectrum.
    :param convergence_limit:   float, convergence limit for RV changes during loop.
    :param plot_ax_A:           matplotlib.axes.axes. Used to update RV plots during iterations.
    :param plot_ax_B:           matplotlib.axes.axes. Used to update RV plots during iterations.
    :param plot_ax_d1:          matplotlib.axes.axes. Extra plot to examine BF A in detail for 1 or 2 spectra.
    :param plot_ax_d2:          matplotlib.axes.axes. Extra plot to examine BF B in detail for 1 or 2 spectra.
    :param rv_lower_limit:      float. Lower limit for the RV. See main routine for details. Only used for plots here.

    :return:    RV_collection_A,        RV_collection_B, (fits_A, fits_B)
                RV_collection_A:        updated values for the RV of component A.
                RV_collection_B:        updated values for the RV of component B.
                fits_A, fits_B:         np.ndarrays storing the rotational BF profile fits found for each spectrum.
    """
    RV_collection_A = deepcopy(RV_collection_A)
    RV_collection_B = deepcopy(RV_collection_B)
    n_spectra = inv_flux_collection[0, :].size
    v_span = ifitparamsA.bf_velocity_span
    BRsvd_template_A = BroadeningFunction(inv_flux_collection[~buffer_mask, 0],
                                          inv_flux_templateA[~buffer_mask], v_span, delta_v)
    BRsvd_template_B = BroadeningFunction(inv_flux_collection[~buffer_mask, 0],
                                          inv_flux_templateB[~buffer_mask], v_span, delta_v)
    BRsvd_template_A.smooth_sigma = ifitparamsA.bf_smooth_sigma
    BRsvd_template_B.smooth_sigma = ifitparamsB.bf_smooth_sigma

    if plot_ax_A is not None:
        plot_ax_A.clear()
        plot_ax_A.set_xlim([-v_span/2, +v_span/2])
        plot_ax_A.set_xlabel('Velocity shift [km/s]')
    if plot_ax_B is not None:
        plot_ax_B.clear()
        plot_ax_B.set_xlim([-v_span/2, +v_span/2])
        plot_ax_B.set_xlabel('Velocity shift [km/s]')
    if plot_ax_d1 is not None:
        plot_ax_d1.clear()
        plot_ax_d1.set_xlabel('Velocity shift [km/s]')
        plot_ax_d1.set_title('Component A, spectrum 16 and 19')
    if plot_ax_d2 is not None:
        plot_ax_d2.clear()
        plot_ax_d2.set_xlabel('Velocity shift [km/s]')
        plot_ax_d2.set_title('Component B, spectrum 16 and 19')

    fits_A = np.empty(shape=(n_spectra,), dtype=MinimizerResult)
    fits_B = np.empty(shape=(n_spectra,), dtype=MinimizerResult)

    for i in range(0, n_spectra):
        iterations = 0
        while True:
            iterations += 1
            RMS_RV_A = -RV_collection_A[i]
            corrected_flux_A = inv_flux_collection[:, i] -shift_spectrum(separated_flux_B, RV_collection_B[i], delta_v)
            corrected_flux_A = corrected_flux_A[~buffer_mask]
            ifitparamsA.RV = RV_collection_A[i]

            RV_collection_A[i], model_A = radial_velocity_single_component(corrected_flux_A, BRsvd_template_A,
                                                                           ifitparamsA)
            RMS_RV_A = np.abs(RMS_RV_A + RV_collection_A[i])
            if RMS_RV_A < convergence_limit:
                break
            elif iterations > iteration_limit:
                warnings.warn(f'RV: spectrum {i} did not reach convergence limit {convergence_limit} for component A.')
                break

        iterations = 0
        while True:
            iterations += 1
            RMS_RV_B = -RV_collection_B[i]
            corrected_flux_B = inv_flux_collection[:, i] -shift_spectrum(separated_flux_A, RV_collection_A[i], delta_v)
            corrected_flux_B = corrected_flux_B[~buffer_mask]
            ifitparamsB.RV = RV_collection_B[i]

            RV_collection_B[i], model_B = radial_velocity_single_component(corrected_flux_B, BRsvd_template_B,
                                                                           ifitparamsB)

            RMS_RV_B = np.abs(RMS_RV_B + RV_collection_B[i])
            if RMS_RV_B < convergence_limit:
                break
            elif iterations > iteration_limit:
                warnings.warn(f'RV: spectrum {i} did not reach convergence limit {convergence_limit} for component B.',
                              category=Warning)
                break

        fits_A[i], fits_B[i] = model_A[0], model_B[0]

        if plot_ax_A is not None and i < 20:
            update_bf_plot(plot_ax_A, model_A, i)
            if rv_lower_limit != 0.0:
                plot_ax_A.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_A.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
        if plot_ax_B is not None and i < 20:
            update_bf_plot(plot_ax_B, model_B, i)
            if rv_lower_limit != 0.0:
                plot_ax_B.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_B.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
        if plot_ax_d1 is not None and (i==19 or i==16):
            update_bf_plot(plot_ax_d1, model_A, i)
        if plot_ax_d2 is not None and (i==19 or i==16):
            update_bf_plot(plot_ax_d2, model_B, i)

    return RV_collection_A, RV_collection_B, (fits_A, fits_B)


def initialize_ssr_plots():
    # RVs and separated spectra
    fig_1 = plt.figure(figsize=(16, 9))
    gs_1 = fig_1.add_gridspec(2, 2)
    f1_ax1 = fig_1.add_subplot(gs_1[0, :])
    f1_ax2 = fig_1.add_subplot(gs_1[1, 0])
    f1_ax3 = fig_1.add_subplot(gs_1[1, 1])

    # Select BF fits
    fig_2 = plt.figure(figsize=(16, 9))
    gs_2 = fig_2.add_gridspec(1, 2)
    f2_ax1 = fig_2.add_subplot(gs_2[:, 0])
    f2_ax2 = fig_2.add_subplot(gs_2[:, 1])

    # Broadening function fits A
    fig_3 = plt.figure(figsize=(16, 9))
    gs_3 = fig_3.add_gridspec(1, 1)
    f3_ax1 = fig_3.add_subplot(gs_3[:, :])

    # Broadening function fits B
    fig_4 = plt.figure(figsize=(16, 9))
    gs_4 = fig_4.add_gridspec(1, 1)
    f4_ax1 = fig_4.add_subplot(gs_4[:, :])
    return f1_ax1, f1_ax2, f1_ax3, f2_ax1, f2_ax2, f3_ax1, f4_ax1


def plot_ssr_iteration(
        f1_ax1, f1_ax2, f1_ax3, separated_flux_A, separated_flux_B, wavelength, flux_template_A,
        flux_template_B, RV_A, RV_B, time, period, flux_collection, buffer_mask, rv_lower_limit, rv_proximity_limit
):
    f1_ax1.clear(); f1_ax2.clear(); f1_ax3.clear()
    separated_flux_A, separated_flux_B = separated_flux_A[~buffer_mask], separated_flux_B[~buffer_mask]
    wavelength = wavelength[~buffer_mask]
    flux_template_A, flux_template_B = flux_template_A[~buffer_mask], flux_template_B[~buffer_mask]
    flux_collection = flux_collection[~buffer_mask, :]

    if period is None:
        xval = time
        f1_ax1.set_xlabel('BJD - 245000')
    else:
        xval = np.mod(time, period) / period
        f1_ax1.set_xlabel('Phase')
    if rv_lower_limit == 0.0:
        RV_below_limit_mask = np.abs(RV_A - RV_B) < rv_proximity_limit
    else:
        RV_below_limit_mask = np.abs(RV_A) < rv_lower_limit
    f1_ax1.plot(xval[~RV_below_limit_mask], RV_A[~RV_below_limit_mask], 'b*')
    f1_ax1.plot(xval[~RV_below_limit_mask], RV_B[~RV_below_limit_mask], 'r*')
    f1_ax1.plot(xval[RV_below_limit_mask], RV_A[RV_below_limit_mask], 'bx')
    f1_ax1.plot(xval[RV_below_limit_mask], RV_B[RV_below_limit_mask], 'rx')

    f1_ax2.plot(wavelength, 1-separated_flux_A, 'b', linewidth=2)
    f1_ax2.plot(wavelength, 1-flux_template_A, '--', color='grey', linewidth=0.5)

    f1_ax3.plot(wavelength, 1-separated_flux_B, 'r', linewidth=2)
    f1_ax3.plot(wavelength, 1-flux_template_B, '--', color='grey', linewidth=0.5)

    f1_ax1.set_ylabel('Radial Velocity [km/s]')
    f1_ax2.set_ylabel('Normalized Flux')
    f1_ax2.set_xlabel('Wavelength [Å]')
    f1_ax3.set_xlabel('Wavelength [Å]')

    plt.draw_all()
    plt.pause(3)
    # plt.show(block=True)


def spectral_separation_routine(
        inv_flux_collection: np.ndarray, inv_flux_templateA: np.ndarray, inv_flux_templateB: np.ndarray, delta_v: float,
        ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters, wavelength: np.ndarray,
        time_values: np.ndarray, RV_guess_collection: np.ndarray, convergence_limit=1E-5, iteration_limit=10, plot=True,
        period=None, buffer_mask=None, rv_lower_limit=0.0, rv_proximity_limit=0.0, suppress_print=False,
        convergence_limit_scs=1E-7, estimate_error=False, return_unbuffered=True,
        use_spectra=True or np.ndarray
):
    """
    Routine that separates component spectra and calculates radial velocities by iteratively calling
    separate_component_spectra() and recalculate_RVs() to attempt to converge towards the correct RV values for the
    system. Requires good starting guesses on the component RVs. It also plots each iteration (if enabled) to follow
    along the results.

    The routine must be provided with normalized spectra sampled in the same wavelengths equi-spaced in velocity space.
    It is recommended to supply spectra that are buffered (padded) in the ends, and a buffer_mask that indicates this.
    This will limit the effect of loop-back when rolling (shifting) the spectra. spf.limit_wavelength_interval() can be
    used to produce a buffered data-set (and others of the same size).

    The spectra used for calculating the separated spectra should cover a wide range of orbital phases to provide a
    decent averaged spectrum. They should have a high S/N, and should not include significant emission lines. Avoid
    using spectra with RVs of the two components close to each others, or spectra from within eclipses.

    Additional features included to improve either the RV fitting process or the spectral separation:
      - Fitted v*sin(i) values are meaned and used as fit guesses for next iteration. Simultaneously, the parameter
        is bounded to the limits v*sin(i) * [1-0.5, 1+0.5], in order to ensure stable fitting for all spectra.

      - The routine can estimate a minimum precision error if it converges, if estimate_error is True. It does this by
        doing 10 more iterations after convergence is reached, and calculating the standard deviation of the resulting
        RVs. Note that this is likely much lower than the true error, since it only measures the deviations as a result
        of the routine repeatedly including different noise patterns in its estimate for the separated fluxes and
        fitting results. This should not in any way be used as actual errors when lower than other estimates.

      - The routine can be provided with an array of indices specifying which spectra to use when creating the separated
        spectra. This is the recommended way to designate. A lower RV limit on the primary component, or a proximity
        limit of the two RVs, can also be provided isntead. However, this does not consider any other important
        factors for why you would leave out a spectrum (low S/N, bad reduction, within eclipses).
        Providing rv_lower_limit (or rv_proximity_limit) while also supplying use_spectra, can be useful since the
        routines avoids using the "bad" component B RVs when calculating RMS values.
        Note that if use_spectra is provided as an array, the two rv limit options will only be used for RMS
        calculation, and not for spectral separation.

    :param inv_flux_collection:   np.ndarray shape (:, n_spectra). Collection of inverted fluxes for each spectrum.
    :param inv_flux_templateA:    np.ndarray shape (:, ). Inverted flux of template spectrum for component A.
    :param inv_flux_templateB:    np.ndarray shape (:, ). Inverted flux of template spectrum for component B.
    :param delta_v:               float. The sampling delta_v of the velocity grid (resolution). Example 1.0 km/s
    :param ifitparamsA:           InitialFitParameters. Fitting and broadening function parameters for component A.
    :param ifitparamsB:           InitialFitParameters. Fitting and broadening function parameters for component B.
    :param wavelength:            np.ndarray shape (:, ). Wavelength values for the spectra (both program and template).
    :param time_values:           np.ndarray shape (n_spectra, ). Time values of the program spectra (for plotting).
    :param RV_guess_collection:   np.ndarray shape (n_spectra, 2). Initial RV guesses for each component (A: [:, 0]).
    :param convergence_limit:     float. Convergence limit for the routine. If both components' RV changed by less than
                                    this value from last iteration, the routine returns.
    :param iteration_limit:       int. Maximum number of allowed iterations before the routine returns.
    :param plot:                  bool. Indicates whether plots should be produced and updated during iterations.
    :param period:                float. A period of the binary system for phase-plotting.
    :param buffer_mask:           np.ndarray of bools, shape (:, ). If supplied, the routine treats the spectra as
                                    buffered (padded) in the ends, and will cut them out by masking the data-set with
                                    this array before calculating RV values.
    :param rv_lower_limit:       float. A lower RV limit on component A for the spectral separation. Overrides
                                    rv_proximity_limit. This limit will only consider component A when deciding if
                                    the data-set is reasonable enough. Useful for when RV for component B is unstable.
                                    Requires that spectra has been decently corrected for systemic RV beforehand.
    :param rv_proximity_limit:   float. The RV limit for the spectral separation. If RV_A-RV_B is below,
                                    the individual spectrum will not be used for calculating the separated spectra, as
                                    it will be treated as mixed.
    :param suppress_print:        string. Indicates whether console printing should be suppressed. If 'scs', prints from
                                    separate_component_spectra() will not be shown. If 'ssr', prints from this routine
                                    will not be shown (but separate_component_spectra() will). If 'all', not prints are
                                    allowed.
    :param convergence_limit_scs: float. Convergence limit to pass along to the function separate_component_spectra().
    :param estimate_error:        bool. Indicates if the minimum precision error of the routine should be estimated in
                                    case the routine reaches the convergence limit.
    :param return_unbuffered:     bool. Indicates if returned results should be unbuffered (un-padded) or not. Default
                                    is True, meaning the shorter arrays are returned.
    :param use_spectra:           bool, np.ndarray. If True, use all spectra for calculating the "separated spectra".
                                    If np.ndarray or list, this should indicate the indices of the spectra to
                                    use from 0 to n_spectra-1.

    :return:    RV_collection_A,  RV_collection_B, separated_flux_A, separated_flux_B, wavelength, iteration_errors
                RV_collection_A:  np.ndarray shape (n_spectra, ). RV values of component A for each program spectrum.
                RV_collection_B:  same, but for component B (includes values below rv_lower_limit).
                separated_flux_A: np.ndarray shape (:*, ). The found "separated" or "disentangled" spectrum for A.
                                    It is an inverted flux (1-normalized_flux).
                separated_flux_B: np.ndarray shape (:*, ). The found "separated" or "disentangled" spectrum for B.
                wavelength:       np.ndarray shape (:*, ). Wavelength values for the separated spectra.
                iteration_errors: (error_RVs_A, error_RVs_B)
                    error_RVs_A:  np.ndarray shape (n_spectra, :). Minimum error estimate of the RVs for A obtained by
                                    repeating continuing iteration after convergence limit is reached.
                    error_RVs_B:  np.ndarray shape (n_spectra, :). Minimum error estimate of the RVs for B.

            Note on :*
                if buffer_mask is provided, the returned spectra will be the un-buffered versions, meaning
                separated_flux_A.size = inv_flux_templateA[buffer_mask].size. Same for the returned wavelength.
                This can be disabled by setting return_unbuffered=False.
    """

    suppress_scs = False; suppress_ssr = False
    if suppress_print == 'scs': suppress_scs = True
    elif suppress_print == 'ssr': suppress_ssr = True
    elif suppress_print == 'all': suppress_scs = True; suppress_ssr = True
    skip_convergence_check = False

    RV_collection_A, RV_collection_B = RV_guess_collection[:, 0], RV_guess_collection[:, 1]

    error_RVs_A = np.zeros((10, RV_collection_A.size))
    error_RVs_B = np.zeros((10, RV_collection_A.size))

    if buffer_mask is None:
        buffer_mask = np.zeros(wavelength.shape, dtype=bool)

    # Initialize plot figures
    if plot:
        f1_ax1, f1_ax2, f1_ax3, f2_ax1, f2_ax2, f3_ax1, f4_ax1 = initialize_ssr_plots()
    else:
        f2_ax1=None; f2_ax2=None; f3_ax1 = None; f4_ax1 = None

    # Iterative loop that repeatedly separates the spectra from each other in order to calculate new RVs (Gonzales 2005)
    iterations = 0
    print('Spectral Separation: ')
    while True:
        print(f'\nIteration {iterations}.')
        if rv_lower_limit == 0.0:
            RV_mask = np.abs(RV_collection_A-RV_collection_B) > rv_proximity_limit
        else:
            RV_mask = np.abs(RV_collection_A) > rv_lower_limit
        RMS_A, RMS_B = -RV_collection_A, -RV_collection_B[RV_mask]

        separated_flux_A, separated_flux_B = separate_component_spectra(
            inv_flux_collection, RV_collection_A, RV_collection_B, delta_v, convergence_limit_scs, suppress_scs,
            rv_proximity_limit=rv_proximity_limit, rv_lower_limit=rv_lower_limit, use_spectra=use_spectra)

        RV_collection_A, RV_collection_B, (fits_A, fits_B) \
            = recalculate_RVs(inv_flux_collection, separated_flux_A, separated_flux_B, RV_collection_A,
                              RV_collection_B, inv_flux_templateA, inv_flux_templateB, delta_v, ifitparamsA,
                              ifitparamsB, buffer_mask, plot_ax_A=f3_ax1, plot_ax_B=f4_ax1, plot_ax_d1=f2_ax1,
                              plot_ax_d2=f2_ax2, rv_lower_limit=rv_lower_limit)

        if plot:
            plot_ssr_iteration(f1_ax1, f1_ax2, f1_ax3, separated_flux_A, separated_flux_B,
                               wavelength, inv_flux_templateA, inv_flux_templateB, RV_collection_A,
                               RV_collection_B, time_values, period, inv_flux_collection, buffer_mask, rv_lower_limit,
                               rv_proximity_limit)

        # Average vsini values for future fit guess and limit allowed fit area
        vsini_A, vsini_B = np.empty(shape=fits_A.shape), np.empty(shape=fits_B.shape)
        for i in range(0, vsini_A.size):
            _, _, vsini_A[i], _, _, _ = get_fit_parameter_values(fits_A[i].params)
            _, _, vsini_B[i], _, _, _ = get_fit_parameter_values(fits_B[i].params)
        ifitparamsA.vsini = np.mean(vsini_A)
        ifitparamsB.vsini = np.mean(vsini_B)
        ifitparamsA.vsini_vary_limit = 0.5
        ifitparamsB.vsini_vary_limit = 0.5

        iterations += 1
        RMS_A += RV_collection_A
        RMS_B += RV_collection_B[RV_mask]
        RMS_A = np.sqrt(np.sum(RMS_A**2)/RV_collection_A.size)
        RMS_B = np.sqrt(np.sum(RMS_B**2)/RV_collection_B[RV_mask].size)
        if not suppress_ssr:
            print(f'RV_A RMS: {RMS_A}. ' +
                  f'RV_B RMS: {RMS_B}.')
        if RMS_A < convergence_limit and RMS_B < convergence_limit and skip_convergence_check is False:
            print(f'Spectral separation routine terminates after reaching convergence limit {convergence_limit}.')
            if estimate_error is True:
                print('\nBeginning iteration error estimation.')
                skip_convergence_check = True
                iterations = 0
                iteration_limit = 9
            else:
                break
        if skip_convergence_check is True:
            error_RVs_A[iterations, :] = RV_collection_A
            error_RVs_B[iterations, :] = RV_collection_B
        if iterations >= iteration_limit:
            if skip_convergence_check is False:
                warnings.warn(f'RV convergence limit of {convergence_limit} not reached in {iterations} iterations.',
                              category=Warning)
            print('Spectral separation routine terminates.')
            break

    if skip_convergence_check is True:  # Meaning successful convergence and completed error estimate
        error_RVs_A = np.std(error_RVs_A, axis=0)
        error_RVs_B = np.std(error_RVs_B, axis=0)
    iteration_errors = (error_RVs_A, error_RVs_B)
    ifitparams = (ifitparamsA, ifitparamsB)

    if return_unbuffered:
        return RV_collection_A, RV_collection_B, separated_flux_A[buffer_mask], separated_flux_B[buffer_mask], \
               wavelength[buffer_mask], iteration_errors, ifitparams
    else:
        return RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B, wavelength, iteration_errors, \
               ifitparams


def estimate_errors(
        wavelength_interval_size: float, inv_flux_collection: np.ndarray, inv_flux_templateA: np.ndarray,
        inv_flux_templateB: np.ndarray, separated_flux_A: np.ndarray, separated_flux_B: np.ndarray, delta_v: float,
        ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters, wavelength: np.ndarray,
        RV_collection_A: np.ndarray, RV_collection_B: np.ndarray, times: np.ndarray, wavelength_buffer_size=100.0,
        plot=False, period=1.0
):
    wavelength_interval_collection = []
    flux_interval_collection = []
    templateA_interval_collection = []
    templateB_interval_collection = []
    sep_flux_A_interval_collection = []
    sep_flux_B_interval_collection = []
    interval_buffer_mask = []
    w_interval_start = wavelength[0] + wavelength_buffer_size

    # # Splits the spectra into intervals of length wavelength_interval_size # #
    while True:
        if w_interval_start + wavelength_interval_size > wavelength[-1] - wavelength_buffer_size:
            w_interval_end = wavelength[-1] - wavelength_buffer_size
        else:
            w_interval_end = w_interval_start + wavelength_interval_size

        if w_interval_end - w_interval_start < wavelength_interval_size // 2:
            break

        wavelength_interval = (w_interval_start, w_interval_end)

        _, (wavelength_buffered, flux_buffered, buffer_mask, buffer_mask_internal) = \
            spf.limit_wavelength_interval(wavelength_interval, wavelength, inv_flux_collection,
                                          buffer_size=wavelength_buffer_size, even_length=True)

        _, (_, templateA_buffered, _, _) = \
            spf.limit_wavelength_interval(wavelength_interval, wavelength, inv_flux_templateA,
                                          buffer_mask=buffer_mask_internal, even_length=True)
        _, (_, templateB_buffered, _, _) = \
            spf.limit_wavelength_interval(wavelength_interval, wavelength, inv_flux_templateB,
                                          buffer_mask=buffer_mask_internal, even_length=True)
        _, (_, sep_flux_A_buffered, _, _) = \
            spf.limit_wavelength_interval(wavelength_interval, wavelength, separated_flux_A,
                                          buffer_mask=buffer_mask_internal, even_length=True)
        _, (_, sep_flux_B_buffered, _, _) = \
            spf.limit_wavelength_interval(wavelength_interval, wavelength, separated_flux_B,
                                          buffer_mask=buffer_mask_internal, even_length=True)

        wavelength_interval_collection.append(wavelength_buffered)
        flux_interval_collection.append(flux_buffered)
        templateA_interval_collection.append(templateA_buffered)
        templateB_interval_collection.append(templateB_buffered)
        sep_flux_A_interval_collection.append(sep_flux_A_buffered)
        sep_flux_B_interval_collection.append(sep_flux_B_buffered)
        interval_buffer_mask.append(buffer_mask)

        w_interval_start = w_interval_end

    RV_estimates_A = np.empty((RV_collection_A.size, len(wavelength_interval_collection)))
    RV_estimates_B = np.empty((RV_collection_B.size, len(wavelength_interval_collection)))

    # # Calculate RVs using limited part of spectra # #
    for i in range(0, len(wavelength_interval_collection)):
        if plot:
            # RVs and separated spectra
            fig_1 = plt.figure(figsize=(16, 9))
            gs_1 = fig_1.add_gridspec(2, 2)
            f1_ax1 = fig_1.add_subplot(gs_1[0, :])
            f1_ax2 = fig_1.add_subplot(gs_1[1, 0])
            f1_ax3 = fig_1.add_subplot(gs_1[1, 1])

            # Broadening function fits A
            fig_2 = plt.figure(figsize=(16, 9))
            gs_2 = fig_2.add_gridspec(1, 1)
            f2_ax1 = fig_2.add_subplot(gs_2[:, :])

            # Broadening function fits B
            fig_3 = plt.figure(figsize=(16, 9))
            gs_3 = fig_3.add_gridspec(1, 1)
            f3_ax1 = fig_3.add_subplot(gs_3[:, :])
        else:
            f2_ax1 = None; f3_ax1 = None

        RV_estimates_A[:, i], RV_estimates_B[:, i], _ = recalculate_RVs(
            flux_interval_collection[i], sep_flux_A_interval_collection[i], sep_flux_B_interval_collection[i],
            RV_collection_A, RV_collection_B, templateA_interval_collection[i], templateB_interval_collection[i],
            delta_v, ifitparamsA, ifitparamsB, interval_buffer_mask[i], plot_ax_A=f2_ax1, plot_ax_B=f3_ax1
        )

        if plot:
            phase = np.mod(times, period) / period
            f1_ax1.plot(phase, RV_estimates_A[:, i], 'b*')
            f1_ax1.plot(phase, RV_estimates_B[:, i], 'r*')
            f1_ax2.plot(wavelength_interval_collection[i], 1-sep_flux_A_interval_collection[i], 'b', linewidth=2)
            f1_ax2.plot(wavelength_interval_collection[i], 1-templateA_interval_collection[i], '--', color='grey',
                        linewidth=0.5)
            f1_ax3.plot(wavelength_interval_collection[i], 1 - sep_flux_B_interval_collection[i], 'r', linewidth=2)
            f1_ax3.plot(wavelength_interval_collection[i], 1 - templateB_interval_collection[i], '--', color='grey',
                        linewidth=0.5)
            f1_ax1.set_ylabel('Radial Velocity [km/s]')
            f1_ax2.set_ylabel('Normalized Flux')
            f1_ax2.set_xlabel('Wavelength [Å]')
            f1_ax3.set_xlabel('Wavelength [Å]')
            plt.show(block=True)

    errors_RV_A = np.std(RV_estimates_A, axis=1)
    errors_RV_B = np.std(RV_estimates_B, axis=1)
    return (errors_RV_A, errors_RV_B), (RV_estimates_A, RV_estimates_B)


def estimate_errors_2(
        wavelength_interval_size: int, inv_flux_collection: np.ndarray,
        inv_flux_templateA: np.ndarray, inv_flux_templateB: np.ndarray, delta_v: float,
        ifitparamsA: InitialFitParameters, ifitparamsB: InitialFitParameters, wavelength: np.ndarray,
        time_values: np.ndarray, RV_collection: np.ndarray, convergence_limit=1E-5,
        iteration_limit=10, plot=True, period=None, wavelength_buffer_size=100, rv_lower_limit=0.0,
        suppress_print=False, convergence_limit_scs=1E-7, use_spectra=True
):

    wavelength_interval_collection = []
    flux_interval_collection = []
    templateA_interval_collection = []
    templateB_interval_collection = []
    interval_buffer_mask = []
    w_interval_start = wavelength[0] + wavelength_buffer_size
    while True:
        if w_interval_start + wavelength_interval_size > wavelength[-1] - wavelength_buffer_size:
            w_interval_end = wavelength[-1] - wavelength_buffer_size
        else:
            w_interval_end = w_interval_start + wavelength_interval_size

        if w_interval_end - w_interval_start < wavelength_interval_size // 2:
            break

        w_interval = (w_interval_start, w_interval_end)

        _, (wl_buffered, fl_buffered, buffer_mask, buffer_mask_internal) = \
            spf.limit_wavelength_interval(w_interval, wavelength, inv_flux_collection,
                                          buffer_size=wavelength_buffer_size, even_length=True)
        _, (_, flA_buffered, _, _) = spf.limit_wavelength_interval(w_interval, wavelength, inv_flux_templateA,
                                                                   buffer_mask=buffer_mask_internal, even_length=True)
        _, (_, flB_buffered, _, _) = spf.limit_wavelength_interval(w_interval, wavelength, inv_flux_templateB,
                                                                   buffer_mask=buffer_mask_internal, even_length=True)

        wavelength_interval_collection.append(wl_buffered)
        flux_interval_collection.append(fl_buffered)
        templateA_interval_collection.append(flA_buffered)
        templateB_interval_collection.append(flB_buffered)
        interval_buffer_mask.append(buffer_mask)

        w_interval_start = w_interval_end

    RV_A_interval_values = np.empty((RV_collection[:, 0].size, len(wavelength_interval_collection)))
    RV_B_interval_values = np.empty((RV_collection[:, 0].size, len(wavelength_interval_collection)))
    for i in range(0, len(wavelength_interval_collection)):
        current_wl, current_fl = wavelength_interval_collection[i], flux_interval_collection[i]
        current_fltA, current_fltB = templateA_interval_collection[i], templateB_interval_collection[i]
        current_buffer = interval_buffer_mask[i]

        RV_A_temp, RV_B_temp, _, _, _, _, _ = spectral_separation_routine(
            current_fl, current_fltA, current_fltB, delta_v, ifitparamsA, ifitparamsB, current_wl, time_values,
            RV_collection, convergence_limit, iteration_limit, plot, period, buffer_mask=current_buffer,
            rv_lower_limit=rv_lower_limit, suppress_print=suppress_print, convergence_limit_scs=convergence_limit_scs,
            use_spectra=use_spectra
        )
        if plot:
            plt.close('all')
        RV_A_interval_values[:, i] = RV_A_temp
        RV_B_interval_values[:, i] = RV_B_temp

    # bad_RVB_value_mask = np.abs(RV_A_interval_values) < rv_lower_limit
    RV_errors_A = np.std(RV_A_interval_values, axis=1)
    RV_errors_B = np.std(RV_B_interval_values, axis=1)
    return RV_errors_A, RV_errors_B
