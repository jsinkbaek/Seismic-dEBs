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
"""

from RV.library.calculate_radial_velocities import radial_velocity_single_component
from RV.library.broadening_function_svd import *
from RV.library.initial_fit_parameters import InitialFitParameters
import RV.library.spectrum_processing_functions as spf
from copy import deepcopy
import matplotlib
import RV.library.broadening_function_svd as bfsvd
from scipy.interpolate import interp1d
import scipy.constants as scc
from matplotlib.backends.backend_pdf import PdfPages


def shift_spectrum(flux, radial_velocity_shift, delta_v):
    """
    Performs a wavelength shift of a spectrum. Two-step process:
     - First, it performs a low resolution roll of the flux elements. Resolution here is defined by delta_v (which must
       be the velocity spacing of the flux array). Example precision: 1 km/s (with delta_v=1.0)
     - Second, it performs linear interpolation to produce a small shift to correct for the low precision.
     The spectrum must be evenly spaced in velocity space in order for this function to properly work.
    :param flux:                    np.ndarray size (:, ), flux values of the spectrum.
    :param radial_velocity_shift:   float, the shift in velocity units (km/s)
    :param delta_v:                 float, the resolution of the spectrum in km/s.
    :return flux_shifted:           np.ndarray size (:, ), shifted flux values of the spectrum.

    shift = radial_velocity_shift / delta_v
    indices = np.linspace(0, flux.size, flux.size)
    indices_shifted = indices - shift
    flux_shifted = np.interp(indices_shifted, indices, flux)
    """
    # Perform large delta_v precision shift
    large_shift = np.floor(radial_velocity_shift / delta_v)
    flux_ = np.roll(flux, int(large_shift))

    # Perform small < delta_v precision shift using linear interpolation
    small_shift = radial_velocity_shift / delta_v - large_shift
    indices = np.linspace(0, flux.size, flux.size)
    indices_shifted = indices - small_shift  # minus will give same shift direction as np.roll()
    # flux_shifted = interp1d(indices, flux_, fill_value='extrapolate')(indices_shifted)
    flux_shifted = np.interp(indices_shifted, indices, flux_)

    return flux_shifted


def shift_wavelength_spectrum(wavelength, flux, radial_velocity_shift):
    wavelength_shifted = wavelength * (1 + radial_velocity_shift/(scc.speed_of_light / 1000))
    # flux_shifted = interp1d(wavelength_shifted, flux, fill_value='extrapolate')(wavelength)
    flux_shifted = np.interp(wavelength, wavelength_shifted, flux)
    return flux_shifted


def separate_component_spectra(
        flux_collection, radial_velocity_collection_A, radial_velocity_collection_B, delta_v, convergence_limit,
        ifitparamsA: InitialFitParameters, ifitparamsB: InitialFitParameters, suppress_scs=False, max_iterations=20,
        rv_proximity_limit=0.0, rv_lower_limit=0.0, ignore_component_B=False
):
    """
    Assumes that component A is the dominant component in the spectrum. Attempts to separate the two components using
    RV shifts and averaged spectra.

    :param flux_collection:               np.ndarray shape (datasize, nspectra) of all the observed spectra
    :param radial_velocity_collection_A:  np.ndarray shape (nspectra, ) of radial velocity values for component A
    :param radial_velocity_collection_B:  np.ndarray shape (nspectra, ) of radial velocity values for component B
    :param delta_v:                       float, the sampling size of the spectra in velocity space
    :param convergence_limit:             float, the precision needed to break while loop
    :param ifitparamsA:                   InitialFitParameters. Only used for variable "use_for_spectral_separation".
                                          Indicates which spectra should be used for calculating the separated spectrum
                                          for component A.
    :param ifitparamsB:                   Same as ifitparamsA, but for component B instead.
    :param suppress_scs:                  bool, indicates if printing should be suppressed
    :param max_iterations:                int, maximum number of allowed iterations before breaking loop
    :param rv_proximity_limit:            float, RV limit in order to add separated spectrum (if closer, the
                                          components are expected to be mixed, and are not included to avoid pollution)
    :param rv_lower_limit:                float. RV limit using only component A. Overrides rv_proximity_limit if also
                                          provided. Useful if RV for B is unstable.
    :param ignore_component_B:            bool. If True, separate_component_spectra will assume that component B is
                                          non-existing in the spectra for calculating component A.

    :return separated_flux_A, separated_flux_B:   the separated and meaned total component spectra of A and B.
    """
    n_spectra = flux_collection[0, :].size
    separated_flux_B = np.zeros((flux_collection[:, 0].size,))  # Set to 0 before iteration
    separated_flux_A = np.zeros((flux_collection[:, 0].size,))

    use_spectra_A = ifitparamsA.use_for_spectral_separation
    use_spectra_B = ifitparamsB.use_for_spectral_separation

    iteration_counter = 0
    while True:
        RMS_values_A = -separated_flux_A
        RMS_values_B = -separated_flux_B
        iteration_counter += 1
        separated_flux_A = np.zeros((flux_collection[:, 0].size,))
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (use_spectra_A is None) or (i in use_spectra_A):
                rvA = radial_velocity_collection_A[i]
                rvB = radial_velocity_collection_B[i]
                if rv_lower_limit == 0.0 and use_spectra_A is None:
                    condition = np.abs(rvA-rvB) > rv_proximity_limit
                elif use_spectra_A is None:
                    condition = np.abs(rvA) > rv_lower_limit
                else:
                    condition = True

                if condition:
                    shifted_flux_A = shift_spectrum(flux_collection[:, i], -rvA, delta_v)
                    if ignore_component_B is False:
                        separated_flux_A += shifted_flux_A - shift_spectrum(separated_flux_B, rvB - rvA, delta_v)
                    else:
                        separated_flux_A += shifted_flux_A
                    n_used_spectra += 1
            elif use_spectra_A.size != 0:
                pass
            else:
                raise TypeError(f'use_spectra_A is either of wrong type ({type(use_spectra_A)}), empty, or wrong value.\n' +
                                f'Expected type: {type(True)} or np.ndarray. Expected value if bool: True')
        separated_flux_A = separated_flux_A / n_used_spectra

        separated_flux_B = np.zeros((flux_collection[:, 0].size,))
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (use_spectra_B is None) or (i in use_spectra_B):
                rvA = radial_velocity_collection_A[i]
                rvB = radial_velocity_collection_B[i]
                if rv_lower_limit == 0.0 and use_spectra_B is None:
                    condition = np.abs(rvA-rvB) > rv_proximity_limit
                elif use_spectra_B is None:
                    condition = np.abs(rvA) > rv_lower_limit
                else:
                    condition = True

                if condition:
                    shifted_flux_B = shift_spectrum(flux_collection[:, i], -rvB, delta_v)
                    separated_flux_B += shifted_flux_B - shift_spectrum(separated_flux_A, rvA - rvB, delta_v)
                    n_used_spectra += 1
            elif use_spectra_B.size != 0:
                pass
            else:
                raise TypeError(f'use_spectra_B is either of wrong type ({type(use_spectra_B)}), empty, or wrong value.\n' +
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


def _update_bf_plot(plot_ax, model, index):
    fit = model[0]
    model_values = model[1]
    velocity_values = model[2]
    bf_smooth_values = model[4]
    _, RV, _, _, _, _ = get_fit_parameter_values(fit.params)
    plot_ax.plot(velocity_values, 1+0.02*bf_smooth_values/np.max(bf_smooth_values)-0.05*index)
    plot_ax.plot(velocity_values, 1+0.02*model_values/np.max(bf_smooth_values)-0.05*index, 'k--')
    plot_ax.plot(np.ones(shape=(2,))*RV,
                 [1-0.05*index-0.005, 1+0.025*np.max(model_values)/np.max(bf_smooth_values)-0.05*index],
                 color='grey')


def recalculate_RVs(
        inv_flux_collection: np.ndarray, separated_flux_A: np.ndarray, separated_flux_B: np.ndarray,
        RV_collection_A: np.ndarray, RV_collection_B: np.ndarray, inv_flux_templateA: np.ndarray,
        inv_flux_templateB: np.ndarray, delta_v: float, ifitparamsA: InitialFitParameters,
        ifitparamsB: InitialFitParameters, buffer_mask: np.ndarray, iteration_limit=10, convergence_limit=1e-5,
        plot_ax_A=None, plot_ax_B=None, plot_ax_d1=None, plot_ax_d2=None, rv_lower_limit=0.0, period=None,
        time_values=None
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

    bf_fitres_A = np.empty(shape=(n_spectra,), dtype=tuple)
    bf_fitres_B = np.empty(shape=(n_spectra,), dtype=tuple)

    for i in range(0, n_spectra):
        iterations = 0
        while True:
            iterations += 1
            RMS_RV_A = -RV_collection_A[i]

            corrected_flux_A = inv_flux_collection[:, i] - shift_spectrum(separated_flux_B, RV_collection_B[i], delta_v)

            if period is not None and ifitparamsB.ignore_at_phase is not None and time_values is not None:
                if _check_for_total_eclipse(time_values[i], period, ifitparamsB.ignore_at_phase) is True:
                    corrected_flux_A = inv_flux_collection[:, i]

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

            corrected_flux_B = inv_flux_collection[:, i] - shift_spectrum(separated_flux_A, RV_collection_A[i], delta_v)

            if period is not None and ifitparamsA.ignore_at_phase is not None and time_values is not None:
                if _check_for_total_eclipse(time_values[i], period, ifitparamsA.ignore_at_phase) is True:
                    corrected_flux_B = inv_flux_collection[:, i]

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

        bf_fitres_A[i], bf_fitres_B[i] = model_A, model_B

        if plot_ax_A is not None and i < 20:
            _update_bf_plot(plot_ax_A, model_A, i)
            if rv_lower_limit != 0.0:
                plot_ax_A.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_A.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
        if plot_ax_B is not None and i < 20:
            _update_bf_plot(plot_ax_B, model_B, i)
            if rv_lower_limit != 0.0:
                plot_ax_B.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_B.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
        if plot_ax_d1 is not None and (i==19 or i==16):
            _update_bf_plot(plot_ax_d1, model_A, i)
        if plot_ax_d2 is not None and (i==19 or i==16):
            _update_bf_plot(plot_ax_d2, model_B, i)

    return RV_collection_A, RV_collection_B, (bf_fitres_A, bf_fitres_B)


def _check_for_total_eclipse(time_value, period, eclipse_phase_area):
    phase = np.mod(time_value, period)/period
    lower = eclipse_phase_area[0]
    upper = eclipse_phase_area[1]
    if lower < upper:
        condition = (phase > lower) & (phase < upper)
    elif lower > upper:
        condition = (phase > lower) | (phase < upper)
    else:
        raise ValueError('eclipse_phase_area must comprise of a lower and an upper value that are separate.')
    return condition


def _initialize_ssr_plots():
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


def _plot_ssr_iteration(
        f1_ax1, f1_ax2, f1_ax3, separated_flux_A, separated_flux_B, wavelength, flux_template_A,
        flux_template_B, RV_A, RV_B, time, period, buffer_mask, rv_lower_limit, rv_proximity_limit
):
    f1_ax1.clear(); f1_ax2.clear(); f1_ax3.clear()
    separated_flux_A, separated_flux_B = separated_flux_A[~buffer_mask], separated_flux_B[~buffer_mask]
    wavelength = wavelength[~buffer_mask]
    flux_template_A, flux_template_B = flux_template_A[~buffer_mask], flux_template_B[~buffer_mask]

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


def save_multi_image(filename):
    """
    https://www.tutorialspoint.com/saving-all-the-open-matplotlib-figures-in-one-file-at-once
    """
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def spectral_separation_routine(
        inv_flux_collection: np.ndarray, inv_flux_templateA: np.ndarray, inv_flux_templateB: np.ndarray, delta_v: float,
        ifitparamsA: InitialFitParameters, ifitparamsB: InitialFitParameters, wavelength: np.ndarray,
        time_values: np.ndarray, RV_guess_collection: np.ndarray, convergence_limit=1E-5, iteration_limit=10, plot=True,
        period=None, buffer_mask=None, rv_lower_limit=0.0, rv_proximity_limit=0.0, suppress_print=False,
        convergence_limit_scs=1E-7, return_unbuffered=True, save_plot_path=None,
        ignore_component_B=False, save_extras=True
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
    :param return_unbuffered:     bool. Indicates if returned results should be unbuffered (un-padded) or not. Default
                                    is True, meaning the shorter arrays are returned.
    :param save_plot_path:        str, path to save iteration plots to. If None, plots are not saved
    :param ignore_component_B:    bool. If True, separate_component_spectra will assume that component B is
                                    non-existing in the spectra for calculating component A.

    :return:    RV_collection_A,  RV_collection_B, separated_flux_A, separated_flux_B, wavelength
                RV_collection_A:  np.ndarray shape (n_spectra, ). RV values of component A for each program spectrum.
                RV_collection_B:  same, but for component B (includes values below rv_lower_limit).
                separated_flux_A: np.ndarray shape (:*, ). The found "separated" or "disentangled" spectrum for A.
                                    It is an inverted flux (1-normalized_flux).
                separated_flux_B: np.ndarray shape (:*, ). The found "separated" or "disentangled" spectrum for B.
                wavelength:       np.ndarray shape (:*, ). Wavelength values for the separated spectra.

            Note on :*
                if buffer_mask is provided, the returned spectra will be the un-buffered versions, meaning
                separated_flux_A.size = inv_flux_templateA[buffer_mask].size. Same for the returned wavelength.
                This can be disabled by setting return_unbuffered=False.
    """
    suppress_scs = False; suppress_ssr = False
    if suppress_print == 'scs': suppress_scs = True
    elif suppress_print == 'ssr': suppress_ssr = True
    elif suppress_print == 'all': suppress_scs = True; suppress_ssr = True

    RV_collection_A, RV_collection_B = deepcopy(RV_guess_collection[:, 0]), deepcopy(RV_guess_collection[:, 1])
    ifitparamsA, ifitparamsB = deepcopy(ifitparamsA), deepcopy(ifitparamsB)

    if buffer_mask is None:
        buffer_mask = np.zeros(wavelength.shape, dtype=bool)

    # Initialize plot figures
    if plot:
        f1_ax1, f1_ax2, f1_ax3, f2_ax1, f2_ax2, f3_ax1, f4_ax1 = _initialize_ssr_plots()
    else:
        f2_ax1 = None; f2_ax2=None; f3_ax1 = None; f4_ax1 = None

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
            inv_flux_collection, RV_collection_A, RV_collection_B, delta_v, convergence_limit_scs,
            ifitparamsA, ifitparamsB, suppress_scs, rv_proximity_limit=rv_proximity_limit,
            rv_lower_limit=rv_lower_limit, ignore_component_B=ignore_component_B
        )

        RV_collection_A, RV_collection_B, (bf_fitres_A, bf_fitres_B) = recalculate_RVs(
            inv_flux_collection, separated_flux_A, separated_flux_B, RV_collection_A, RV_collection_B,
            inv_flux_templateA, inv_flux_templateB, delta_v, ifitparamsA, ifitparamsB, buffer_mask, plot_ax_A=f3_ax1,
            plot_ax_B=f4_ax1, plot_ax_d1=f2_ax1, plot_ax_d2=f2_ax2, rv_lower_limit=rv_lower_limit, period=period,
            time_values=time_values
        )

        if plot:
            _plot_ssr_iteration(
                f1_ax1, f1_ax2, f1_ax3, separated_flux_A, separated_flux_B, wavelength, inv_flux_templateA,
                inv_flux_templateB, RV_collection_A, RV_collection_B, time_values, period,
                buffer_mask, rv_lower_limit, rv_proximity_limit
            )

        # Average vsini values for future fit guess and limit allowed fit area
        vsini_A, vsini_B = np.empty(shape=bf_fitres_A.shape), np.empty(shape=bf_fitres_B.shape)
        for i in range(0, vsini_A.size):
            _, _, vsini_A[i], _, _, _ = get_fit_parameter_values(bf_fitres_A[i][0].params)
            _, _, vsini_B[i], _, _, _ = get_fit_parameter_values(bf_fitres_B[i][0].params)
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
        if RMS_A < convergence_limit and RMS_B < convergence_limit:
            print(f'Spectral separation routine terminates after reaching convergence limit {convergence_limit}.')
            break
        if iterations >= iteration_limit:
            warnings.warn(f'RV convergence limit of {convergence_limit} not reached in {iterations} iterations.',
                          category=Warning)
            print('Spectral separation routine terminates.')
            break

    ifitparams = (ifitparamsA, ifitparamsB)

    RVb_flags = np.zeros(RV_collection_B.shape)
    RVb_flags[RV_mask] = 1.0

    if save_extras is True:
        save_separation_data(
            'Data/additionals/separation_routine/', wavelength[~buffer_mask], time_values, RV_collection_A,
            RV_collection_B, RV_guess_collection, separated_flux_A[~buffer_mask], separated_flux_B[~buffer_mask],
            bf_fitres_A, bf_fitres_B, RVb_flags, inv_flux_templateA, inv_flux_templateB
        )

    if save_plot_path is not None:
        save_multi_image(save_plot_path)

    if return_unbuffered:
        return RV_collection_A, RV_collection_B, separated_flux_A[~buffer_mask], separated_flux_B[~buffer_mask], \
               wavelength[~buffer_mask], ifitparams, RVb_flags
    else:
        return RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B, wavelength, ifitparams, RVb_flags


def save_separation_data(
        location, wavelength, time_values, RVs_A, RVs_B, RVs_initial, separated_flux_A, separated_flux_B, bf_fitres_A,
        bf_fitres_B, RVb_flags, template_flux_A, template_flux_B
):
    filename_bulk = str(int(np.min(wavelength))) + '_' + str(int(np.max(wavelength)))
    
    rvA_array = np.empty((RVs_A.size, 2))
    rvA_array[:, 0], rvA_array[:, 1] = time_values, RVs_A
    
    rvB_array = np.empty((RVs_B.size, 3))
    rvB_array[:, 0], rvB_array[:, 1], rvB_array[:, 2] = time_values, RVs_B, RVb_flags
    
    sep_array = np.empty((wavelength.size, 5))
    sep_array[:, 0], sep_array[:, 1], sep_array[:, 2] = wavelength, separated_flux_A, separated_flux_B
    sep_array[:, 3], sep_array[:, 4] = template_flux_A, template_flux_B

    np.savetxt(location + filename_bulk + 'rvA.txt', rvA_array)
    np.savetxt(location + filename_bulk + 'rv_initial.txt', RVs_initial)
    np.savetxt(location + filename_bulk + 'rvB.txt', rvB_array)
    np.savetxt(location + filename_bulk + 'sep_flux.txt', sep_array)

    vel_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
    bf_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
    bf_smooth_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
    model_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
    for i in range(0, bf_fitres_A.size):
        model_vals_A, bf_velocity_A, bf_vals_A, bf_smooth_vals_A = bf_fitres_A[i][1:]
        vel_array[i, :] = bf_velocity_A
        bf_array[i, :] = bf_vals_A
        bf_smooth_array[i, :] = bf_smooth_vals_A
        model_array[i, :] = model_vals_A
    np.savetxt(location + filename_bulk + 'velocities_A.txt', vel_array)
    np.savetxt(location + filename_bulk + 'bfvals_A.txt', bf_array)
    np.savetxt(location + filename_bulk + 'bfsmooth_A.txt', bf_smooth_array)
    np.savetxt(location + filename_bulk + 'models_A.txt', model_array)

    for i in range(0, bf_fitres_B.size):
        model_vals_B, bf_velocity_B, bf_vals_B, bf_smooth_vals_B = bf_fitres_B[i][1:]
        vel_array[i, :] = bf_velocity_B
        bf_array[i, :] = bf_vals_B
        bf_smooth_array[i, :] = bf_smooth_vals_B
        model_array[i, :] = model_vals_B
    np.savetxt(location + filename_bulk + 'velocities_B.txt', vel_array)
    np.savetxt(location + filename_bulk + 'bfvals_B.txt', bf_array)
    np.savetxt(location + filename_bulk + 'bfsmooth_B.txt', bf_smooth_array)
    np.savetxt(location + filename_bulk + 'models_B.txt', model_array)


def estimate_errors_rv_only(
        wavelength_interval_size: float, inv_flux_collection: np.ndarray, inv_flux_templateA: np.ndarray,
        inv_flux_templateB: np.ndarray, separated_flux_A: np.ndarray, separated_flux_B: np.ndarray, delta_v: float,
        ifitparamsA: InitialFitParameters, ifitparamsB: InitialFitParameters, wavelength: np.ndarray,
        RV_collection_A: np.ndarray, RV_collection_B: np.ndarray, times: np.ndarray, wavelength_buffer_size=100.0,
        plot=False, period=1.0
):
    interval_results = _create_wavelength_intervals(
        wavelength, wavelength_interval_size, inv_flux_collection, inv_flux_templateA, inv_flux_templateB,
        wavelength_buffer_size, separated_flux_A=separated_flux_A, separated_flux_B=separated_flux_B
    )
    (wavelength_interval_collection, flux_interval_collection, templateA_interval_collection,
     templateB_interval_collection, sep_flux_A_interval_collection, sep_flux_B_interval_collection, interval_buffer_mask
     ) = interval_results

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
            fig_2 = plt.figure(figsize=(8, 9))
            gs_2 = fig_2.add_gridspec(1, 1)
            f2_ax1 = fig_2.add_subplot(gs_2[:, :])

            # Broadening function fits B
            fig_3 = plt.figure(figsize=(8, 9))
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

    errors_RV_A = np.std(RV_estimates_A, axis=1) / np.sqrt(RV_estimates_A[0, :].size)
    errors_RV_B = np.std(RV_estimates_B, axis=1) / np.sqrt(RV_estimates_B[0, :].size)
    return (errors_RV_A, errors_RV_B), (RV_estimates_A, RV_estimates_B)


def _create_wavelength_intervals(
        wavelength, wavelength_interval_size, inv_flux_collection, inv_flux_templateA, inv_flux_templateB,
        wavelength_buffer_size, separated_flux_A: np.ndarray = None, separated_flux_B: np.ndarray = None
):
    wavelength_interval_collection = []
    flux_interval_collection = []
    templateA_interval_collection = []
    templateB_interval_collection = []
    interval_buffer_mask = []
    separated_A_interval_collection = []
    separated_B_interval_collection = []
    w_interval_start = wavelength[0] + wavelength_buffer_size
    while True:
        if w_interval_start + wavelength_interval_size > wavelength[-1] - wavelength_buffer_size:
            w_interval_end = wavelength[-1] - wavelength_buffer_size
        else:
            w_interval_end = w_interval_start + wavelength_interval_size

        if w_interval_end - w_interval_start < wavelength_interval_size // 2:
            break

        w_interval = (w_interval_start, w_interval_end)
        if separated_flux_A is not None and separated_flux_B is not None:
            _, _, wl_buffered, flux_buffered_list, buffer_mask = spf.limit_wavelength_interval_multiple_spectra(
                w_interval, wavelength, inv_flux_collection, inv_flux_templateA, inv_flux_templateB, separated_flux_A,
                separated_flux_B, buffer_size=wavelength_buffer_size, even_length=True
            )
            [fl_buffered, flA_buffered, flB_buffered, sflA_buffered, sflB_buffered] = flux_buffered_list
            separated_A_interval_collection.append(sflA_buffered)
            separated_B_interval_collection.append(sflB_buffered)
        else:
            _, _, wl_buffered, flux_buffered_list, buffer_mask = spf.limit_wavelength_interval_multiple_spectra(
                w_interval, wavelength, inv_flux_collection, inv_flux_templateA, inv_flux_templateB,
                buffer_size=wavelength_buffer_size, even_length=True
            )
            [fl_buffered, flA_buffered, flB_buffered] = flux_buffered_list

        wavelength_interval_collection.append(wl_buffered)
        flux_interval_collection.append(fl_buffered)
        templateA_interval_collection.append(flA_buffered)
        templateB_interval_collection.append(flB_buffered)
        interval_buffer_mask.append(buffer_mask)

        w_interval_start = w_interval_end
    if separated_flux_A is not None and separated_flux_B is not None:
        return (wavelength_interval_collection, flux_interval_collection, templateA_interval_collection,
                templateB_interval_collection, separated_A_interval_collection, separated_B_interval_collection,
                interval_buffer_mask)
    else:
        return (wavelength_interval_collection, flux_interval_collection, templateA_interval_collection,
                templateB_interval_collection, interval_buffer_mask)


def estimate_errors_2(
        wavelength_interval_size: int, inv_flux_collection: np.ndarray,
        inv_flux_templateA: np.ndarray, inv_flux_templateB: np.ndarray, delta_v: float,
        ifitparamsA: InitialFitParameters, ifitparamsB: InitialFitParameters, wavelength: np.ndarray,
        time_values: np.ndarray, RV_collection: np.ndarray, convergence_limit=1E-5,
        iteration_limit=10, plot=True, period=None, wavelength_buffer_size=100, rv_lower_limit=0.0,
        suppress_print=False, convergence_limit_scs=1E-7, save_bf_plots=False, ignore_component_B=False
):
    (wavelength_interval_collection, flux_interval_collection, templateA_interval_collection,
     templateB_interval_collection, interval_buffer_mask) = _create_wavelength_intervals(
        wavelength, wavelength_interval_size, inv_flux_collection, inv_flux_templateA, inv_flux_templateB,
        wavelength_buffer_size
    )

    RV_A_interval_values = np.empty((RV_collection[:, 0].size, len(wavelength_interval_collection)))
    RV_B_interval_values = np.empty((RV_collection[:, 0].size, len(wavelength_interval_collection)))
    for i in range(0, len(wavelength_interval_collection)):
        current_wl, current_fl = wavelength_interval_collection[i], flux_interval_collection[i]
        current_fltA, current_fltB = templateA_interval_collection[i], templateB_interval_collection[i]
        current_buffer = interval_buffer_mask[i]

        if save_bf_plots:
            matplotlib.rcParams.update({'font.size': 25})
            for k in range(0, current_fl[0, :].size):
                current_wl_unbuffered = current_wl[~current_buffer]
                current_fl_unbuffered = current_fl[~current_buffer, :]
                current_fltA_unbuffered = current_fltA[~current_buffer]

                BF = bfsvd.BroadeningFunction(current_fl_unbuffered[:, k], current_fltA_unbuffered,
                                              ifitparamsA.bf_velocity_span, delta_v)
                BF.smooth_sigma = ifitparamsA.bf_smooth_sigma
                BF.solve()
                BF.smooth()
                plt.figure(figsize=(16, 9))
                plt.plot(BF.velocity, BF.bf_smooth / np.max(BF.bf_smooth), 'r', linewidth=3)
                plt.xlabel('Velocity Shift [km/s]')
                plt.ylabel('Normalized Smoothed Broadening Function')
                plt.xlim([-100, 100])
                plt.tight_layout()
                plt.savefig(fname=f'../figures/BF_plots/{k}/{current_wl_unbuffered[0]}_{current_wl_unbuffered[-1]}.png', dpi=400)
                plt.close()

        if plot:
            save_plot_path = f'../figures/spectrum_plots/group_plots/{current_wl[~current_buffer][0]}_{current_wl[~current_buffer][1]}_multi.pdf'
        else:
            save_plot_path = None

        RV_A_temp, RV_B_temp, _, _, _, _, _ = spectral_separation_routine(
            current_fl, current_fltA, current_fltB, delta_v, ifitparamsA, ifitparamsB, current_wl, time_values,
            RV_collection, convergence_limit, iteration_limit, plot, period, buffer_mask=current_buffer,
            rv_lower_limit=rv_lower_limit, suppress_print=suppress_print, convergence_limit_scs=convergence_limit_scs,
            save_plot_path=save_plot_path, ignore_component_B=ignore_component_B
        )
        if plot:
            plt.close('all')
        RV_A_interval_values[:, i] = RV_A_temp
        RV_B_interval_values[:, i] = RV_B_temp

    # bad_RVB_value_mask = np.abs(RV_A_interval_values) < rv_lower_limit
    RV_errors_A = np.std(RV_A_interval_values, axis=1) / np.sqrt(RV_A_interval_values[0, :].size)
    RV_errors_B = np.std(RV_B_interval_values, axis=1) / np.sqrt(RV_B_interval_values[0, :].size)
    return RV_errors_A, RV_errors_B, (RV_A_interval_values, RV_B_interval_values)
