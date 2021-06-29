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

from RV.library.calculate_radial_velocities import radial_velocities_of_multiple_spectra, \
    radial_velocity_single_component
from RV.library.broadening_function_svd import *
from RV.library.initial_fit_parameters import InitialFitParameters
from lmfit.minimizer import MinimizerResult
import RV.library.spectrum_processing_functions as spf


def shift_spectrum(flux, radial_velocity_shift, delta_v):
    shift = int(radial_velocity_shift / delta_v)
    return np.roll(flux, shift)


def separate_component_spectra(flux_collection, radial_velocity_collection_A, radial_velocity_collection_B, delta_v,
                               convergence_limit, suppress_scs=False, max_iterations=20, rv_lower_limit=0.0):
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
    :param rv_lower_limit:                float, lower RV limit in order to add separated spectrum (if below, the
                                          components are expected to be mixed, and are not included to avoid pollution)

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
            rvA = radial_velocity_collection_A[i]
            rvB = radial_velocity_collection_B[i]
            if np.abs(rvA) > rv_lower_limit:
                shifted_flux_A = shift_spectrum(flux_collection[:, i], -rvA, delta_v)
                separated_flux_A += shifted_flux_A - shift_spectrum(separated_flux_B, rvB - rvA, delta_v)
                n_used_spectra += 1
        separated_flux_A = separated_flux_A / n_used_spectra

        separated_flux_B = np.zeros((flux_collection[:, 0].size,))
        n_used_spectra = 0
        for i in range(0, n_spectra):
            rvA = radial_velocity_collection_A[i]
            rvB = radial_velocity_collection_B[i]
            if np.abs(rvA) > rv_lower_limit:
                shifted_flux_B = shift_spectrum(flux_collection[:, i], -rvB, delta_v)
                separated_flux_B += shifted_flux_B - shift_spectrum(separated_flux_A, rvA - rvB, delta_v)
                n_used_spectra += 1
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


def update_bf_plot(plot_ax, model, index, rv_lower_limit):
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


def recalculate_RVs(flux_collection_inverted, separated_flux_A, separated_flux_B, RV_collection_A, RV_collection_B,
                    flux_templateA_inverted, flux_templateB_inverted, delta_v, ifitparamsA:InitialFitParameters,
                    ifitparamsB:InitialFitParameters, buffer_mask, plot_ax_A=None, plot_ax_B=None, rv_lower_limit=0.0):
    n_spectra = flux_collection_inverted[0, :].size
    v_span = ifitparamsA.bf_velocity_span
    BRsvd_template_A = BroadeningFunction(flux_collection_inverted[~buffer_mask, 0],
                                          flux_templateA_inverted[~buffer_mask], v_span, delta_v)
    BRsvd_template_B = BroadeningFunction(flux_collection_inverted[~buffer_mask, 0],
                                          flux_templateB_inverted[~buffer_mask], v_span, delta_v)
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

    fits_A = np.empty(shape=(n_spectra,), dtype=MinimizerResult)
    fits_B = np.empty(shape=(n_spectra,), dtype=MinimizerResult)

    for i in range(0, n_spectra):
        corrected_flux_A = flux_collection_inverted[:, i] -shift_spectrum(separated_flux_B, RV_collection_B[i], delta_v)
        corrected_flux_B = flux_collection_inverted[:, i] -shift_spectrum(separated_flux_A, RV_collection_A[i], delta_v)

        corrected_flux_A = corrected_flux_A[~buffer_mask]
        corrected_flux_B = corrected_flux_B[~buffer_mask]

        ifitparamsA.RV = RV_collection_A[i]
        ifitparamsB.RV = RV_collection_B[i]

        RV_collection_A[i], model_A = radial_velocity_single_component(corrected_flux_A, BRsvd_template_A, ifitparamsA)
        RV_collection_B[i], model_B = radial_velocity_single_component(corrected_flux_B, BRsvd_template_B, ifitparamsB)

        fits_A[i], fits_B[i] = model_A[0], model_B[0]

        if plot_ax_A is not None and i < 20:
            update_bf_plot(plot_ax_A, model_A, i, rv_lower_limit)
            if rv_lower_limit != 0.0:
                plot_ax_A.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_A.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
        if plot_ax_B is not None and i < 20:
            update_bf_plot(plot_ax_B, model_B, i, rv_lower_limit)
            if rv_lower_limit != 0.0:
                plot_ax_B.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_B.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
    return RV_collection_A, RV_collection_B, (fits_A, fits_B)


def initialize_ssr_plots():
    # RVs and separated spectra
    fig_1 = plt.figure(figsize=(16, 9))
    gs_1 = fig_1.add_gridspec(2, 2)
    f1_ax1 = fig_1.add_subplot(gs_1[0, :])
    f1_ax2 = fig_1.add_subplot(gs_1[1, 0])
    f1_ax3 = fig_1.add_subplot(gs_1[1, 1])

    # Program spectra
    fig_2 = plt.figure(figsize=(16, 9))
    gs_2 = fig_2.add_gridspec(1, 1)
    f2_ax1 = fig_2.add_subplot(gs_2[:, :])

    # RV corrected program spectra
    fig_3 = plt.figure(figsize=(16, 9))
    gs_3 = fig_3.add_gridspec(1, 2)
    f3_ax1 = fig_3.add_subplot(gs_3[0, 0])
    f3_ax2 = fig_3.add_subplot(gs_3[0, 1])

    # Broadening function fits A
    fig_4 = plt.figure(figsize=(16, 9))
    gs_4 = fig_4.add_gridspec(1, 1)
    f4_ax1 = fig_4.add_subplot(gs_4[:, :])

    # Broadening function fits B
    fig_5 = plt.figure(figsize=(16, 9))
    gs_5 = fig_5.add_gridspec(1, 1)
    f5_ax1 = fig_5.add_subplot(gs_5[:, :])
    return f1_ax1, f1_ax2, f1_ax3, f2_ax1, f3_ax1, f3_ax2, f4_ax1, f5_ax1


def plot_ssr_iteration(f1_ax1, f1_ax2, f1_ax3, f2_ax1, f3_ax1, f3_ax2, separated_flux_A, separated_flux_B, wavelength,
                       flux_template_A, flux_template_B, RV_A, RV_B, time, period, flux_collection, buffer_mask,
                       rv_lower_limit):
    f1_ax1.clear(); f1_ax2.clear(); f1_ax3.clear(); f2_ax1.clear()
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

    for i in range(0, np.min([5, flux_collection[0, :].size])):
        f2_ax1.plot(wavelength, 1-0.15*i-(flux_collection[:, i]*0.1))
        f2_ax1.plot(wavelength, np.ones(shape=wavelength.shape)-0.15*i, '--', color='grey', linewidth=0.5)
    f2_ax1.set_xlabel('Wavelength [Å]')
    f2_ax1.set_xlabel('Normalized Flux')

    for i in range(0, np.min([5, flux_collection[0, :].size])):
        f3_ax1.plot(wavelength, 1-0.15*i-0.1*shift_spectrum(flux_collection[:, i], -RV_A[i], delta_v=1.0))
        f3_ax2.plot(wavelength, 1-0.15*i-0.1*shift_spectrum(flux_collection[:, i], -RV_B[i], delta_v=1.0))
    f3_ax1.set_xlabel('Wavelength [Å]')
    f3_ax2.set_xlabel('Wavelength [Å]')
    f3_ax1.set_xlim([5385, 5405])
    f3_ax2.set_xlim([5385, 5405])

    plt.draw_all()
    plt.pause(3)
    # plt.show(block=True)


def spectral_separation_routine(flux_collection_inverted, flux_templateA_inverted, flux_templateB_inverted, delta_v,
                                ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters, wavelength,
                                time_values, RV_guess_collection, convergence_limit=1E-5, iteration_limit=10, plot=True,
                                period=None, buffer_mask=None, rv_lower_limit=0.0, suppress_print=False,
                                convergence_limit_scs=1E-7):
    suppress_scs = False; suppress_ssr = False
    if suppress_print == 'scs': suppress_scs = True
    elif suppress_print == 'ssr': suppress_ssr = True
    elif suppress_print == 'all': suppress_scs = True; suppress_ssr = True

    RV_collection_A, RV_collection_B = RV_guess_collection[:, 0], RV_guess_collection[:, 1]

    if buffer_mask is None:
        buffer_mask = np.zeros(wavelength.shape, dtype=bool)

    # Create buffer mask
    # buffer_mask = (wavelength > wavelength[0] + wavelength_buffer_size) & \
    #               (wavelength < wavelength[-1] - wavelength_buffer_size)
    # if np.mod(wavelength[buffer_mask].size, 2) != 0.0:
    #     indxs = np.argwhere(buffer_mask)
    #     buffer_mask[indxs[-1]] = False

    # Initialize plot figures
    if plot:
        f1_ax1, f1_ax2, f1_ax3, f2_ax1, f3_ax1, f3_ax2, f4_ax1, f5_ax1 = initialize_ssr_plots()
    else:
        f4_ax1 = None; f5_ax1 = None

    # Iterative loop that repeatedly separates the spectra from each other in order to calculate new RVs (Gonzales 2005)
    iterations = 0
    print('Spectral Separation: ')
    while True:
        print(f'\nIteration {iterations}.')
        RV_mask = np.abs(RV_collection_A) > rv_lower_limit
        RMS_A, RMS_B = -RV_collection_A, -RV_collection_B[RV_mask]

        separated_flux_A, separated_flux_B = separate_component_spectra(flux_collection_inverted, RV_collection_A,
                                                                        RV_collection_B, delta_v, convergence_limit_scs,
                                                                        suppress_scs, rv_lower_limit=rv_lower_limit)

        RV_collection_A, RV_collection_B, (fits_A, fits_B) \
            = recalculate_RVs(flux_collection_inverted, separated_flux_A, separated_flux_B, RV_collection_A,
                              RV_collection_B, flux_templateA_inverted, flux_templateB_inverted, delta_v, ifitparamsA,
                              ifitparamsB, buffer_mask, f4_ax1, f5_ax1, rv_lower_limit=rv_lower_limit)

        if plot:
            plot_ssr_iteration(f1_ax1, f1_ax2, f1_ax3, f2_ax1, f3_ax1, f3_ax2, separated_flux_A, separated_flux_B,
                               wavelength, flux_templateA_inverted, flux_templateB_inverted, RV_collection_A,
                               RV_collection_B, time_values, period, flux_collection_inverted, buffer_mask,
                               rv_lower_limit=rv_lower_limit)

        # Average vsini values for future fit guess and limit allowed fit area
        vsini_A, vsini_B = np.empty(shape=fits_A.shape), np.empty(shape=fits_B.shape)
        for i in range(0, vsini_A.size):
            _, _, vsini_A[i], _, _, _ = get_fit_parameter_values(fits_A[i].params)
            _, _, vsini_B[i], _, _, _ = get_fit_parameter_values(fits_B[i].params)
        ifitparamsA.vsini = np.mean(vsini_A)
        ifitparamsB.vsini = np.mean(vsini_B)
        ifitparamsA.vsini_vary_limit = 0.3
        ifitparamsB.vsini_vary_limit = 0.3

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
    return RV_collection_A, RV_collection_B, separated_flux_A[buffer_mask], separated_flux_B[buffer_mask], \
           wavelength[buffer_mask]


def estimate_errors(wavelength_interval_size, flux_collection_inverted, flux_templateA_inverted,
                    flux_templateB_inverted, delta_v, ifitparamsA:InitialFitParameters,
                    ifitparamsB:InitialFitParameters, wavelength, time_values, RV_collection, convergence_limit=1E-5,
                    iteration_limit=10, plot=True, period=None, wavelength_buffer_size=100, rv_lower_limit=0.0,
                    suppress_print=False, convergence_limit_scs=1E-7):

    wavelength_intervals = []
    flux_intervals = []
    flux_templateA_intervals = []
    flux_templateB_intervals = []
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
            spf.limit_wavelength_interval(w_interval, wavelength, flux_collection_inverted,
                                          buffer_size=wavelength_buffer_size, even_length=True)
        _, (_, flA_buffered, _, _) = spf.limit_wavelength_interval(w_interval, wavelength, flux_templateA_inverted,
                                                                   buffer_mask=buffer_mask_internal, even_length=True)
        _, (_, flB_buffered, _, _) = spf.limit_wavelength_interval(w_interval, wavelength, flux_templateB_inverted,
                                                                   buffer_mask=buffer_mask_internal, even_length=True)

        wavelength_intervals.append(wl_buffered)
        flux_intervals.append(fl_buffered)
        flux_templateA_intervals.append(flA_buffered)
        flux_templateB_intervals.append(flB_buffered)
        interval_buffer_mask.append(buffer_mask)

        w_interval_start = w_interval_end

    RV_A_interval_values = np.empty((RV_collection[:, 0].size, len(wavelength_intervals)))
    RV_B_interval_values = np.empty((RV_collection[:, 0].size, len(wavelength_intervals)))
    for i in range(0, len(wavelength_intervals)):
        current_wl, current_fl = wavelength_intervals[i], flux_intervals[i]
        current_fltA, current_fltB = flux_templateA_intervals[i], flux_templateB_intervals[i]
        current_buffer = interval_buffer_mask[i]

        RV_A_temp, RV_B_temp, _, _, _ = spectral_separation_routine(current_fl, current_fltA, current_fltB, delta_v,
                                                                    ifitparamsA, ifitparamsB, current_wl, time_values,
                                                                    RV_collection, convergence_limit, iteration_limit,
                                                                    plot, period, current_buffer, rv_lower_limit,
                                                                    suppress_print, convergence_limit_scs)
        if plot:
            plt.close('all')
        RV_A_interval_values[:, i] = RV_A_temp
        RV_B_interval_values[:, i] = RV_B_temp

    # bad_RVB_value_mask = np.abs(RV_A_interval_values) < rv_lower_limit
    RV_errors_A = np.std(RV_A_interval_values, axis=1)
    RV_errors_B = np.std(RV_B_interval_values, axis=1)
    return RV_errors_A, RV_errors_B
