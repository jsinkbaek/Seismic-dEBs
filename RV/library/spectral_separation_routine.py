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


def shift_spectrum(flux, radial_velocity_shift, delta_v):
    shift = int(radial_velocity_shift / delta_v)
    return np.roll(flux, shift)


def separate_component_spectra(flux_collection, radial_velocity_collection_A, radial_velocity_collection_B, delta_v,
                               convergence_limit, max_iterations=20):
    """
    Assumes that component A is the dominant component in the spectrum.
    :param flux_collection:               np.ndarray shape (datasize, nspectra) of all the observed spectra
    :param radial_velocity_collection_A:  np.ndarray shape (nspectra, ) of radial velocity values for component A
    :param radial_velocity_collection_B:  np.ndarray shape (nspectra, ) of radial velocity values for component B
    :param delta_v:                       float, the sampling size of the spectra in velocity space
    :param convergence_limit:             float, the precision needed to break while loop
    :param max_iterations:                int, maximum number of allowed iterations before breaking loop

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
        for i in range(0, n_spectra):
            rvA = radial_velocity_collection_A[i]
            rvB = radial_velocity_collection_B[i]
            shifted_flux_A = shift_spectrum(flux_collection[:, i], -rvA, delta_v)
            separated_flux_A += shifted_flux_A - shift_spectrum(separated_flux_B, rvB - rvA, delta_v)
        separated_flux_A = separated_flux_A / n_spectra

        separated_flux_B = np.zeros((flux_collection[:, 0].size,))
        for i in range(0, n_spectra):
            rvA = radial_velocity_collection_A[i]
            rvB = radial_velocity_collection_B[i]
            shifted_flux_B = shift_spectrum(flux_collection[:, i], -rvB, delta_v)
            separated_flux_B += shifted_flux_B - shift_spectrum(separated_flux_A, rvA - rvB, delta_v)
        separated_flux_B = separated_flux_B / n_spectra

        RMS_values_A += separated_flux_A
        RMS_values_B += separated_flux_B
        RMS_A = np.sum(RMS_values_A**2)/RMS_values_A.size
        RMS_B = np.sum(RMS_values_B**2)/RMS_values_B.size
        if RMS_A < convergence_limit and RMS_B < convergence_limit:
            print(f'Separate Component Spectra: Convergence limit of {convergence_limit} successfully reached in '
                  f'{iteration_counter} iterations. \nReturning last separated spectra.')
            break
        elif iteration_counter >= max_iterations:
            warnings.warn(f'Warning: Iteration limit of {max_iterations} reached without reaching convergence limit'
                          f' of {convergence_limit}. \nCurrent RMS_A: {RMS_A}. RMS_B: {RMS_B} \n'
                          'Returning last separated spectra.')
            break

    return separated_flux_A, separated_flux_B


def update_bf_plot(plot_ax, model, index):
    fit = model[0]
    model_values = model[1]
    velocity_values = model[2]
    bf_smooth_values = model[4]
    amplitude, RV, _, _, _, _ = get_fit_parameter_values(fit.params)
    plot_ax.plot(velocity_values, 1+0.025*bf_smooth_values/np.max(bf_smooth_values)-0.05*index)
    plot_ax.plot(velocity_values, 1+0.025*model_values/np.max(bf_smooth_values)-0.05*index, 'k--')
    plot_ax.plot(np.ones(shape=(2,))*RV, [1-0.05*index,
                                          1+0.025*np.max(model_values)/np.max(bf_smooth_values)-0.05*index], '--',
                 color='grey')


def recalculate_RVs(flux_collection_inverted, separated_flux_A, separated_flux_B, RV_collection_A, RV_collection_B,
                    flux_templateA_inverted, flux_templateB_inverted, delta_v, ifitparamsA:InitialFitParameters,
                    ifitparamsB:InitialFitParameters, plot_ax_A=None, plot_ax_B=None):
    n_spectra = flux_collection_inverted[0, :].size
    v_span = ifitparamsA.bf_velocity_span
    BRsvd_template_A = BroadeningFunction(flux_collection_inverted[:, 0], flux_templateA_inverted, v_span, delta_v)
    BRsvd_template_B = BroadeningFunction(flux_collection_inverted[:, 0], flux_templateB_inverted, v_span, delta_v)
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
        corrected_flux_b = flux_collection_inverted[:, i] -shift_spectrum(separated_flux_A, RV_collection_A[i], delta_v)

        ifitparamsA.RV = RV_collection_A[i]
        ifitparamsB.RV = RV_collection_B[i]

        RV_collection_A[i], model_A = radial_velocity_single_component(corrected_flux_A, BRsvd_template_A, ifitparamsA)
        RV_collection_B[i], model_B = radial_velocity_single_component(corrected_flux_b, BRsvd_template_B, ifitparamsB)

        fits_A[i], fits_B[i] = model_A[0], model_B[0]

        if plot_ax_A is not None and i < 20:
            update_bf_plot(plot_ax_A, model_A, i)
        if plot_ax_B is not None and i < 20:
            update_bf_plot(plot_ax_B, model_B, i)
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
                       flux_template_A, flux_template_B, RV_A, RV_B, time, period, flux_collection):
    f1_ax1.clear(); f1_ax2.clear(); f1_ax3.clear(); f2_ax1.clear()
    if period is None:
        xval = time
        f1_ax1.set_xlabel('BJD - 245000')
    else:
        xval = np.mod(time, period) / period
        f1_ax1.set_xlabel('Phase')
    f1_ax1.plot(xval, RV_A, 'b*')
    f1_ax1.plot(xval, RV_B, 'r*')

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
                                period=None):
    RV_collection_A, RV_collection_B = RV_guess_collection[:, 0], RV_guess_collection[:, 1]

    # Initialize plot figures
    if plot:
        f1_ax1, f1_ax2, f1_ax3, f2_ax1, f3_ax1, f3_ax2, f4_ax1, f5_ax1 = initialize_ssr_plots()
    else:
        f4_ax1 = None; f5_ax1 = None

    # Iterative loop that repeatedly separates the spectra from each other in order to calculate new RVs (Gonzales 2005)
    iterations = 0
    while True:
        RMS_A, RMS_B = -RV_collection_A, -RV_collection_B

        separated_flux_A, separated_flux_B = separate_component_spectra(flux_collection_inverted, RV_collection_A,
                                                                        RV_collection_B, delta_v, convergence_limit)

        RV_collection_A, RV_collection_B, (fits_A, fits_B) \
            = recalculate_RVs(flux_collection_inverted, separated_flux_A, separated_flux_B, RV_collection_A,
                              RV_collection_B, flux_templateA_inverted, flux_templateB_inverted, delta_v, ifitparamsA,
                              ifitparamsB, f4_ax1, f5_ax1)

        if plot:
            plot_ssr_iteration(f1_ax1, f1_ax2, f1_ax3, f2_ax1, f3_ax1, f3_ax2, separated_flux_A, separated_flux_B,
                               wavelength, flux_templateA_inverted, flux_templateB_inverted, RV_collection_A,
                               RV_collection_B, time_values, period, flux_collection_inverted)

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
        RMS_B += RV_collection_B
        if np.sum(RMS_A**2) < convergence_limit and np.sum(RMS_B**2) < convergence_limit:
            print('Spectral separation routine terminates.')
            break
        if iterations >= iteration_limit:
            warnings.warn(f'RV convergence limit of {convergence_limit} not reached in {iterations} iterations.')
            print('Spectral separation routine terminates.')
            break
    return RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B

