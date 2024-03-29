from RV.library.spectrum_processing_functions import resample_to_equal_velocity_steps
import RV.library.test.test_library.radial_velocity_functions as rf
from RV.library.calculate_radial_velocities import *
import matplotlib.pyplot as plt
from RV.library.test.test_library.detect_peaks import detect_peaks
import os


def test_radial_velocity_from_broadening_function(data_path):
    data          = np.loadtxt(data_path+'15162_epoch1.txt')
    data_template = np.loadtxt(data_path+'5250_40_02_template.dat')
    wavelength, flux = data[:, 0], data[:, 1]
    wavelength_template, flux_template = data_template[:, 0], data_template[:, 1]
    delta_v = 1.1
    wavelength, flux          = resample_to_equal_velocity_steps(wavelength, delta_v, flux)
    wavelength, flux_template = resample_to_equal_velocity_steps(wavelength_template, delta_v, flux_template,
                                                                 wavelength)
    # Plot flux
    plt.figure()
    plt.plot(wavelength, flux)
    plt.plot(wavelength, flux_template)
    plt.show()

    # Invert fluxes
    flux_inverted = np.mean(flux) - flux
    flux_template_inverted = np.mean(flux_template) - flux_template

    # Plot inverted flux
    plt.figure()
    plt.plot(wavelength, flux_inverted)
    plt.plot(wavelength, flux_template_inverted)
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Inverted Flux')
    plt.legend(['Observation', 'Template'])
    plt.show()

    # Cross correlation
    cross_correlation   = rf.cross_correlate(flux, flux_template, plot=False)[0]
    # wavelength_shift_cc = wavelength - wavelength[len(wavelength)//2]
    velocity_shift_cc   = np.linspace(-wavelength.size//2, wavelength.size//2, wavelength.size) * delta_v

    # Broadening Function
    BFsvd = BroadeningFunction(flux_inverted, flux_template_inverted, 401, delta_v, plot_w=True)
    BFsvd.solve()
    BFsvd.smooth()

    print(BFsvd.bf.shape)

    # Plot
    plt.plot(velocity_shift_cc, cross_correlation)
    plt.plot(BFsvd.velocity, BFsvd.bf/np.max(BFsvd.bf))
    plt.legend(['Cross Correlation', 'BF vel window 301 dv 2.0'])
    plt.xlabel('Velocity shift [km/s]')
    plt.xlim(-200, 200)
    plt.show()

    plt.plot(velocity_shift_cc, cross_correlation)
    # plt.plot(BFsvd.velocity, BFsvd.bf / np.max(BFsvd.bf))
    plt.plot(BFsvd.velocity, BFsvd.bf_smooth / np.max(BFsvd.bf_smooth))
    plt.legend(['Cross Correlation', 'BF smoothed dv 2.0'])
    plt.xlabel('Velocity shift [km/s]')
    plt.xlim(-200, 200)
    plt.show()

    # detect peaks in cross correlation and find RVs
    peak_idx = detect_peaks(cross_correlation, mph=0.2)
    print('RV values cross correlation')
    print([velocity_shift_cc[i] for i in peak_idx])

    # fit profile in BFsvd and find RVs
    initial_fitparams_A = InitialFitParameters(3.0, 19800, 50, 0.68)
    initial_fitparams_B = InitialFitParameters(3.0, 19800, 100, 0.68)
    RVs, fits = radial_velocity_from_broadening_function(flux_inverted, BFsvd, initial_fitparams_A, initial_fitparams_B)
    print(RVs)

    plt.figure()
    plt.plot(BFsvd.velocity, BFsvd.bf_smooth)
    plt.plot(BFsvd.velocity, fits[0], 'k--')
    plt.plot(BFsvd.velocity, fits[2], 'k--')
    plt.show()


def test_multiple_spectra(data_path):
    data_template = np.loadtxt(data_path+'5250_40_02_template.dat')
    delta_v = 2.0
    wavelength_template, flux_template = data_template[:, 0], data_template[:, 1]
    wavelength, flux_template = resample_to_equal_velocity_steps(wavelength_template, delta_v, flux_template)

    flux_collection = np.empty((flux_template.size, 14))
    i = 0
    for filename in os.listdir(data_path):
        if '15162' in filename:
            data = np.loadtxt(data_path+filename)
            _, flux_collection[:, i] = resample_to_equal_velocity_steps(data[:, 0], delta_v, data[:, 1],
                                                                                 wavelength)
        i += 1

    # Invert fluxes
    flux_collection_inverted = np.mean(flux_collection, axis=0) - flux_collection
    flux_template_inverted   = np.mean(flux_template) - flux_template

    # Broadening function fit
    initial_fitparams_A = InitialFitParameters(3.0, 19800, 50, 0.68)
    initial_fitparams_B = InitialFitParameters(3.0, 19800, 100, 0.68)
    RVs_A, RVs_B, _ = radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_template_inverted, delta_v,
                                                            initial_fitparams_A, initial_fitparams_B,
                                                            number_of_parallel_jobs=8, plot=True, bf_velocity_span=801)
    print(RVs_A)
    print(RVs_B)


def main():
    data_path = 'test_data/'
    # test_radial_velocity_from_broadening_function(data_path)
    test_multiple_spectra(data_path)


if __name__ == "__main__":
    main()