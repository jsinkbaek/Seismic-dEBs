import numpy as np
import matplotlib.pyplot as plt
import os


class IntervalResult:
    def __init__(
            self, time_values_A, time_values_B, wavelength, separated_spectrum_A, separated_spectrum_B, RV_A, RV_B, bf_velocity_A, bf_vals_A,
            bf_smooth_A, bf_model_vals_A, bf_velocity_B, bf_vals_B, bf_smooth_B, bf_model_vals_B, wavelength_a=None,
            wavelength_b=None
    ):
        self.time_values_A = time_values_A
        self.time_values_B = time_values_B
        self.wavelength = wavelength
        self.separated_spectrum_A = separated_spectrum_A
        self.separated_spectrum_B = separated_spectrum_B
        self.RV_A = RV_A
        self.RV_B = RV_B
        self.bf_velocity_A = bf_velocity_A
        self.bf_vals_A = bf_vals_A
        self.bf_smooth_A = bf_smooth_A
        self.bf_model_vals_A = bf_model_vals_A
        self.bf_velocity_B = bf_velocity_B
        self.bf_vals_B = bf_vals_B
        self.bf_smooth_B = bf_smooth_B
        self.bf_model_vals_B = bf_model_vals_B
        if wavelength_a is None:
            self.wavelength_a = int(np.round(np.min(wavelength)))
        if wavelength_b is None:
            self.wavelength_b = int(np.round(np.max(wavelength)))


class RoutineResults:
    def __init__(self, *args: IntervalResult):
        if len(args) != 0:
            self.interval_results = [x for x in args]
        else:
            self.interval_results = []

    @property
    def time_values_A(self, index=None):
        if index is not None:
            return self.interval_results[index].time_values_A
        else:
            return [x.time_values_A for x in self.interval_results]

    @property
    def time_values_B(self, index=None):
        if index is not None:
            return self.interval_results[index].time_values_B
        else:
            return [x.time_values_B for x in self.interval_results]

    @property
    def RV_A(self, index=None):
        if index is not None:
            return self.interval_results[index].RV_A
        else:
            return [x.RV_A for x in self.interval_results]

    @property
    def RV_B(self, index=None):
        if index is not None:
            return self.interval_results[index].RV_B
        else:
            return [x.RV_B for x in self.interval_results]

    @property
    def wavelengths(self, interval=None, index=None):
        if index is not None:
            return self.interval_results[index].wavelength
        elif interval is not None:
            res = None
            for interval_result in self.interval_results:
                if interval_result.wavelength_a == interval[0] and interval_result.wavelength_b == interval[1]:
                    return interval_result.wavelength
            if res is None:
                raise ValueError('No interval found with the same wavelength_a and wavelength_B.')
        else:
            interval_wavelengths = [x.wavelength for x in self.interval_results]
            return interval_wavelengths

    @property
    def wavelength_a(self, index=None):
        if index is not None:
            return self.interval_results[index].wavelength_a
        else:
            return [x.wavelength_a for x in self.interval_results]

    @property
    def wavelength_b(self, index=None):
        if index is not None:
            return self.interval_results[index].wavelength_b
        else:
            return [x.wavelength_b for x in self.interval_results]

    @property
    def interval(self, index=None):
        if index is not None:
            return self.wavelength_a(index), self.wavelength_b(index)
        else:
            return [(x.wavelength_a, x.wavelength_b) for x in self.interval_results]

    @property
    def separated_spectra_A(self, index=None):
        if index is not None:
            return self.interval_results[index].separated_spectrum_A
        else:
            return [x.separated_spectrum_A for x in self.interval_results]

    @property
    def separated_spectra_B(self, index=None):
        if index is not None:
            return self.interval_results[index].separated_spectrum_B
        else:
            return [x.separated_spectrum_B for x in self.interval_results]

    @property
    def bf_results(self, index=None):
        if index is not None:
            return (self.interval_results[index].bf_velocity_A, self.interval_results[index].bf_vals_A,
                    self.interval_results[index].bf_smooth_A, self.interval_results[index].bf_model_vals_A,
                    self.interval_results[index].bf_velocity_B,
                    self.interval_results[index].bf_vals_B, self.interval_results[index].bf_smooth_B,
                    self.interval_results[index].bf_model_vals_B)
        else:
            return [(x.bf_velocity_A, x.bf_vals_A, x.bf_smooth_A, x.bf_model_vals_A, x.bf_velocity_B, x.bf_vals_B, x.bf_smooth_B,
                    x.bf_model_vals_B) for x in self.interval_results]

    def append_interval(self, new_result: IntervalResult):
        self.interval_results.append(new_result)


def load_routine_results(folder_path: str, filename_bulk_list: list):
    routine_results = RoutineResults()
    for filename_bulk in filename_bulk_list:
        rvA_array = np.loadtxt(folder_path+filename_bulk+'rvA.txt')
        rvB_array = np.loadtxt(folder_path+filename_bulk+'rvB.txt')
        sep_array = np.loadtxt(folder_path+filename_bulk+'sep_flux.txt')
        velA_array = np.loadtxt(folder_path+filename_bulk+'velocities_A.txt')
        bfA_array = np.loadtxt(folder_path+filename_bulk+'bfvals_A.txt')
        bfA_smooth_array = np.loadtxt(folder_path+filename_bulk+'bfsmooth_A.txt')
        modelA_array = np.loadtxt(folder_path+filename_bulk+'models_A.txt')
        velB_array = np.loadtxt(folder_path+filename_bulk+'velocities_B.txt')
        bfB_array = np.loadtxt(folder_path+filename_bulk+'bfvals_B.txt')
        bfB_smooth_array = np.loadtxt(folder_path+filename_bulk+'bfsmooth_B.txt')
        modelB_array = np.loadtxt(folder_path+filename_bulk+'models_B.txt')

        wavelength, separated_flux_A, separated_flux_B = sep_array[:, 0], sep_array[:, 1], sep_array[:, 2]
        time_values_A, RV_A = rvA_array[:, 0], rvA_array[:, 1]
        time_values_B, RV_B = rvB_array[:, 0], rvB_array[:, 1]

        interval_result = IntervalResult(
            time_values_A, time_values_B, wavelength, separated_flux_A, separated_flux_B, RV_A, RV_B, velA_array,
            bfA_array, bfA_smooth_array, modelA_array, velB_array, bfB_array, bfB_smooth_array, modelB_array
        )
        routine_results.append_interval(interval_result)




