import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class IntervalResult:
    def __init__(
            self, time_values_A, time_values_B,
            wavelength, separated_spectrum_A, separated_spectrum_B, template_spectrum_A, template_spectrum_B,
            RV_A, RV_B, RV_B_flags, RV_A_initial, RV_B_initial,
            bf_velocity_A, bf_vals_A, bf_smooth_A, bf_model_vals_A,
            bf_velocity_B, bf_vals_B, bf_smooth_B, bf_model_vals_B,
            wavelength_a=None, wavelength_b=None
    ):
        self.time_values_A = time_values_A
        self.time_values_B = time_values_B
        self.wavelength = wavelength
        self.separated_spectrum_A = separated_spectrum_A
        self.separated_spectrum_B = separated_spectrum_B
        self.template_flux_A = template_spectrum_A
        self.template_flux_B = template_spectrum_B
        self.RV_A = RV_A
        self.RV_B = RV_B
        self.RV_B_flags = RV_B_flags
        self.RV_A_initial = RV_A_initial
        self.RV_B_initial = RV_B_initial
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
    def time_values_A(self):
        return [x.time_values_A for x in self.interval_results]

    @property
    def time_values_B(self):
        return [x.time_values_B for x in self.interval_results]

    @property
    def RV_A(self):
        return [x.RV_A for x in self.interval_results]

    @property
    def RV_B(self):
        return [x.RV_B for x in self.interval_results]

    @property
    def RV_B_flags(self):
        return [x.RV_B_flags for x in self.interval_results]

    @property
    def RV_A_initial(self):
        return [x.RV_A_initial for x in self.interval_results]

    @property
    def RV_B_initial(self):
        return [x.RV_B_initial for x in self.interval_results]

    @property
    def wavelengths(self, interval=None):
        if interval is not None:
            res = None
            for interval_result in self.interval_results:
                if interval_result.wavelength_a == interval[0] and interval_result.wavelength_b == interval[1]:
                    return interval_result.wavelength
            if res is None:
                raise ValueError('No interval found with the same wavelength_a and wavelength_B.')
        else:
            return [x.wavelength for x in self.interval_results]

    @property
    def wavelength_a(self):
        return [x.wavelength_a for x in self.interval_results]

    @property
    def wavelength_b(self):
        return [x.wavelength_b for x in self.interval_results]

    @property
    def interval(self):
        return [(x.wavelength_a, x.wavelength_b) for x in self.interval_results]

    @property
    def separated_spectra_A(self):
        return [x.separated_spectrum_A for x in self.interval_results]

    @property
    def separated_spectra_B(self):
        return [x.separated_spectrum_B for x in self.interval_results]

    @property
    def template_flux_A(self):
        return [x.template_flux_A for x in self.interval_results]

    @property
    def template_flux_B(self):
        return [x.template_flux_B for x in self.interval_results]

    @property
    def bf_results(self):
        return [
            [x.bf_velocity_A, x.bf_vals_A, x.bf_smooth_A, x.bf_model_vals_A, x.bf_velocity_B, x.bf_vals_B,
             x.bf_smooth_B, x.bf_model_vals_B]
            for x in self.interval_results
        ]

    def append_interval(self, new_result: IntervalResult):
        self.interval_results.append(new_result)


def load_routine_results(folder_path: str, filename_bulk_list: list):
    routine_results = RoutineResults()
    for filename_bulk in filename_bulk_list:
        rvA_array = np.loadtxt(folder_path+filename_bulk+'_rvA.txt')
        rvB_array = np.loadtxt(folder_path+filename_bulk+'_rvB.txt')
        sep_array = np.loadtxt(folder_path+filename_bulk+'_sep_flux.txt')
        velA_array = np.loadtxt(folder_path+filename_bulk+'_velocities_A.txt')
        bfA_array = np.loadtxt(folder_path+filename_bulk+'_bfvals_A.txt')
        bfA_smooth_array = np.loadtxt(folder_path+filename_bulk+'_bfsmooth_A.txt')
        modelA_array = np.loadtxt(folder_path+filename_bulk+'_models_A.txt')
        velB_array = np.loadtxt(folder_path+filename_bulk+'_velocities_B.txt')
        bfB_array = np.loadtxt(folder_path+filename_bulk+'_bfvals_B.txt')
        bfB_smooth_array = np.loadtxt(folder_path+filename_bulk+'_bfsmooth_B.txt')
        modelB_array = np.loadtxt(folder_path+filename_bulk+'_models_B.txt')
        rvs_initial = np.loadtxt(folder_path+filename_bulk+'_rv_initial.txt')

        wavelength, separated_flux_A, separated_flux_B = sep_array[:, 0], sep_array[:, 1], sep_array[:, 2]
        template_flux_A, template_flux_B = sep_array[:, 3], sep_array[:, 4]
        time_values_A, RV_A = rvA_array[:, 0], rvA_array[:, 1]
        time_values_B, RV_B, RV_B_flags = rvB_array[:, 0], rvB_array[:, 1], rvB_array[:, 2]
        RV_A_initial, RV_B_initial = rvs_initial[:, 0], rvs_initial[:, 1]

        interval_result = IntervalResult(
            time_values_A, time_values_B, wavelength, separated_flux_A, separated_flux_B, template_flux_A,
            template_flux_B, RV_A, RV_B, RV_B_flags, RV_A_initial, RV_B_initial, velA_array, bfA_array,
            bfA_smooth_array, modelA_array, velB_array, bfB_array, bfB_smooth_array, modelB_array
        )
        routine_results.append_interval(interval_result)
    return routine_results


def plot_rv_and_separated_spectra(evaluation_data: RoutineResults, period: float, block=True):
    matplotlib.rcParams.update({'font.size': 20})
    for i in range(0, len(evaluation_data.interval_results)):
        fig = plt.figure(figsize=(16, 9))
        gspec = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gspec[0, :])
        ax2 = fig.add_subplot(gspec[1, 0])
        ax3 = fig.add_subplot(gspec[1, 1])

        phase_A = np.mod(evaluation_data.time_values_A[i], period) / period
        phase_B = np.mod(evaluation_data.time_values_B[i], period) / period

        flag_mask = evaluation_data.RV_B_flags[i].astype(bool)

        ax1.plot(phase_A, evaluation_data.RV_A[i], 'b*')
        ax1.plot(phase_B[flag_mask], evaluation_data.RV_B[i][flag_mask], 'r*')
        ax1.plot(phase_B[~flag_mask], evaluation_data.RV_B[i][~flag_mask], 'rx')
        ax1.set_xlabel('Orbital Phase')
        ax1.set_ylabel('Radial Velocity - system velocity (km/s)', fontsize=15)

        ax2.plot(evaluation_data.wavelengths[i], 1-evaluation_data.separated_spectra_A[i], 'b', linewidth=2)
        ax2.plot(evaluation_data.wavelengths[i], 1-evaluation_data.template_flux_A[i], '--', color='grey', linewidth=0.5)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Normalized Separated Flux', fontsize=15)

        ax3.plot(evaluation_data.wavelengths[i], 1-evaluation_data.separated_spectra_B[i], 'r', linewidth=2)
        ax3.plot(evaluation_data.wavelengths[i], 1-evaluation_data.template_flux_B[i], '--', color='grey', linewidth=0.5)
        ax2.set_xlabel('Wavelength (Å)')
        ax3.set_xlabel('Wavelength (Å)')
        fig.suptitle(f'Interval results {evaluation_data.wavelength_a[i]}-{evaluation_data.wavelength_b[i]} Å ')
        plt.tight_layout()

    plt.show(block=block)


def plot_smoothed_broadening_functions(evaluation_data: RoutineResults, block=True):
    matplotlib.rcParams.update({'font.size': 25})
    for i in range(0, len(evaluation_data.interval_results)):
        fig = plt.figure(figsize=(16, 9))
        gspec = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gspec[0, 0])
        ax2 = fig.add_subplot(gspec[0, 1])

        bf_results = evaluation_data.bf_results[i]
        vel_A, bf_smooth_A, models_A = bf_results[0], bf_results[2], bf_results[3]
        vel_B, bf_smooth_B, models_B = bf_results[4], bf_results[6], bf_results[7]
        RV_A = evaluation_data.RV_A[i]
        RV_B = evaluation_data.RV_B[i]
        flag_mask = evaluation_data.RV_B_flags[i].astype(bool)
        RV_B_initial = evaluation_data.RV_B_initial[i]

        ax1.set_xlim([np.min(vel_A), np.max(vel_A)])
        ax2.set_xlim([np.min(vel_A), np.max(vel_A)])

        ax1.plot(np.zeros(shape=(2,)), [0.0, 1.05], '--', color='grey')
        ax2.plot(np.zeros(shape=(2,)), [0.0, 1.05], '--', color='grey')
        for k in range(0, vel_A[:, 0].size):
            offset = k/vel_A[:, 0].size
            scale = (0.5/vel_A[:, 0].size)/np.max(bf_smooth_A[k, :])
            ax1.plot(vel_A[k, :], 1 + scale*bf_smooth_A[k, :] - offset)
            ax1.plot(vel_A[k, :], 1 + scale*models_A[k, :] - offset, 'k--')
            ax1.plot(np.ones(shape=(2,))*RV_A[k], [1-offset*1.01, 1+5/4 * scale*np.max(models_A[k, :])-offset],
                     color='blue')

            scale = (0.5/vel_B[:, 0].size)/np.max(bf_smooth_B[k, :])
            ax2.plot(vel_B[k, :], 1 + scale*bf_smooth_B[k, :] - offset)
            ax2.plot(vel_B[k, :], 1 + scale*models_B[k, :] - offset, 'k--')
            if ~flag_mask[k]:
                ax2.plot(np.ones(shape=(2,))*RV_B[k], [1-offset*1.01, 1+5/4 * scale*np.max(models_B[k, :])-offset],
                         color='red')
            else:
                ax2.plot(np.ones(shape=(2,)) * RV_B[k], [1-offset*1.01, 1+5/4 * scale*np.max(models_B[k, :])-offset],
                         color='blue')
            ax2.plot(np.ones(shape=(2,)) * RV_B_initial[k], [1-offset*1.01, 1+5/4*scale*np.max(models_B[k, :])-offset],
                     color='grey')
            ax1.set_ylabel('Normalized, smoothed Broadening Function')
            ax1.set_xlabel('Velocity Shift (km/s)')
            ax2.set_xlabel('Velocity Shift (km/s)')
            fig.suptitle(f'Interval results {evaluation_data.wavelength_a[i]}-{evaluation_data.wavelength_b[i]} Å ')
            ax1.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                right=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            ax2.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                right=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            plt.tight_layout()
    plt.show(block=block)


def compare_interval_results(evaluation_data: RoutineResults):
    pass


def compare_separated_spectra_with(evaluation_data: RoutineResults):
    pass


