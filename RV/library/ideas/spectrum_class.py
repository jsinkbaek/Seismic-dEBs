import numpy as np


class Spectrum:
    def __init__(self, wavelength: np.ndarray, inverted_flux: np.ndarray, time_value: float, RV_A=None, error_RV_A=None,
                 RV_B=None, error_RV_B=None):
        self.wavelength = wavelength
        self.inverted_flux = inverted_flux
        self.time_value = time_value
        self.RV_A = RV_A
        self.RV_B = RV_B
        self.error_A = error_RV_A
        self.error_B = error_RV_B


class SpectrumCollection:
    def __init__(self, *args: tuple or Spectrum or np.ndarray):
        if len(args) == 0:
            self.wavelength = None
            self.spectrum_array = np.array([], dtype=Spectrum)
            self.inverted_flux_array = np.array([])
            self.time_values_array = np.array([])
        else:
            restructured_input = self._restructure_input(*args)
            self.spectrum_array, self.wavelength, self.inverted_flux_array, self.time_values_array = restructured_input


    @staticmethod
    def _restructure_input(*args):
        spectrum_array = np.empty((len(args),), dtype=Spectrum)
        if all(isinstance(arg, tuple) for arg in args):
            for i, arg in enumerate(args):
                wavelength, inverted_flux, time_value = arg[0], arg[1], arg[2]
                if len(arg) != 3:
                    raise ValueError('Unknown number of arguments to spectrum tuple')
                spectrum = Spectrum(wavelength, inverted_flux, time_value)
                spectrum_array[i] = spectrum
                if i == 0:
                    inverted_flux_array = np.empty((inverted_flux.size, spectrum_array.size))
                    time_values_array = np.empty((spectrum_array.size,))
                inverted_flux_array[:, i] = inverted_flux
                time_values_array[i] = time_value

        elif all(isinstance(arg, np.ndarray) for arg in args):
            wavelength = args[0]
            inverted_flux_array = args[1]
            time_values_array = args[2]
            for i in range(0, len(inverted_flux_array[0, :])):
                spectrum_array[i] = Spectrum(wavelength, inverted_flux_array[:, i], time_values_array[i])

        elif all(isinstance(arg, Spectrum) for arg in args):
            wavelength = args[0].wavelength
            inverted_flux_array = np.empty((wavelength.size, len(args)))
            time_values_array = np.empty((len(args),))
            for i, arg in enumerate(args):
                spectrum_array[i] = arg
                inverted_flux_array[:, i] = arg.inverted_flux
                time_values_array[i] = arg.time_value

        else:
            raise ValueError('Wrong input. All args must have the same type.')

        return spectrum_array, wavelength, inverted_flux_array, time_values_array






