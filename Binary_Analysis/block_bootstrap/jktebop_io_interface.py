import numpy as np
from typing import List
import os

outfile_row = {'sb_ratio': 184, 'sum_radii': 185, 'ratio_radii': 186, 'limbd_A1': 187, 'limbd_B1': 188, 'incl': 189,
               'ecc': 190, 'perilong': 191, 'grav_dark_A': 192, 'grav_dark_B': 193, 'refl_light_A': 194,
               'refl_light_B': 195,'phot_mass_ratio': 196, '3_light': 197, 'phase_corr': 198, 'light_scale_factor': 199,
               'integration_ring': 200, 'period': 201, 'ephemeris_tbase': 202, 'limbd_A2': 203, 'limbd_B2': 204,
               'rv_amp_A': 205, 'rv_amp_B': 206, 'system_rv_A': 207, 'system_rv_B': 208}
outfile_col = {'sb_ratio': 4, 'sum_radii': 5, 'ratio_radii': 5, 'limbd_A1': 4, 'limbd_B1': 4, 'incl': 3,
               'ecc': 2, 'perilong': 2, 'grav_dark_A': 3, 'grav_dark_B': 3, 'refl_light_A': 4, 'refl_light_B': 4,
               'phot_mass_ratio': 4, '3_light': 4, 'phase_corr': 3, 'light_scale_factor': 4, 'integration_ring': 3,
               'period': 4, 'ephemeris_tbase': 3, 'limbd_A2': 4, 'limbd_B2': 4, 'rv_amp_A': 4, 'rv_amp_B': 4,
               'system_rv_A': 4, 'system_rv_B': 4}


def pull_parameters_from_outfile(folder: str, parameter_names: List[str]):
    parameter_values = np.empty((len(parameter_names), ))
    rows = [outfile_row[x] for x in parameter_names]
    cols = [outfile_col[x] for x in parameter_names]
    with open(folder+'param.out', 'r') as f:
        list_of_lines = f.readlines()
        for i, line in enumerate(list_of_lines):
            if i in rows:
                split = line.split()
                parameter_values[i] = float(split[cols[i]])
    return parameter_values


def setup_sample_folder(time_lc, lc_mag, error_lc, time_rvA, time_rvB, rvA, rvB, error_rvA, error_rvB, index):
    """
    Sets up the JKTEBOP folder in the work directory with a drawn sample of light curve data and RVs.
    Requires that the current working directory is the same as this python file.
    """

