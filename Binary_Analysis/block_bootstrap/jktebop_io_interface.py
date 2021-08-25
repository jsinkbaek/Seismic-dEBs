import numpy as np
from typing import List
import os
from shutil import copy2
import subprocess
from shutil import rmtree

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
                index = rows.index(i)
                split = line.split()
                parameter_values[index] = float(split[cols[index]])
    return parameter_values


def setup_sample_folder(
        time_lc, lc_mag, error_lc, time_rvA, time_rvB, rvA, rvB, error_rvA, error_rvB, index,
        infile_name='infile.default', makefile_name='Makefile.default'
):
    """
    Sets up a JKTEBOP folder in the work directory with a drawn sample of light curve data and RVs.
    Requires that the current directory is the same as this python file.
    """
    if 'work' not in os.listdir():
        raise IOError('work folder not in current directory.')
    os.mkdir(f'work/{index}')
    copy2('work/'+infile_name, f'work/{index}/'+'infile.sample')
    copy2('work/'+makefile_name, f'work/{index}/'+'Makefile')
    lc_array = np.array([time_lc, lc_mag, error_lc]).T
    rvA_array = np.array([time_rvA, rvA, error_rvA]).T
    rvB_array = np.array([time_rvB, rvB, error_rvB]).T
    np.savetxt(f'work/{index}/lc.sample', lc_array)
    np.savetxt(f'work/{index}/rvA.sample', rvA_array)
    np.savetxt(f'work/{index}/rvB.sample', rvB_array)


def run_jktebop_on_sample(
        time_lc, lc_mag, error_lc, time_rvA, time_rvB, rvA, rvB, error_rvA, error_rvB, sample_index,
        parameter_names: List[str],
        infile_name='infile.default', makefile_name='Makefile.default'
):
    setup_sample_folder(
        time_lc, lc_mag, error_lc, time_rvA, time_rvB, rvA, rvB, error_rvA, error_rvB, sample_index,
        infile_name, makefile_name
    )
    subprocess.run('make', cwd=f'work/{sample_index}/', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    parameter_values = pull_parameters_from_outfile(f'work/{sample_index}/', parameter_names)
    return parameter_values


def clean_work_folder():
    if 'work' not in os.listdir():
        raise IOError('work folder not in current directory.')
    entries = os.listdir('work/')
    for entry in entries:
        if os.path.isdir('work/'+entry):
            try:
                int(entry)
                rmtree('work/'+entry)
            except ValueError:
                pass

