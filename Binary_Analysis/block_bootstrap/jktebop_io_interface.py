import numpy as np
from typing import List
import os
from shutil import copy2
import subprocess
from shutil import rmtree

outfile_row = {'sb_ratio': '1  Surf. bright. ratio',
               'sum_radii': '2  Sum of frac radii',
               'ratio_radii': '3  Ratio of the radii',
               'limbd_A1': '4  Limb darkening A1',
               'limbd_B1': '5  Limb darkening B1',
               'incl': '6  Orbit inclination',
               'ecc': '7  Eccentricity',
               'perilong': '8  Periastronlongitude',
               'grav_dark_A': '9  Grav darkening A',
               'grav_dark_B': '10  Grav darkening B',
               'refl_light_A': '10  Grav darkening B',
               'refl_light_B': '12  Reflected light B',
               'phot_mass_ratio': '13  Phot mass ratio',
               '3_light': '15  Third light (L_3)',
               'phase_corr': '16  Phase correction',
               'light_scale_factor': '17  Light scale factor',
               'integration_ring': '18  Integration ring',
               'period': '19  Orbital period (P)',
               'ephemeris_tbase': '20  Ephemeris timebase',
               'limbd_A2': '21  Limb darkening A2',
               'limbd_B2': '24  Limb darkening B2',
               'rv_amp_A': '27  RV amplitude star A',
               'rv_amp_B': '28  RV amplitude star B',
               'system_rv_A': '29  Systemic RV star A',
               'system_rv_B': '30  Systemic RV star B',
               'sma_rsun': 'Orbital semimajor axis (Rsun):',
               'mass_A': 'Mass of star A (Msun)',
               'mass_B': 'Mass of star B (Msun)',
               'radius_A': 'Radius of star A (Rsun)',
               'radius_B': 'Radius of star B (Rsun)',
               'logg_A': 'Log surface gravity of star A (cgs):',
               'logg_B': 'Log surface gravity of star B (cgs):'}

outfile_col = {'sb_ratio': 4, 'sum_radii': 5, 'ratio_radii': 5, 'limbd_A1': 4, 'limbd_B1': 4, 'incl': 3,
               'ecc': 2, 'perilong': 2, 'grav_dark_A': 3, 'grav_dark_B': 3, 'refl_light_A': 4, 'refl_light_B': 4,
               'phot_mass_ratio': 4, '3_light': 4, 'phase_corr': 3, 'light_scale_factor': 4, 'integration_ring': 3,
               'period': 4, 'ephemeris_tbase': 3, 'limbd_A2': 4, 'limbd_B2': 4, 'rv_amp_A': 5, 'rv_amp_B': 5,
               'system_rv_A': 5, 'system_rv_B': 5, 'sma_rsun': 4, 'mass_A': 5, 'mass_B': 5, 'radius_A': 5,
               'radius_B': 5, 'logg_A': 7, 'logg_B': 7}


def pull_parameters_from_outfile(folder: str, parameter_names: List[str]):
    parameter_values = np.empty((len(parameter_names), ))
    rows = [outfile_row[x] for x in parameter_names]
    cols = [outfile_col[x] for x in parameter_names]
    with open(folder+'param.out', 'r') as f:
        list_of_lines = f.readlines()
        for line in list_of_lines:
            for i in range(0, len(rows)):
                if rows[i] in line:
                    split = line.split()
                    try:
                        parameter_values[i] = float(split[cols[i]])
                    except ValueError as ve:
                        print(ve)
                        print('folder:', folder)
                        print('Column index:', cols[i])
                        print('Current parameter:', rows[i])
                        return None
    return parameter_values


def setup_sample_folder(
        light_curve, rvA, rvB, index,
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
    np.savetxt(f'work/{index}/lc.sample', light_curve)
    np.savetxt(f'work/{index}/rvA.sample', rvA)
    np.savetxt(f'work/{index}/rvB.sample', rvB)


def run_jktebop_on_sample(
        light_curve, rvA, rvB,
        sample_index,
        parameter_names: List[str],
        infile_name='infile.default', makefile_name='Makefile.default'
):
    """

    :param light_curve: required shape (:, 3). 1st column must be time values, 2nd lc flux magnitude,
                        3rd error on mag.
    :param rvA:         required shape (:, 3). 1st column time values, 2nd rv values, 3rd error on rv
    :param rvB:
    :param sample_index:
    :param parameter_names:
    :param infile_name:
    :param makefile_name:
    :return:
    """

    setup_sample_folder(light_curve, rvA, rvB, sample_index, infile_name, makefile_name)
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

