import numpy as np
from numpy.random import default_rng
from Binary_Analysis.block_bootstrap.jktebop_io_interface import run_jktebop_on_sample, clean_work_folder
from joblib import Parallel, delayed


def draw_sample(lc_blocks: np.ndarray, rvA, rvB):
    """

    :param lc_blocks: required shape (:, 3, nblocks). 1st column must be time values, 2nd lc flux magnitude,
                        3rd error on mag.
    :param rvA:       required shape (:, 3). 1st column time values, 2nd rv values, 3rd error on rv
    :param rvB:
    :return:
    """
    rng = default_rng()
    block_indices = rng.integers(low=0, high=lc_blocks[0, 0, :].size, size=lc_blocks[0, 0, :].size)
    drawn_blocks = lc_blocks[:, :, block_indices]

    # Concatenate the blocks from axis=2 to produce a single light curve (shape (nblocks*blocksize, 3)) from the draw
    lc_sample_list = []
    for i in range(0, drawn_blocks[0, 0, :].size):
        current_block = drawn_blocks[:, :, i]
        nan_mask = np.isnan(current_block[:, 0])
        lc_sample_list.append(current_block[~nan_mask, :])

    lc_sample = np.concatenate(lc_sample_list, axis=0)

    rvA_draw_indices = rng.integers(low=0, high=rvA.shape[0], size=rvA.shape[0])
    rvA_sample = rvA[rvA_draw_indices, :]

    rvB_draw_indices = rng.integers(low=0, high=rvB.shape[0], size=rvB.shape[0])
    rvB_sample = rvB[rvB_draw_indices, :]

    return lc_sample, rvA_sample, rvB_sample


def block_bootstrap(lc_blocks, rvA, rvB, repetitions, parameter_names, n_jobs=4):
    """
    :param lc_blocks: required shape (:, 3, nblocks). 1st column must be time values, 2nd lc flux magnitude,
                        3rd error on mag.
    :param rvA:       required shape (:, 3). 1st column time values, 2nd rv values, 3rd error on rv
    :param rvB:
    :param repetitions:
    :param parameter_names:
    :param n_jobs:
    :return:
    """
    clean_work_folder()
    job_results = Parallel(n_jobs=n_jobs)(
        delayed(_loop_function)(lc_blocks, rvA, rvB, parameter_names, i) for i in range(0, repetitions)
    )
    return evaluate_runs(job_results)


def _loop_function(lc_blocks, rvA, rvB, parameter_names, index):
    lc_sample, rvA_sample, rvB_sample = draw_sample(lc_blocks, rvA, rvB)
    parameter_values = run_jktebop_on_sample(lc_sample, rvA, rvB, index, parameter_names)
    return parameter_values


def evaluate_runs(job_results):
    vals = np.empty((len(job_results), len(job_results[0])))
    for i in range(0, len(job_results)):
        vals[i, :] = job_results[i]

    params_mean = np.mean(vals, axis=0)
    params_std = np.std(vals, axis=0)
    return params_mean, params_std

