import numpy as np
from numpy.random import default_rng
from Binary_Analysis.block_bootstrap.jktebop_io_interface import run_jktebop_on_sample, clean_work_folder
from joblib import Parallel, delayed
from typing import List


def draw_sample(lc_blocks: np.ndarray, rvA, rvB, block_midtime: List[np.ndarray] or np.ndarray):
    """

    :param lc_blocks: required shape (:, 3, nblocks). 1st column must be time values, 2nd lc flux magnitude,
                        3rd error on mag.
    :param rvA:       required shape (:, 3). 1st column time values, 2nd rv values, 3rd error on rv
    :param rvB:
    :param block_midtime:
    :return:
    """
    rng = default_rng()
    block_indices = rng.integers(low=0, high=lc_blocks[0, 0, :].size, size=lc_blocks[0, 0, :].size)
    drawn_blocks = lc_blocks[:, :, block_indices]
    if block_midtime is None:
        pass
    elif isinstance(block_midtime, list):
        for i in range(0, len(block_midtime)):
            if i==0:
                start, rel_end = 0, block_midtime[i].size
            else:
                start = block_midtime[i-1].size
                rel_end = block_midtime[i].size
            midtime_indices = rng.integers(low=0, high=rel_end, size=rel_end)
            for k in range(0, rel_end):
                relative_time = drawn_blocks[:, 0, k+start] - block_midtime[i][k]
                new_midtime = block_midtime[i][midtime_indices[k]]
                drawn_blocks[:, 0, k+start] = relative_time + new_midtime
    else:
        midtime_indices = rng.integers(low=0, high=drawn_blocks[0, 0, :].size, size=drawn_blocks[0, 0, :].size)
        for k in range(0, drawn_blocks[0, 0, :].size):
            relative_time = drawn_blocks[:, 0, k] - block_midtime[k]
            new_midtime = block_midtime[midtime_indices[k]]
            drawn_blocks[:, 0, k] = relative_time + new_midtime

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


def block_bootstrap(
        lc_blocks, rvA, rvB, repetitions, parameter_names, n_jobs=4,
        block_midtime: List[np.ndarray] or np.ndarray = None
):
    """
    :param lc_blocks:       required shape (:, 3, nblocks). 1st column must be time values, 2nd lc flux magnitude,
                                3rd error on mag.
    :param rvA:             required shape (:, 3). 1st column time values, 2nd rv values, 3rd error on rv
    :param rvB:
    :param repetitions:
    :param parameter_names:
    :param n_jobs:
    :param block_midtime:

        midtime of each block which can be sampled from (to shuffle blocks). If a List of arrays, it will be assumed
        that there are multiple independent block types, which should not have their midtimes mixed. Here,
        lc_blocks[:, :, 0:block_midtime[0].size] will be assumed to be the first block type,
        lc_blocks[:, :, block_midtime[0].size:block_midtime[1].size] will be assumed to be the
        next, and so on. A random amount of each block-type will still be drawn, they are just restricted to midtimes
        of the same type. An example of where this is useful:
            When each block corresponds to 1 eclipse and not 1 period, the primary and secondary eclipse should
            obviously not be swapped around in time.

    :return:
    """
    clean_work_folder()
    job_results = Parallel(n_jobs=n_jobs)(
        delayed(_loop_function)(lc_blocks, rvA, rvB, parameter_names, block_midtime, i) for i in range(0, repetitions)
    )
    return evaluate_runs(job_results)


def _loop_function(lc_blocks, rvA, rvB, parameter_names, block_midtime, index):
    lc_sample, rvA_sample, rvB_sample = draw_sample(lc_blocks, rvA, rvB, block_midtime)
    parameter_values = run_jktebop_on_sample(lc_sample, rvA, rvB, index, parameter_names)
    return parameter_values


def evaluate_runs(job_results):
    vals = np.empty((len(job_results), len(job_results[0])))
    for i in range(0, len(job_results)):
        if job_results[i] is not None:
            vals[i, :] = job_results[i]
        else:
            vals[i, :] = np.nan

    params_mean = np.mean(vals[~np.isnan(vals[:, 0]), :], axis=0)
    params_std = np.std(vals[~np.isnan(vals[:, 0]), :], axis=0)
    return params_mean, params_std, vals

