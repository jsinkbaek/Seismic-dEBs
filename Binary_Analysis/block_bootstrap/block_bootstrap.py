import numpy as np
from numpy.random import default_rng
from Binary_Analysis.block_bootstrap.jktebop_io_interface import run_jktebop_on_sample, clean_work_folder
from joblib import Parallel, delayed
from typing import List, Tuple


def draw_sample(
        lc_blocks: np.ndarray, rvA, rvB, block_midtime: List[np.ndarray] or np.ndarray, rvA_model, rvB_model,
        draw_random_rv_obs=True
):
    """

    :param lc_blocks: required shape (:, 3, nblocks). 1st column must be time values, 2nd lc flux magnitude,
                        3rd error on mag.
    :param rvA:       required shape (:, 3). 1st column time values, 2nd rv values, 3rd error on rv
    :param rvB:
    :param block_midtime:
    :param rvA_model:
    :param rvB_model:
    :return:
    """
    lc_sample = draw_lc_sample(lc_blocks, block_midtime)
    rvA_sample, rvB_sample = draw_rv_sample(rvA, rvB, rvA_model, rvB_model, draw_random_timestamps=draw_random_rv_obs)
    return lc_sample, rvA_sample, rvB_sample


def draw_lc_sample(lc_blocks: np.ndarray, block_midtime: List[np.ndarray] or np.ndarray):
    rng = default_rng()
    drawn_blocks = np.copy(lc_blocks)

    # Draw random mid_times to assign to each block
    if block_midtime is None:
        pass
    elif isinstance(block_midtime, list):
        for i in range(0, len(block_midtime)):
            if i == 0:
                start, rel_end = 0, block_midtime[i].size
            else:
                start = block_midtime[i - 1].size
                rel_end = block_midtime[i].size
            midtime_indices = rng.integers(low=0, high=rel_end, size=rel_end)
            for k in range(0, rel_end):
                relative_time = drawn_blocks[:, 0, k + start] - block_midtime[i][k]
                new_midtime = block_midtime[i][midtime_indices[k]]
                drawn_blocks[:, 0, k + start] = relative_time + new_midtime
    else:
        midtime_indices = rng.integers(low=0, high=drawn_blocks[0, 0, :].size, size=drawn_blocks[0, 0, :].size)
        for k in range(0, drawn_blocks[0, 0, :].size):
            relative_time = drawn_blocks[:, 0, k] - block_midtime[k]
            new_midtime = block_midtime[midtime_indices[k]]
            drawn_blocks[:, 0, k] = relative_time + new_midtime

    # Draw random blocks to add to light curve sample
    block_indices = rng.integers(low=0, high=lc_blocks[0, 0, :].size, size=lc_blocks[0, 0, :].size)
    drawn_blocks = drawn_blocks[:, :, block_indices]

    # Concatenate the blocks from axis=2 to produce a single light curve (shape (nblocks*blocksize, 3)) from the draw
    lc_sample_list = []
    for i in range(0, drawn_blocks[0, 0, :].size):
        current_block = drawn_blocks[:, :, i]
        nan_mask = np.isnan(current_block[:, 0])
        lc_sample_list.append(current_block[~nan_mask, :])

    lc_sample = np.concatenate(lc_sample_list, axis=0)
    return lc_sample


def draw_rv_sample(rvA, rvB, rvA_model, rvB_model, draw_random_timestamps=True):
    rng = default_rng()

    # Draw random rvs for sample
    if draw_random_timestamps is True:
        rvA_draw_indices = rng.integers(low=0, high=rvA.shape[0], size=rvA.shape[0])
        rvA_sample = rvA[rvA_draw_indices, :]

        rvB_draw_indices = rng.integers(low=0, high=rvB.shape[0], size=rvB.shape[0])
        rvB_sample = rvB[rvB_draw_indices, :]
    else:
        rvA_sample = rvA
        rvB_sample = rvB
        rvA_draw_indices = np.indices((rvA[:, 0].size, ))
        rvB_draw_indices = np.indices((rvB[:, 0].size, ))

    # Generate synthetic rv data from model + random residual
    if rvA_model is not None:
        residual_A = rvA[:, 1] - rvA_model
        residual_B = rvB[:, 1] - rvB_model
        # Draw
        residual_indices_A = rng.integers(low=0, high=rvA.shape[0], size=rvA.shape[0])
        residual_indices_B = rng.integers(low=0, high=rvB.shape[0], size=rvB.shape[0])
        # Value
        rvA_sample[:, 1] = rvA_model[rvA_draw_indices] + residual_A[residual_indices_A]
        rvB_sample[:, 1] = rvB_model[rvB_draw_indices] + residual_B[residual_indices_B]
        # Error
        rvA_sample[:, 2] = rvA[residual_indices_A, 2]
        rvB_sample[:, 2] = rvB[residual_indices_B, 2]
    return rvA_sample, rvB_sample


def draw_residual_sample(
        model_blocks, residuals, error_residuals, rvA, rvB, rvA_model, rvB_model, draw_random_rv_obs=False
):
    lc_sample = draw_residual_lc_sample(model_blocks, residuals, error_residuals)
    rvA_sample, rvB_sample = draw_rv_sample(rvA, rvB, rvA_model, rvB_model, draw_random_timestamps=draw_random_rv_obs)
    return lc_sample, rvA_sample, rvB_sample


def draw_residual_lc_sample(model_blocks, residuals, error_residuals):
    """

    :param model_blocks: np.ndarray shape (:, 2, nblocks) (time, model) in second axis
    :param residuals:       np.ndarray shape (ndatapoints)
    :param error_residuals: np.ndarray shape (ndatapoints)
    :return:
    """
    model_blocks = np.copy(model_blocks)
    # Make room for errors
    model_blocks = np.append(
        model_blocks, np.zeros((model_blocks[:, 0, 0].size, 1, model_blocks[0, 0, :].size)), axis=1
    )

    rng = default_rng()
    lc_sample_list = []
    for i in range(0, model_blocks[0, 0, :].size):
        current_block = model_blocks[:, :, i]
        nan_mask = np.isnan(current_block[:, 0])
        current_block = current_block[~nan_mask, :]

        index_0 = rng.integers(low=0, high=residuals.size-1, size=1)[0]
        indices = range(index_0, index_0+current_block[:, 0].size)
        current_block[:, 1] = current_block[:, 1] + np.take(residuals, indices, mode='wrap')
        current_block[:, 2] = np.take(error_residuals, indices, mode='wrap')

        lc_sample_list.append(current_block)

    lc_sample = np.concatenate(lc_sample_list, axis=0)
    return lc_sample


def _divide_into_subgroups(lc_blocks: np.ndarray, subgroup_divisions: Tuple[int]):
    # Divide each block into possible subgroups to draw from
    subgroups_all_lightcurves = []
    for k in range(0, lc_blocks[0, 0, :].size):
        current_block = lc_blocks[~np.isnan(lc_blocks[:, 0, k]), :, k]
        subgroups = []
        for i in range(0, len(subgroup_divisions)):
            block_size = current_block[:, 0].size
            sub_block_size = block_size / subgroup_divisions[i]
            sub_block_size = int(np.floor(sub_block_size))
            sub_blocks = np.empty((sub_block_size, 3, block_size - sub_block_size + 1))
            for j in range(0, block_size - sub_block_size + 1):
                start = j
                end = sub_block_size + j
                sub_blocks[:, :, j] = current_block[start:end, :]
            subgroups.append(sub_blocks)
        subgroups_all_lightcurves.append(subgroups)
    return subgroups_all_lightcurves


def draw_lc_sample_variable_moving_blocks(
        lc_blocks: np.ndarray, subgroups_all_lightcurves: list, period: float, subgroup_divisions: Tuple[int]
):
    rng = default_rng()

    # Draw amount of each subgroup (amount of each subgroup is n_draws * subgroup_division to preserve approximate
    # data size)
    share = rng.integers(0, len(subgroup_divisions), lc_blocks[0, 0, :].size)
    bincount = np.bincount(share, minlength=len(subgroup_divisions))
    draw_amount_groups = bincount * subgroup_divisions

    # Draw sub blocks from lightcurve blocks
    drawn_sub_blocks = []
    for i in range(0, len(subgroup_divisions)):
        share_lcs = rng.integers(0, len(subgroups_all_lightcurves), draw_amount_groups[i])
        draw_amount_lcs = np.bincount(share_lcs, minlength=len(subgroups_all_lightcurves))
        for k in range(0, len(subgroups_all_lightcurves)):
            if draw_amount_lcs[k] != 0:
                current_sub_blocks = np.copy(subgroups_all_lightcurves[k][i])
                sub_block_indices = rng.integers(0, current_sub_blocks[0, 0, :].size, draw_amount_lcs[k])
                drawn_sub_blocks.append(current_sub_blocks[:, :, sub_block_indices])

    # Draw random timestamps to give drawn sub blocks (while retaining current phase)
    for i in range(0, len(drawn_sub_blocks)):
        new_time_indices = rng.integers(0, lc_blocks[0, 0, :].size, drawn_sub_blocks[i][0, 0, :].size)
        time_startpoints = lc_blocks[0, 0, new_time_indices]
        phase_startpoints = np.mod(time_startpoints, period)/period
        for k in range(0, drawn_sub_blocks[i][0, 0, :].size):
            phase = np.mod(drawn_sub_blocks[i][:, 0, k], period)/period
            phase_diff = phase - phase_startpoints[k]
            new_time = time_startpoints[k] + phase_diff*period
            drawn_sub_blocks[i][:, 0, k] = new_time

    # Combine drawn sub-blocks to light curve sample
    lc_sample_list = []
    for i in range(0, len(drawn_sub_blocks)):
        for k in range(0, drawn_sub_blocks[i][0, 0, :].size):
            lc_sample_list.append(drawn_sub_blocks[i][:, :, k])
    lc_sample = np.concatenate(lc_sample_list, axis=0)

    return lc_sample


def block_bootstrap(
        lc_blocks, rvA, rvB, repetitions, parameter_names, n_jobs=4,
        block_midtime: List[np.ndarray] or np.ndarray = None, rvA_model=None, rvB_model=None,
        infile_name='infile.default', draw_random_rv_obs=True
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

    :param rvA_model:
        Model values for best rvA fit. Required shape (:, ), with same length as rvA[:, 0]. Since the amount of RV data
        points is much smaller than for the light curves, blocks make no reasonable sense. If this parameter and
        rvB_model is None, random RV data-points will simply be drawn for each sample. If they are provided, residual
        permutation will instead be performed, where synthetic RV data is generated from the model + a random residual
        from the component dataset. It will be provided with the same error associated to the residual data point. The
        drawn sample will still be randomly selected from the time-values available.

    :param rvB_model:

    :return:
    """
    clean_work_folder()
    job_results = Parallel(n_jobs=n_jobs)(
        delayed(_loop_function)(
            lc_blocks, rvA, rvB, parameter_names, block_midtime, rvA_model, rvB_model, infile_name, draw_random_rv_obs,
            i
        )
        for i in range(0, repetitions)
    )
    return evaluate_runs(job_results)


def residual_block_bootstrap(
        model_blocks, residuals, error_residuals, rvA, rvB,
        repetitions, parameter_names, n_jobs=4,
        rvA_model=None, rvB_model=None, draw_random_rv_obs=False,
        infile_name='infile.default'
):
    clean_work_folder()
    job_results = Parallel(n_jobs=n_jobs)(
        delayed(_loop_function_residual)(
            model_blocks, residuals, error_residuals, rvA, rvB, parameter_names, rvA_model, rvB_model,
            infile_name, draw_random_rv_obs,
            i
        )
        for i in range(0, repetitions)
    )
    return evaluate_runs(job_results)


def block_bootstrap_variable_moving_blocks(
        lc_blocks, rvA, rvB, repetitions, parameter_names,
        subgroup_divisions: Tuple[int, ...], period: float,
        n_jobs=4,
        rvA_model=None, rvB_model=None,
        infile_name='infile.default'
):
    clean_work_folder()
    subgroups_all_lightcurves = _divide_into_subgroups(lc_blocks, subgroup_divisions)
    job_results = Parallel(n_jobs=n_jobs)(
        delayed(_loop_function_variable_moving_blocks)(
            lc_blocks, subgroups_all_lightcurves, period, subgroup_divisions, rvA, rvB, parameter_names, rvA_model,
            rvB_model, infile_name, i) for i in range(0, repetitions)
        )
    return evaluate_runs(job_results)


def _loop_function(
        lc_blocks, rvA, rvB, parameter_names, block_midtime, rvA_model, rvB_model, infile_name, draw_random_rv_obs,
        index
):
    lc_sample, rvA_sample, rvB_sample = draw_sample(
        lc_blocks, rvA, rvB, block_midtime, rvA_model, rvB_model, draw_random_rv_obs
    )
    parameter_values = run_jktebop_on_sample(
        lc_sample, rvA_sample, rvB_sample, index, parameter_names, infile_name=infile_name
    )
    return parameter_values


def _loop_function_residual(
        model_blocks, residuals, error_residuals, rvA, rvB, parameter_names, rvA_model, rvB_model, infile_name,
        draw_random_rv_obs, index
):
    lc_sample, rvA_sample, rvB_sample = draw_residual_sample(
        model_blocks, residuals, error_residuals, rvA, rvB, rvA_model, rvB_model, draw_random_rv_obs
    )
    parameter_values = run_jktebop_on_sample(
        lc_sample, rvA_sample, rvB_sample, index, parameter_names, infile_name=infile_name
    )
    return parameter_values


def _loop_function_variable_moving_blocks(
        lc_blocks, subgroups_all_lightcurves, period, subgroup_divisions, rvA, rvB, parameter_names,
        rvA_model, rvB_model, infile_name, index
):
    lc_sample = draw_lc_sample_variable_moving_blocks(
        lc_blocks, subgroups_all_lightcurves, period, subgroup_divisions
    )
    rvA_sample, rvB_sample = draw_rv_sample(rvA, rvB, rvA_model, rvB_model)
    parameter_values = run_jktebop_on_sample(lc_sample, rvA_sample, rvB_sample, index, parameter_names,
                                             infile_name=infile_name)
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

