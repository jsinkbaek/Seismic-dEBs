import numpy as np


# # # Convenience functions # # #
def split_kepler_residual_lightcurve(lc_model, period, nblocks):
    time = lc_model[:, 0]
    model = lc_model[:, 4]
    phase = np.mod(time, period) / period

    indices = np.argwhere((np.diff(phase) > 0.2) | (np.diff(phase) < -0.002))[:, 0] + 1
    indices_secondary = np.argwhere((np.diff(phase) > 0.2) | ((np.diff(phase) < -0.002) & (phase[:-1] < 0.3)))[:, 0] + 1
    indices_primary = np.argwhere(((np.diff(phase) < -0.002) & (phase[:-1] > 0.3)))[:, 0] + 1
    sub_arrays_secondary = []
    sub_arrays_primary = []
    diff_prim = []
    diff_sec = []
    len_prim = []
    len_sec = []

    for i in range(0, len(indices)):
        if i == 0:
            start = 0
            end = indices[i]
        else:
            start = indices[i - 1]
            end = indices[i]
        if indices[i] in indices_secondary:
            sub_arrays_secondary.append(
                np.array([time[start:end], model[start:end]]).T
            )
            diff_sec.append(time[end - 1] - time[start])
            len_sec.append(time[start:end].size)

        elif indices[i] in indices_primary:
            sub_arrays_primary.append(
                np.array([time[start:end], model[start:end]]).T
            )
            diff_prim.append(time[end - 1] - time[start])
            len_prim.append(time[start:end].size)

        else:
            raise ValueError('index not in either')

    nblocks_sec = nblocks  # int(np.rint(secondary_time/block_length))
    nblocks_prim = nblocks  # int(np.rint(primary_time/block_length))

    sub_arrays_split = []

    # Concatenate all eclipses to one list of arrays
    blen_prim = []
    blen_sec = []
    for i in range(0, len(sub_arrays_primary)):
        blen_prim.append(sub_arrays_primary[i].shape[0] / nblocks)
        sub_arrays_split += np.array_split(sub_arrays_primary[i], nblocks_prim, axis=0)
    for i in range(0, len(sub_arrays_secondary)):
        blen_sec.append(sub_arrays_secondary[i].shape[0] / nblocks)
        sub_arrays_split += np.array_split(sub_arrays_secondary[i], nblocks_sec, axis=0)

    row_size = np.max([x.shape[0] for x in sub_arrays_split])
    lc_blocks = np.empty((row_size, 2, len(sub_arrays_split)))
    lc_blocks[:] = np.nan
    for i in range(0, len(sub_arrays_split)):
        current_array = sub_arrays_split[i]
        lc_blocks[0:current_array.shape[0], :, i] = current_array

    return lc_blocks


def split_tess_residual_lightcurve(lc_model, nblocks):
    model = lc_model[:, 4]
    time = lc_model[:, 0]
    lc = np.array([time, model]).T
    mask_secondary = time < np.mean(time)
    lc_block_secondary = lc[mask_secondary, :]
    lc_block_primary = lc[~mask_secondary, :]

    sub_arrays_secondary = np.array_split(lc_block_secondary, nblocks, axis=0)
    sub_arrays_primary = np.array_split(lc_block_primary, nblocks, axis=0)

    sub_arrays_split = sub_arrays_secondary + sub_arrays_primary

    row_size = np.max([x.shape[0] for x in sub_arrays_split])
    lc_blocks = np.empty((row_size, 2, len(sub_arrays_split)))
    lc_blocks[:] = np.nan
    for i in range(0, len(sub_arrays_split)):
        current_array = sub_arrays_split[i]
        lc_blocks[0:current_array.shape[0], :, i] = current_array

    return lc_blocks