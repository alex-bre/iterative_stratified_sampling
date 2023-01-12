import numpy as np
import pandas as pd
import warnings


def _match_groups_encoding(data, ref_data):
    """Match group encoding: if e.g. one group has elements 'A' and 'C', and the other group 'A', 'B', 'C', the
    numerical encoding will be adjusted so that group B is counted with 0 values in the one group.
    """
    data_dict = set(data).union(set(ref_data))
    data_dict = list(data_dict)
    data_dict.sort()
    data = pd.Series(data).replace(data_dict, np.arange(len(data_dict))).values
    ref_data = pd.Series(ref_data).replace(data_dict, np.arange(len(data_dict))).values
    return data, ref_data


def _get_groups_weights(data, ref_data):
    """Get the counts for each discrete group in the dataset and the resulting relative weights that should be applied to
    the samples in the original data to match the distribution in ref_data.
    """
    data = np.array(data, dtype=str)
    ref_data = np.array(ref_data, dtype=str)
    # data, ref_data = _match_groups_encoding(data, ref_data)
    data = np.array(data, dtype=int)
    if len(data.shape) > 1:
        raise ValueError(f"Wrong input: data must be 1d list or 1d ndarray. Passed: {data.shape}.")
    ref_data = np.array(ref_data, dtype=int)
    if len(ref_data.shape) > 1:
        raise ValueError(f"Wrong input: ref_data must be 1d list or 1d ndarray. Passed: {ref_data.shape}.")
    minlength = int(np.max([np.max(ref_data), np.max(data)]) + 1)
    counts = np.bincount(data, minlength=minlength)
    ref_counts = np.bincount(ref_data, minlength=minlength)
    weights = ref_counts / counts
    weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
    weights_per_sample = weights[data]
    return weights_per_sample, weights, counts, ref_counts


def _get_overlap_groups(data, ref_data):
    """Transform the elements of data to string. Concatenate the rows and join the string values to create new group
    labels.
    """
    data = np.array(data, dtype=str)
    ref_data = np.array(ref_data, dtype=str)
    data = list(map("_".join, zip(*data)))
    ref_data = list(map("_".join, zip(*ref_data)))
    data, ref_data = _match_groups_encoding(data, ref_data)
    return data, ref_data


def _iterative_stratified_sampling(data, ref_weights, frac=0.8, random_state=42):
    """Match the objective distribution by iteratively removing sample indices from the original data. The sampling is
    performed by calculating the group that has the highest difference when subtracting the relative proportion of the
    group in the original data set to the reference weights.
    """
    ref_weights = ref_weights / np.sum(ref_weights)
    rng = np.random.default_rng(random_state)
    orig_len = len(data)
    new_len = frac * orig_len
    idxs = np.arange(orig_len)
    while len(data) > new_len:
        weights = np.bincount(data, minlength=len(ref_weights))
        weights = weights / np.sum(weights)
        weights_diff = weights - ref_weights
        if not weights_diff.any():
            warnings.warn(f"Given frac {frac} is smaller than actual match {len(data) / orig_len}", UserWarning)
            break
        label_to_remove = np.argmax(weights_diff)
        idx_to_remove = np.where(data == label_to_remove)[0]
        idx_to_remove = rng.choice(idx_to_remove, size=1, replace=False)
        data = np.delete(data, idx_to_remove)
        idxs = np.delete(idxs, idx_to_remove)
    return idxs


def _exact_stratified_sampling(data, weights, random_state=42):
    """Exactly match the objective distribution by randomly choosing the number of samples from each group that is coded
    as relative portions of the original length in weights.
    IMPORTANT NOTE: This method is only approximately giving the right result, as the values may become complex float
    numbers and slight numerical errors occur due to rounding etc.
    """
    num_to_sample = (weights / np.max(weights))
    num_to_sample = num_to_sample.astype(np.float32)
    rng = np.random.default_rng(random_state)
    idxs = []
    for n, frac in enumerate(num_to_sample):
        sub_idxs = np.argwhere(data == n)[:, 0]
        size = int(len(sub_idxs) * frac)
        random_idxs = rng.choice(sub_idxs, size=size, replace=False)
        idxs.extend(random_idxs)
    return idxs


def iterative_stratified_sampling(data, ref_data, frac=0.8, random_state=42, exact=True):
    """Perform stratified sampling based on the given data by considering the distribution of ref_data.
    IMPORTANT NOTE: This method is only approximately giving the right result, as the values may become complex float
    numbers and slight numerical errors occur due to rounding etc.

    Parameters
    ----------
    data : list or array-like of shape (n_attributes, n_samples) when considering multiple attributes at once,
    or of shape (n_samples) when considering one attribute.
    Holds the discrete attributes of the data that will be resampled. The attributes should be similar to the attributes
    in ref_data.

    ref_data : {list, array-like} of shape (n_attributes, n_samples) when considering multiple attributes at once,
    or of shape (n_samples) when considering one attribute.
    Holds the discrete attributes of the reference data. The attributes should be similar to the attributes in ref_data.

    frac : float, default=0.8
    Must be a value between 0 and 1. Specifies how much data should be left after resampling (percentage from original).

    random_state : int, default=42
    Determines random selection when multiple samples are marked in the removal.

    exact : bool, default=True
    Determines whether an exact match of the reference distribution should be sampled.
    If set to True, frac will be ignored.

    Returns
    ----------
    idxs : list of indices to keep in the new dataset after stratified sampling.
    """
    data = np.array(data)
    ref_data = np.array(ref_data)
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=0)
    if len(ref_data.shape) < 2:
        ref_data = np.expand_dims(ref_data, axis=0)
    if data.shape[0] != ref_data.shape[0]:
        raise ValueError(f"Wrong input: data must have as many rows as ref_data. "
                         f"Passed: data {data.shape}, ref_data {ref_data.shape}.")
    data, ref_data = _get_overlap_groups(data, ref_data)

    weights_per_sample, weights, counts, ref_counts = _get_groups_weights(data, ref_data)

    max_frac = np.count_nonzero(weights_per_sample) / len(weights_per_sample)
    if frac > max_frac:
        warnings.warn(f"Less non-zero values than needed. Adjusting frac from {frac} to {max_frac}", UserWarning)
        frac = max_frac

    if exact:
        idxs = _exact_stratified_sampling(data, weights, random_state=random_state)
    else:
        idxs = _iterative_stratified_sampling(data, ref_counts, frac=frac, random_state=random_state)

    return idxs


def binning(data, bin_width=None, min_val=None, max_val=None):
    """Apply binning return an array that holds the bin idx at every position.

    Parameters
    ----------
    data : list or array-like of shape (n_samples).
    Holds the attributes of the data that will be discretized by binning.

    bin_width : int, float
    Specifies the width of the bins.

    min_val : int, float
    Specifies the left most bin.

    max_val : int, float
    Specifies the right most bin.

    Returns
    ----------
    binned_data : numpy array of the data descretized by the defined binning. Each element holds the bin index.
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    if bin_width is None:
        n_bins = 10
        bin_width = (max_val - min_val) / (n_bins - 1)
    bins = np.arange(min_val, max_val + bin_width, bin_width)  # Number of bins is len(bins) + 1, since left most counts
    binned_data = np.digitize(data, bins)  # right=False --> bins are always < right side
    return binned_data
