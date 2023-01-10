import numpy as np
import pytest
from iterative_stratified_sampling import iterative_stratified_sampling

test_data1 = ["A", "A", "A",
              "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
test_data2 = ["X", "X",
              "Y",
              "X", "X", "X", "X", "X", "X",
              "Y", "Y", "Y", "Y"]
test_data = np.array([test_data1, test_data2])

ref_data1 = ["A", "A",
             "B", "B",
             "A", "A",
             "B", "B"]
ref_data2 = ["X", "Y",
             "X", "Y",
             "X", "Y",
             "X", "Y"]
ref_data = np.array([ref_data1, ref_data2])


def test_exact():
    idxs = iterative_stratified_sampling(test_data, ref_data, exact=True)
    new_data = test_data[:, idxs]
    new_data = list(map("_".join, zip(*new_data)))
    assert new_data == ["A_X", "A_Y", "B_X", "B_Y"]


def test_iter_3perc():
    idxs = iterative_stratified_sampling(test_data, ref_data, frac=0.3077, exact=False)
    new_data = test_data[:, idxs]
    new_data = list(map("_".join, zip(*new_data)))
    assert new_data == ["A_X", "A_Y", "B_X", "B_Y"]


def test_iter_1perc():
    with pytest.warns(UserWarning):
        idxs = iterative_stratified_sampling(test_data, ref_data, frac=0.231, exact=False)
    new_data = test_data[:, idxs]
    new_data = list(map("_".join, zip(*new_data)))
    assert new_data == ["A_X", "A_Y", "B_X", "B_Y"]
