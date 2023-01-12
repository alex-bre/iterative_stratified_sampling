import numpy as np
from iterative_stratified_sampling import iterative_stratified_sampling

test_data = ["A", "A", "A", "A", "A",
             "B", "B",
             "C", "C", "C"]
test_data = np.array(test_data)

ref_data = ["B", "B", "B",
            "A", "A", "A",
            "C", "C"]
ref_data = np.array(ref_data)


def test_exact():
    idxs = iterative_stratified_sampling(test_data, ref_data, exact=True)
    new_data = test_data[idxs].tolist()
    assert new_data == ["A", "A", "B", "B", "C"]


def test_iter60perc():
    idxs = iterative_stratified_sampling(test_data, ref_data, frac=0.6, exact=False)
    new_data = test_data[idxs].tolist()
    assert new_data == ["A", "A", "B", "B", "C", "C"]
