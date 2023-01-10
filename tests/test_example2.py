import numpy as np
from iterative_stratified_sampling import iterative_stratified_sampling

test_data = ["A", "A", "A", "A", "A",
             "B", "B",
             "C", "C", "C"]
test_data = np.array(test_data)

ref_data = ["A", "A", "A",
            "B", "B", "B"
            "C", "C"]
ref_data = np.array(ref_data)


def test_exact():
    idxs = iterative_stratified_sampling(test_data, ref_data, exact=True)
    new_data = test_data[idxs].tolist()
    assert new_data == ["A", "A", "A", "B", "B", "C"]
