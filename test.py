import pytest
from linear_regression import read_data, compute_regression

data_frame = read_data("./resources/apartmentComplexData.txt")
regression, score = compute_regression(data_frame, 8)


def test_model_accuracy():
    accuracy = score["r2_score"]
    assert accuracy > 0.5
