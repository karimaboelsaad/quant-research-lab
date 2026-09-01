import pytest

from src.walk_forward import calculate_common_training_start,calculate_unused_walk_forward_rows


def test_common_training_start_uses_largest_lookback():
    assert calculate_common_training_start([20,60,120],500)==120


def test_common_training_start_requires_scored_training_rows():
    with pytest.raises(ValueError,match="extend beyond"):
        calculate_common_training_start([20,60,120],120)


def test_unused_walk_forward_rows_are_reported():
    assert calculate_unused_walk_forward_rows(1030,400,100,100)==30
