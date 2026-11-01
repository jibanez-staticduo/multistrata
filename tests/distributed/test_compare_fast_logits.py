import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("compare_fast_logits", Path(__file__).parents[2] /
                                            "tools/distributed/compare_fast_logits.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
POLICY = {"logits": {"kl_mean_max": .01, "kl_max_max": .05,
                     "top1_min_fraction": .95, "near_tie_logit_margin": .1}}


def test_distribution_shift_invariance_and_rounding():
    a = np.array([[1000., 999., 998.], [-1000., -1002., -1004.]])
    b = a + 1234.0
    b[0, 1] += .001
    result = module.compare(a, b, POLICY)
    assert result["qualified"] and 0 < result["kl_mean"] < .000001
    assert result["top1_agreement"] == 1


def test_same_top_token_does_not_hide_distribution_regression():
    a = np.array([[2., 1., 0.]])
    b = np.array([[20., -10., -20.]])
    result = module.compare(a, b, POLICY)
    assert result["top1_agreement"] == 1 and not result["qualified"]


def test_near_tie_is_reported_without_exempting_the_gate():
    result = module.compare(np.array([[1., .999, 0.]]), np.array([[.999, 1., 0.]]), POLICY)
    assert result["rows"][0]["near_tie"] and result["stable_margin_top1_agreement"] is None
    assert not result["qualified"]


@pytest.mark.parametrize("candidate", [np.array([[np.nan, 1.]]), np.array([[np.inf, 1.]]), np.zeros((2, 3))])
def test_invalid_arrays_never_qualify(candidate):
    with pytest.raises(ValueError):
        module.compare(np.zeros((1, 2)), candidate, POLICY)
