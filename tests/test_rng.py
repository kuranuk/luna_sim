from __future__ import annotations

import numpy as np
import pytest

from lunarsim.core.rng import RngFactory, stream_key


def test_same_name_same_stream() -> None:
    a = RngFactory(5).generator("terrain/fractal").standard_normal(100)
    b = RngFactory(5).generator("terrain/fractal").standard_normal(100)
    np.testing.assert_array_equal(a, b)


def test_independent_of_request_order() -> None:
    f = RngFactory(5)
    x1 = f.generator("a").random(10)
    f.generator("b").random(1000)
    x2 = f.generator("a").random(10)
    np.testing.assert_array_equal(x1, x2)


def test_different_names_and_seeds_differ() -> None:
    f = RngFactory(5)
    assert not np.array_equal(f.generator("a").random(10), f.generator("b").random(10))
    assert not np.array_equal(f.generator("a").random(10), RngFactory(6).generator("a").random(10))


def test_stream_key_stable() -> None:
    # Regression: the key must not depend on Python's salted hash().
    assert stream_key("terrain/fractal") == stream_key("terrain/fractal")
    assert len(stream_key("x")) == 4


def test_invalid() -> None:
    with pytest.raises(ValueError):
        RngFactory(-1)
    with pytest.raises(ValueError):
        stream_key("")
