# -*- coding: utf-8 -*-

"""Unit tests for observation encoders."""

import numpy as np

from twisterl.rl.observation import (
    make_observation_encoder,
    MultiHotObservationEncoder,
    IdentityObservationEncoder,
)


def test_make_observation_encoder_multi_hot():
    enc = make_observation_encoder(6, {"type": "multi_hot", "dtype": float})
    assert isinstance(enc, MultiHotObservationEncoder)
    obs = [[0, 1], [4]]
    encoded = enc(obs)
    assert encoded.shape == (2, 6)
    assert np.all(encoded[0, [0, 1]] == 1.0)
    assert encoded[1, 4] == 1.0
    assert encoded.dtype == float


def test_make_observation_encoder_identity():
    enc = make_observation_encoder(0, {"type": "identity", "dtype": float})
    assert isinstance(enc, IdentityObservationEncoder)
    obs = [[0.1, 0.2], [0.3, 0.4]]
    encoded = enc(obs)
    assert encoded.shape == (2, 2)
    assert encoded.dtype == float


def test_make_observation_encoder_requires_type():
    try:
        make_observation_encoder(5, {})
    except ValueError as err:
        assert "type" in str(err)
    else:
        raise AssertionError("Expected ValueError when encoder type is missing.")
