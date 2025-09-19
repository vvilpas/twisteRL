# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities to adapt raw observations coming from collectors."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class ObservationEncoder:
    """Base class for objects that convert raw observations to numpy arrays."""

    def __call__(self, obs: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MultiHotObservationEncoder(ObservationEncoder):
    """Expands sparse index observations into multi-hot vectors."""

    def __init__(self, obs_size: int, dtype: type = float):
        self.obs_size = obs_size
        self.dtype = dtype

    def __call__(self, obs: Sequence[Iterable[int]]) -> np.ndarray:
        np_obs = np.zeros((len(obs), self.obs_size), dtype=self.dtype)
        for row_idx, obs_indices in enumerate(obs):
            np_obs[row_idx, obs_indices] = 1.0
        return np_obs


class IdentityObservationEncoder(ObservationEncoder):
    """Leaves the observation untouched (apart from optional dtype casting)."""

    def __init__(self, dtype: type | None = None):
        self.dtype = dtype

    def __call__(self, obs: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        np_obs = np.asarray(obs)
        if self.dtype is not None:
            np_obs = np_obs.astype(self.dtype, copy=False)
        return np_obs


def make_observation_encoder(obs_size: int, config=None) -> ObservationEncoder:
    """Factory that creates an observation encoder from configuration."""

    if config is None:
        raise ValueError("Observation encoder configuration must be provided.")

    if isinstance(config, str):
        encoder_type = config
        params = {}
    else:
        encoder_type = config.get("type")
        if encoder_type is None:
            raise ValueError("Observation encoder configuration must include a 'type'.")
        params = {k: v for k, v in config.items() if k != "type"}

    if encoder_type == "multi_hot":
        dtype = params.get("dtype", int)
        return MultiHotObservationEncoder(obs_size, dtype=dtype)

    if encoder_type == "identity":
        dtype = params.get("dtype")
        return IdentityObservationEncoder(dtype=dtype)

    raise ValueError(f"Unknown observation encoder type: {encoder_type}")
