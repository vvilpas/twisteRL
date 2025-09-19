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

import numpy as np
from abc import ABC, abstractmethod


class ObservationEncoder(ABC):
    @abstractmethod
    def encode(self, obs):
        """Transform raw observations into encoded format."""
        pass


class OneHotEncoder(ObservationEncoder):
    def __init__(self, obs_size):
        self.obs_size = obs_size

    def encode(self, obs):
        """Convert discrete observations to one-hot encoded vectors."""
        np_obs = np.zeros((len(obs), self.obs_size), dtype=float)
        for i, obs_i in enumerate(obs):
            np_obs[i, obs_i] = 1.0
        return np_obs


class IdentityEncoder(ObservationEncoder):
    def encode(self, obs):
        """Pass observations through unchanged."""
        return np.array(obs, dtype=float)


def create_encoder(encoder_type, obs_size):
    """Factory function to create encoders based on configuration."""
    if encoder_type == "one_hot":
        return OneHotEncoder(obs_size)
    elif encoder_type == "identity":
        return IdentityEncoder()
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")