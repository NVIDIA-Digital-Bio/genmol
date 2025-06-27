# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import torch
from typing import Optional, Tuple, Union
from jaxtyping import Bool, Float
from bionemo.moco.distributions.time import UniformTimeDistribution


class AntitheticUniformTimeDistribution(UniformTimeDistribution):
    """A class representing a uniform time distribution."""

    def __init__(
        self,
        min_t: Float = 0.0,
        max_t: Float = 1.0,
        discrete_time: Bool = False,
        nsteps: Optional[int] = None,
        rng_generator: Optional[torch.Generator] = None,
        sampling_eps: Float = 0.0,
    ):
        """Initializes a UniformTimeDistribution object.

        Args:
            min_t (Float): The minimum time value.
            max_t (Float): The maximum time value.
            discrete_time (Bool): Whether the time is discrete.
            nsteps (Optional[int]): Number of nsteps for discretization.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
            sampling_eps: The sampling epsilon.
        """
        super().__init__(discrete_time, nsteps, min_t, max_t, rng_generator)
        self.sampling_eps = sampling_eps

    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Generates a specified number of samples from the uniform time distribution.

        Args:
            n_samples (int): The number of samples to generate.
            device (str): cpu or gpu.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            A tensor of samples.
        """
        if rng_generator is None:
            rng_generator = self.rng_generator
        if self.discrete_time:
            if self.nsteps is None:
                raise ValueError("nsteps cannot be None for discrete time sampling")
            # Generate random values and apply offset like continuous case
            time_step = torch.rand(n_samples, device=device, generator=rng_generator)
            offset = torch.arange(n_samples, device=device) / n_samples
            time_step = (time_step / n_samples + offset) % 1
            # Scale to discrete steps
            time_step = (time_step * self.nsteps).long()
        else:
            time_step = torch.rand(n_samples, device=device, generator=rng_generator)
            offset = torch.arange(n_samples, device=device) / n_samples
            time_step = (time_step / n_samples + offset) % 1
            time_step = (1 - self.sampling_eps) * time_step + self.sampling_eps
            if self.min_t and self.max_t and self.min_t > 0:
                time_step = time_step * (self.max_t - self.min_t) + self.min_t
        return time_step
