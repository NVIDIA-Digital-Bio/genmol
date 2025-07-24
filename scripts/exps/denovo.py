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
import sys
sys.path.append(os.path.realpath('.'))

from time import time
import pandas as pd
from tdc import Oracle, Evaluator
from genmol.sampler import Sampler


if __name__ == '__main__':
    num_samples = 1000
    evaluator = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')
    sampler = Sampler('model.ckpt')
    
    t_start = time()
    samples = sampler.de_novo_generation(num_samples, softmax_temp=0.5, randomness=0.5)
    print(f'Time:\t\t{time() - t_start:.2f} sec')
    df = pd.DataFrame({'smiles': samples, 'qed': oracle_qed(samples), 'sa': oracle_sa(samples)})
    print(f'Validity:\t{len(df["smiles"]) / num_samples}')
    df = df.drop_duplicates('smiles')
    print(f'Uniqueness:\t{len(df["smiles"]) / len(samples)}')
    print(f'Diversity:\t{evaluator(df["smiles"])}')
    df = df[df['qed'] >= 0.6]
    df = df[df['sa'] <= 4]
    print(f'Quality:\t{len(df) / num_samples}')
