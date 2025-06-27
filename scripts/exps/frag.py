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

import pandas as pd
import numpy as np
from tdc import Oracle, Evaluator
from rdkit import DataStructs, Chem, RDLogger
from rdkit.Chem import AllChem
from genmol.sampler import Sampler
RDLogger.DisableLog('rdApp.*')


def get_distance(smiles, df):
    if 'MOL' not in df:
        df['MOL'] = df['smiles'].apply(Chem.MolFromSmiles)
    
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df['MOL']]
    
    fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 1024)
    return np.mean(DataStructs.BulkTanimotoSimilarity(fps, df['FPS'].tolist(), returnDistance=True))


if __name__ == '__main__':
    num_samples = 100
    evaluator = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')
    demo = Sampler('model.ckpt')
    data = pd.read_csv('data/fragments.csv')
    tasks = ['linker_design', 'motif_extension', 'scaffold_decoration', 'superstructure_generation']
    # tasks = ['linker_design_onestep']
    #! Linker Design / Scaffold Morphing = generate a linker fragment that connects given two side chains
    #! Motif Extension = generate molecule with existing motif
    #! Scaffold Decoration = same as Motif Extension but start with larger scaffold 
    #! Superstructure Generation = generate a molecule when a substructure constraint is given
    #! Linker Design (1-step) = generate a linker fragment that connects given two side chains without sequence mixing
    for task in tasks:
        if task == 'linker_design' or task == 'scaffold_morphing':
            task = 'linker_design'
            sampling_fn = lambda f: demo.fragment_linking(f, num_samples, randomness=3)
        elif task == 'motif_extension':
            sampling_fn = lambda f: demo.fragment_completion(f, num_samples, randomness=1.2, gamma=0.3)
        elif task == 'scaffold_decoration':
            sampling_fn = lambda f: demo.fragment_completion(f, num_samples, gamma=0.3)
        elif task == 'superstructure_generation':
            sampling_fn = lambda f: demo.fragment_completion(f, num_samples, gamma=0.4)
        elif task == 'linker_design_onestep':
            task = 'linker_design'
            sampling_fn = lambda f: demo.fragment_linking_onestep(f, num_samples, randomness=3)
        
        validity, uniqueness, diversity, distance, quality = [], [], [], [], []
        for original, fragment in zip(data['smiles'], data[task]):
            samples = sampling_fn(fragment)
            if len(samples) == 0:
                validity.append(0)
                uniqueness.append(0)
                quality.append(0)
                continue
            df = pd.DataFrame({'smiles': samples, 'qed': oracle_qed(samples), 'sa': oracle_sa(samples)})
            validity.append(len(df['smiles']) / num_samples)
            df = df.drop_duplicates('smiles')
            uniqueness.append(len(df['smiles']) / len(samples))
            if len(df['smiles']) == 1:
                diversity.append(0)
            else:
                diversity.append(evaluator(df['smiles']))
            distance.append(get_distance(original, df))
            df = df[df['qed'] >= 0.6]
            df = df[df['sa'] <= 4]
            quality.append(len(df) / num_samples)

        print(f'{task}')
        print(f'\tValidity:\t{np.mean(validity)}')
        print(f'\tUniqueness:\t{np.mean(uniqueness)}')
        print(f'\tDiversity:\t{np.mean(diversity)}')
        print(f'\tDistance:\t{np.mean(distance)}')
        print(f'\tQuality:\t{np.mean(quality)}')
        print('-' * 50)
