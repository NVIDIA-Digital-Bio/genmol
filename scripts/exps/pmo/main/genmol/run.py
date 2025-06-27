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
import random
from time import time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from easydict import EasyDict
from scripts.exps.pmo.main.optimizer import BaseOptimizer
from genmol.sampler import Sampler
from genmol.utils.utils_chem import cut


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class GenMol_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = 'GenMol'
        self.oracle_name = self.args.oracle
    
    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        config = EasyDict(config)
        config.seed = self.args.seed
        config.oracle_name = self.args.oracle
        GenMolOpt(config, self.oracle).run()


class GenMolOpt():
    def __init__(self, args, oracle):
        super().__init__()
        # control hyperparameters
        if args.oracle_name in {'albuterol_similarity',
                                'isomers_c7h8n2o2',
                                'isomers_c9h10n2o2pf2cl',
                                'median1', 'qed',
                                'sitagliptin_mpo',
                                'zaleplon_mpo'}:
            args.min_mol_size, args.max_mol_size = 10, 30
        elif args.oracle_name in {'gsk3b', 'jnk3'}:
            args.min_mol_size, args.max_mol_size = 30, 80

        self.args = args
        self.oracle = oracle
        self.sampler = Sampler(args.model_path)
        
        self.set_initial_population()
        self.iter = 0
        
        self.fname = f'main/genmol/results/{args.oracle_name}_{args.seed}.csv'
        print(f'\033[92m{self.fname}\033[0m')
    
    def set_initial_population(self):
        df = pd.read_csv(os.path.join(ROOT_DIR, f'vocab/{self.args.oracle_name}.csv'))
        df = df.iloc[:self.args.population_size]
        self.population = list(zip(df['score'], df['frag']))
    
    def attach(self, frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return mols[idx][0]
    
    def update_population(self, smiles, prop):
        population = {frag for prop, frag in self.population}
        if prop > self.population[-1][0]:
            frags = cut(smiles)
            self.population.extend([(prop, frag) for frag in frags if frag not in population])
            self.population.sort(reverse=True)
            self.population = self.population[:self.args.population_size]
    
    def generate(self):
        for i in range(1000):
            frag1, frag2 = random.sample([frag for prop, frag in self.population], 2)
            smiles = Chem.MolToSmiles(self.attach(frag1, frag2))
            if smiles is None: continue
            if self.iter > self.args.warmup:
                smiles = self.sampler.mask_modification(smiles, gamma=self.args.gamma)
                if smiles is not None:
                    smiles = max(smiles.split('.'), key=len)    # get the largest
            if self.args.min_mol_size <= Chem.MolFromSmiles(smiles).GetNumAtoms() <= self.args.max_mol_size:
                return smiles

    def record(self, smiles, prop):
        with open(os.path.join(ROOT_DIR, self.fname), 'a') as f:
            f.write(f'{smiles},{prop}\n')
    
    def run(self):
        t_start = time()
        for i in range(30000):
            self.iter = i
            
            smiles = self.generate()
            prop = self.oracle(smiles)
            self.update_population(smiles, prop)
            self.record(smiles, prop)

            if self.oracle.finish:
                print(f'[{time() - t_start:.2f} sec] Completed')
                break
        else:
            print(f'[{time() - t_start:.2f} sec] Maximum iteration reached')
        