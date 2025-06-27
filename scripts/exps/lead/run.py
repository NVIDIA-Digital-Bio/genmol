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
import random
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem, QED, RDConfig
from scripts.exps.lead.docking.docking import DockingVina
from genmol.sampler import Sampler
from genmol.utils.utils_chem import cut
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class GenMolOpt():
    def __init__(self, args):
        super().__init__()
        self.args = args

        # df = pd.read_csv('scripts/exps/lead/docking/actives.csv')
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(script_dir, 'docking', 'actives.csv'))
        df = df[df['target'] == self.args.oracle_name]
        self.start_smiles = df['smiles'].iloc[self.args.start_mol_idx]
        start_mol = Chem.MolFromSmiles(self.start_smiles)
        self.start_fp = AllChem.GetMorganFingerprintAsBitVect(start_mol, 2, 2048)
        self.start_prop = df['DS'].iloc[self.args.start_mol_idx]
        print(f'Start SMILES:\t{self.start_smiles}')
        print(f'Start DS:\t{self.start_prop}')

        self.predictor = DockingVina(self.args.oracle_name)
        self.population = [(self.start_prop, frag) for frag in cut(self.start_smiles)]
        print(f'Initial population: {len(self.population)} frags')
        self.sampler = Sampler(self.args.model_path)

        self.fname = f'results/{self.args.oracle_name}_id{self.args.start_mol_idx}_' + \
                     f'thr{self.args.sim_thr}_{self.args.seed}.csv'
        print(f'\033[92m{self.fname}\033[0m')
        self.fname = os.path.join(ROOT_DIR, self.fname)

        if not os.path.exists(os.path.dirname(self.fname)):
            os.mkdir(os.path.dirname(self.fname))
    
    def reward_vina(self, smiles_list):
        reward = - np.array(self.predictor.predict(smiles_list))
        reward = np.clip(reward, 0, None)
        return reward
    
    def reward_qed(self, mols):
        return [QED.qed(m) for m in mols]
    
    def reward_sa(self, mols):
        return [(10 - sascorer.calculateScore(m)) / 9 for m in mols]
    
    def reward_sim(self, mols):
        mol_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
        return DataStructs.BulkTanimotoSimilarity(self.start_fp, mol_fps)
        
    def reward(self, smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        rv = self.reward_vina(smiles_list)
        rq = self.reward_qed(mols)
        rs = self.reward_sa(mols)
        rsim = self.reward_sim(mols)
        return rv, rq, rs, rsim
    
    def attach(self, frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return mols[idx][0]
    
    def update_population(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            if rv > self.start_prop and rq >= 0.6 and rs >= 6/9 and rsim >= self.args.sim_thr:
                frags = {frag for frag in cut(smiles)}
                self.population.extend([(rv, frag) for frag in frags])
        self.population.sort(reverse=True)

    def generate(self):
        for _ in range(1000):
            frag1, frag2 = random.sample([frag for prop, frag in self.population], 2)
            smiles = Chem.MolToSmiles(self.attach(frag1, frag2))
            if smiles is None: continue
            smiles = self.sampler.mask_modification(smiles, min_len=50, gamma=self.args.gamma)
            if smiles is not None:
                smiles = sorted(smiles.split('.'), key=len)[-1]     # get the largest
            return smiles
            
    def record(self, smiles_list, prop_list):
        with open(self.fname, 'a') as f:
            for i in range(len(smiles_list)):
                str = f'{smiles_list[i]},'
                for props in prop_list: str += f'{props[i]},'
                str += '\n'
                f.write(str)

    def run(self):
        t_start = time()
        for i in range(self.args.num_iter):
            smiles_list = [self.generate() for _ in range(self.args.num_gen)]
            prop_list = self.reward(smiles_list)
            self.update_population(smiles_list, prop_list)
            self.record(smiles_list, prop_list)
            print(f'[Iter {i+1:03d}] Top DS: {self.population[0][0]}')
        print(f'{time() - t_start:.2f} sec elapsed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--oracle_name',      type=str,   default='parp1',
                        choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2'])
    parser.add_argument('-i', '--start_mol_idx',    type=int,   default=0, choices=[0, 1, 2])
    parser.add_argument('-d', '--sim_thr',          type=float, default=0.4)
    parser.add_argument('-s', '--seed',             type=int,   default=0)
    parser.add_argument('-m', '--model_path',       type=str,   default='model.ckpt')
    parser.add_argument('--num_gen',                type=int,   default=100)
    parser.add_argument('--num_iter',               type=int,   default=10)
    parser.add_argument('--gamma',                  type=float, default=0)
    args = parser.parse_args()

    GenMolOpt(args).run()
