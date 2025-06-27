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
from collections import defaultdict
from tqdm import trange
import pandas as pd
from tdc import Oracle
from genmol.utils.utils_chem import cut
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    props = ['albuterol_similarity',
             'amlodipine_mpo',
             'celecoxib_rediscovery',
             'deco_hop',
             'drd2',
             'fexofenadine_mpo',
             'gsk3b',
             'isomers_c7h8n2o2',
             'isomers_c9h10n2o2pf2cl',
             'jnk3',
             'median1',
             'median2',
             'mestranol_similarity',
             'osimertinib_mpo',
             'perindopril_mpo',
             'qed',
             'ranolazine_mpo',
             'scaffold_hop',
             'sitagliptin_mpo',
             'thiothixene_rediscovery',
             'troglitazone_rediscovery',
             'valsartan_smarts',
             'zaleplon_mpo']
    
    df = pd.read_csv('data/zinc250k.csv')

    # calculate properties
    for prop in props:
        if prop not in df:
            print(f'Calculating {prop}...')
            df[prop] = Oracle(prop)(df['smiles'].tolist())
            df.to_csv('data/zinc250k.csv', index=False)
    
    # construct vocabulary
    avg_num_frags = 0
    frag2cnt = defaultdict(int)
    frag2score = {prop: defaultdict(float) for prop in props}
    for i in trange(len(df)):
        frags = cut(df['smiles'].iloc[i])
        avg_num_frags += len(frags)
        for frag in frags:
            frag2cnt[frag] += 1
            for prop in props:
                frag2score[prop][frag] += df[prop].iloc[i]
    print(f'Average # of fragments: {avg_num_frags / len(df):.2f}')
    
    foldername = 'scripts/exps/pmo/vocab'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    for prop in props:
        for k in frag2score[prop]:
            # mean property value of the fragment
            frag2score[prop][k] /= frag2cnt[k]
    
        df = pd.DataFrame({'frag': frag2score[prop].keys(),
                           'score': frag2score[prop].values()})
        df = df.sort_values(by='score', ascending=False).iloc[:10000]
        df['size'] = df['frag'].apply(lambda frag: Chem.MolFromSmiles(frag).GetNumAtoms())
        df.to_csv(f'{foldername}/{prop}.csv', index=False)
