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


import argparse
import pandas as pd
from main.optimizer import top_auc
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    file = parser.parse_args().file

    df = pd.read_csv(file, names=['smiles', 'score'])
    df = df.drop_duplicates(subset=['smiles']).iloc[:10000]
    df = df.sort_values(by='score', ascending=False)
    
    print(f"Avg. Top-1:\t{df.iloc[0]['score']}")
    print(f"Avg. Top-10:\t{df.iloc[:10]['score'].mean()}")
    print(f"Avg. Top-100:\t{df.iloc[:100]['score'].mean()}")

    mol_dict = {df.iloc[i]['smiles']: [df.iloc[i]['score'], df.index[i]] for i in range(len(df))}
    print(f"AUC Top-1:\t{top_auc(mol_dict, 1, True, 100, 10000)}")
    print(f"AUC Top-10:\t{top_auc(mol_dict, 10, True, 100, 10000):.3f}")
    print(f"AUC Top-100:\t{top_auc(mol_dict, 100, True, 100, 10000):.3f}")
