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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-d', '--sim_thr', type=float, default=0.4)
    args = parser.parse_args()

    df = pd.read_csv(args.file, names=['smiles', 'DS', 'QED', 'SA', 'SIM', ''])
    num_gen = 1000  # len(df)
    df = df.drop_duplicates(subset=['smiles'])
    print(f'Uniqueness:\t{len(df) / num_gen}')
    
    df = df[df['SIM'] >= args.sim_thr]
    df = df[df['QED'] >= 0.6]
    df = df[df['SA'] >= 6 / 9]
    if not len(df):
        print('Lead optimization failed')
    else:
        df = df.sort_values(by='DS', ascending=False)
        print(f'Top DS:\t\t{(df["DS"].iloc[0])}')
        print(f'Top mol:\t{(df["smiles"].iloc[0])}')
