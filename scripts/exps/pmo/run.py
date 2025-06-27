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

import argparse
import yaml
import os
import sys
from time import time
from tdc import Oracle
from scripts.exps.pmo.main.genmol.run import GenMol_Optimizer as Optimizer

# https://github.com/wenhao-gao/mol_opt/blob/main/run.py
def main():
    start_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='hparams.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-o', '--oracle', default='albuterol_similarity',
                        choices=['albuterol_similarity',
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
                                 'zaleplon_mpo'])
    args = parser.parse_args()
    
    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, 'main/genmol')
    sys.path.append(path_main)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print(f'Optimizing oracle function: {args.oracle}')

    try:
        config = yaml.safe_load(open(args.config))
    except:
        config = yaml.safe_load(open(os.path.join(path_main, args.config)))
    
    oracle = Oracle(name=args.oracle)
    optimizer = Optimizer(args=args)

    print('seed', args.seed)
    optimizer.optimize(oracle=oracle, config=config, seed=args.seed)

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()
