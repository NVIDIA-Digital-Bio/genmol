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


# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

import argparse
import safe as sf


parser = argparse.ArgumentParser()
parser.add_argument('input_path')
parser.add_argument('data_path')
args = parser.parse_args()

with open(args.input_path) as f:
    smiles_list = f.readlines()

safe_list = []
for smiles in smiles_list:
    safe_str = sf.SAFEConverter(ignore_stereo=True).encoder(smiles, allow_empty=True)
    safe_list.append(safe_str + '\n')

with open(args.data_path, 'w') as f:
    f.writelines(safe_list)
