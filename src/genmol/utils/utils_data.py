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
import torch
import datasets
import torch
from safe.tokenizer import SAFETokenizer
from rdkit import RDLogger
from genmol.utils.bracket_safe_converter import safe2bracketsafe
RDLogger.DisableLog('rdApp.*')


ROOT_DIR = os.getcwd()


def get_last_checkpoint(save_dir):
    if os.path.exists(save_dir):
        filenames = os.listdir(save_dir)
        if filenames:
            last_filename = sorted(filenames, key=lambda x: int(x[:-5]))[-1]
            return os.path.join(save_dir, last_filename)
    

def get_tokenizer():
    tk = SAFETokenizer.from_pretrained('datamol-io/safe-gpt').get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk


class Collator:
    def __init__(self, config):
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = config.training.get('use_bracket_safe')
    
    def __call__(self, examples):
        if self.use_bracket_safe:
            for example in examples: example['input'] = safe2bracketsafe(example['input'])

        batch = self.tokenizer([example['input'] for example in examples],
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        del batch['token_type_ids']
        return batch
    

class UserDataset(datasets.Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            self.safe_list = f.readlines()
        self.safe_list = [s.strip('\n') for s in self.safe_list]
        
    def __len__(self):
        return len(self.safe_list)

    def __getitem__(self, indices):
        return {'input': self.safe_list[i] for i in indices}
    

def get_dataloader(config):
    if config.data == 'safe':
        return torch.utils.data.DataLoader(
            datasets.load_dataset('datamol-io/safe-gpt', streaming=True, split='train'),
            batch_size=config.loader.batch_size,
            collate_fn=Collator(config),
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,  # streaming
            persistent_workers=True)

    # User-defined dataset
    return torch.utils.data.DataLoader(
        UserDataset(config.data),
        batch_size=config.loader.batch_size,
        collate_fn=Collator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=True)