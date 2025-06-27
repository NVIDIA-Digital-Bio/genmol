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

from safe.converter import *


class BracketSAFEConverter(SAFEConverter):
    def encoder(
        self,
        inp: Union[str, dm.Mol],
        canonical: bool = True,
        randomize: Optional[bool] = False,
        seed: Optional[int] = None,
        constraints: Optional[List[dm.Mol]] = None,
        allow_empty: bool = False,
        rdkit_safe: bool = True,
    ):
        rng = None
        if randomize:
            rng = np.random.default_rng(seed)
            if not canonical:
                inp = dm.to_mol(inp, remove_hs=False)
                inp = self.randomize(inp, rng)

        if isinstance(inp, dm.Mol):
            inp = dm.to_smiles(inp, canonical=canonical, randomize=False, ordered=False)

        branch_numbers = self._find_branch_number(inp)

        mol = dm.to_mol(inp, remove_hs=False)
        if self.ignore_stereo:
            mol = dm.remove_stereochemistry(mol)

        bond_map_id = 1
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(0)
                atom.SetIsotope(bond_map_id)
                bond_map_id += 1

        if self.require_hs:
            mol = dm.add_hs(mol)
        matching_bonds = self._fragment(mol, allow_empty=allow_empty)
        substructed_ignored = []
        if constraints is not None:
            substructed_ignored = list(
                itertools.chain(
                    *[
                        mol.GetSubstructMatches(constraint, uniquify=True)
                        for constraint in constraints
                    ]
                )
            )

        bonds = []
        for i_a, i_b in matching_bonds:
            # if both atoms of the bond are found in a disallowed substructure, we cannot consider them
            # on the other end, a bond between two substructure to preserved independently is perfectly fine
            if any((i_a in ignore_x and i_b in ignore_x) for ignore_x in substructed_ignored):
                continue
            obond = mol.GetBondBetweenAtoms(i_a, i_b)
            bonds.append(obond.GetIdx())

        if len(bonds) > 0:
            mol = Chem.FragmentOnBonds(
                mol,
                bonds,
                dummyLabels=[(i + bond_map_id, i + bond_map_id) for i in range(len(bonds))],
            )
        
        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if randomize:
            frags = rng.permutation(frags).tolist()
        elif canonical:
            frags = sorted(
                frags,
                key=lambda x: x.GetNumAtoms(),
                reverse=True,
            )

        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [
                atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0
            ]
            frags_str.append(
                Chem.MolToSmiles(
                    frag,
                    isomericSmiles=True,
                    canonical=True,  # needs to always be true
                    rootedAtAtom=non_map_atom_idxs[0],
                )
            )

        scaffold_str = ".".join(frags_str)
        
        # don't capture atom mapping in the scaffold
        attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", scaffold_str))
        if canonical:
            attach_pos = sorted(attach_pos)
        starting_num = 1
        for attach in attach_pos:
            val = str(starting_num) if starting_num < 10 else f"%{starting_num}"
            val = '<' + val + '>'   # bracket added
            # we cannot have anything of the form "\([@=-#-$/\]*\d+\)"
            attach_regexp = re.compile(r"(" + re.escape(attach) + r")")
            scaffold_str = attach_regexp.sub(val, scaffold_str)
            starting_num += 1

        # now we need to remove all the parenthesis around digit only number
        wrong_attach = re.compile(r"\((<[\%\d+]*>)\)")   # bracket added
        scaffold_str = wrong_attach.sub(r"\g<1>", scaffold_str)
        # furthermore, we autoapply rdkit-compatible digit standardization.
        if rdkit_safe:
            pattern = r"\(([=-@#\/\\]{0,2})(%?\d{1,2})\)"
            replacement = r"\g<1>\g<2>"
            scaffold_str = re.sub(pattern, replacement, scaffold_str)
        return scaffold_str


def safe2bracketsafe(safe_str):
    try:
        return BracketSAFEConverter().encoder(Chem.MolFromSmiles(safe_str), allow_empty=True, canonical=False, randomize=True)
    except:
        return safe_str
    

def bracketsafe2safe(safe_str):
    intrafrag_points = [m.group(0) for m in re.finditer(r'(?<!%)\d(?!>)', safe_str)] + \
        [m.group(0).lstrip('%') for m in re.finditer(r'%\d+', safe_str)]
    starting_num = max([int(i) for i in intrafrag_points]) + 1 if intrafrag_points else 0
    interfrag_points = [(m.start(0), m.end(0)) for m in re.finditer(r'<\d+>', safe_str)]
    
    safe_str = list(safe_str)
    for start, end in interfrag_points:
        safe_str[start] = safe_str[end-1] = ' ' # '<', '>' -> ''
        num_to_replace = int(''.join(safe_str[start+1 : end-1])) + starting_num
        num_to_replace = '%' + str(num_to_replace) if num_to_replace >= 10 else str(num_to_replace)
        safe_str[start+1 : end-1] = [num_to_replace] + [' '] * (end - start - 3)
    safe_str = re.sub(' ', '', ''.join(safe_str))
    return safe_str
