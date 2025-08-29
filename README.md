<h1 align="center">GenMol: A Drug Discovery Generalist with Discrete Diffusion</h1>

This is the official code repository for the paper titled [GenMol: A Drug Discovery Generalist with Discrete Diffusion](https://arxiv.org/abs/2501.06158) (ICML 2025).

<p align="center">
    <img width="750" src="assets/concept.png"/>
</p>


## Contribution
+ We introduce GenMol, a model for unified and versatile molecule generation by building masked discrete diffusion that generates SAFE molecular sequences.
+ We propose fragment remasking, an effective strategy for exploring chemical space using molecular fragments as the unit of exploration.
+ We propose molecular context guidance (MCG), a guidance scheme for GenMol to effectively utilize molecular context information.
+ We validate the efficacy and versatility of GenMol on a wide range of drug discovery tasks.

## Installation
Clone this repository:
```bash
git clone https://github.com/NVIDIA-Digital-Bio/genmol.git
cd genmol
```

Run the following command to install the dependencies:
```bash
bash env/setup.sh
```

Run the following command if you encounter the `ImportError: libXrender.so.1` error:
```bash
apt update && apt install -y libsm6 libxext6 && apt-get install -y libxrender-dev
```

Run the following command if you encounter the `ImportError: cannot import name '_CONFIG_FOR_DOC' from 'transformers.models.gpt2.modeling_gpt2'` error:
```bash
#!/bin/bash

# Use CONDA_PREFIX which points to current active environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No conda environment is currently active"
    exit 1
fi

# Comment out all lines in the safe package __init__.py
sed -i 's/^/# /' "$CONDA_PREFIX/lib/python3.10/site-packages/safe/__init__.py"

# Import required packages
echo "from .converter import SAFEConverter, decode, encode" >> "$CONDA_PREFIX/lib/python3.10/site-packages/safe/__init__.py"

echo "Fixed safe package in environment: $CONDA_PREFIX"
```

## Training
We provide the pretrained [checkpoint](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v1). Place `model.ckpt` in the current top genmol directory.

(Optional) To train GenMol from scratch, run the following command:
```bash
torchrun --nproc_per_node ${num_gpu} scripts/train.py hydra.run.dir=${save_dir} wandb.name=${exp_name}
```
Other hyperparameters can be adjusted in `configs/base.yaml`.<br>
The training used 8 NVIDIA A100 GPUs and took ~5 hours.

## (Optional) Training with User-defined Dataset
We used the [SAFE dataset](https://huggingface.co/datasets/datamol-io/safe-gpt) to train GenMol. To use your own training dataset, first convert your SMILES dataset into SAFE by running the following command:
```bash
python scripts/preprocess_data.py ${input_path} ${data_path}
```
`${input_path}` is the path to the dataset file with a SMILES in each row. For example,
```
CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1
NS(=O)(=O)c1cc2c(cc1Cl)NC(C1CC3C=CC1C3)NS2(=O)=O
...
```
`${data_path}` is the path of the processed dataset.

Then, set `data` in `base.yaml` to `${data_path}`.

## *De Novo* Generation
Run the following command to perform *de novo* generation:
```bash
python scripts/exps/denovo.py
```

If you see _pickle.UnpicklingError: invalid load key, '<' error. It is likely coming from /miniconda3/envs/genmol/lib/python3.10/site-packages/tdc/chem_utils/oracle/oracle.py", line 347, in readFragmentScores _fscores = pickle.load(f)

The root cause turned out to be a corrupted or incompletely downloaded pkl file for the SA score. The fix is simple: just grab the correct files from the official RDKit repository:
https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score/fpscores.pkl.gz

Extract the downloaded file into the genmol/oracle directory

The experiment in the paper used 1 NVIDIA A100 GPU.

## Fragment-constrained Generation
Run the following command to perform fragment-constrained generation:
```bash
python scripts/exps/frag.py
```

The experiment in the paper used 1 NVIDIA A100 GPU.

## Goal-directed Hit Generation (PMO Benchmark)

We provide the fragment vocabularies in the folder `scripts/exps/pmo/vocab`.

(Optional) Place [zinc250k.csv](https://www.kaggle.com/datasets/basu369victor/zinc250k) in the `data` folder, then run the following command to construct the fragment vocabularies and label the molecules with property labels:
```bash
python scripts/exps/pmo/get_vocab.py
```

Run the following command to perform goal-directed hit generation:
```bash
python scripts/exps/pmo/run.py -o ${oracle_name}
```
The generated molecules will be saved in `scripts/exps/pmo/main/genmol/results`.

Run the following command to evaluate the result:
```bash
python scripts/exps/pmo/eval.py ${file_name}
# e.g., python scripts/exps/pmo/eval.py scripts/exps/pmo/main/genmol/results/albuterol_similarity_0.csv
```

The experiment in the paper used 1 NVIDIA A100 GPU and took ~2-4 hours for each task.

## Goal-directed Lead Optimization
Run the following command to perform goal-directed lead optimization:
```bash
python scripts/exps/lead/run.py -o ${oracle_name} -i ${start_mol_idx} -d ${sim_threshold}
```
The generated molecules will be saved in `scripts/exps/lead/results`.

Run the following command to evaluate the result:
```bash
python scripts/exps/lead/eval.py ${file_name}
# e.g., python scripts/exps/lead/eval.py scripts/exps/lead/results/parp1_id0_thr0.4_0.csv
```

The experiment in the paper used 1 NVIDIA A100 GPU and took ~10 min for each task.


## License
Copyright @ 2025, NVIDIA Corporation. All rights reserved.<br>
The source code is made available under Apache-2.0.<br>
The model weights are made available under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Citation
If you find this repository and our paper useful, we kindly request to cite our work.
```BibTex
@article{lee2025genmol,
  title     = {GenMol: A Drug Discovery Generalist with Discrete Diffusion},
  author    = {Lee, Seul and Kreis, Karsten and Veccham, Srimukh Prasad and Liu, Meng and Reidenbach, Danny and Peng, Yuxing and Paliwal, Saee and Nie, Weili and Vahdat, Arash},
  journal   = {International Conference on Machine Learning},
  year      = {2025}
}
```
