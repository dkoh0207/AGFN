# Atomic GFlowNets- Pretraining & Finetuning atom based GFlowNets with Inexpensive Rewards for Molecule Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)[![Paper](https://img.shields.io/badge/arXiv-2503.06337-b31b1b.svg)](https://arxiv.org/abs/2503.06337)

## 📄 Paper Reference

**Title**: *Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation*

**Authors**: Mohit Pandey, Gopeshh Subbaraj, Artem Cherkasov, Martin Ester, Emmanuel Bengio

**arXiv**: [arXiv:2503.06337](https://arxiv.org/abs/2503.06337)

This repository provides code and configurations to replicate the pretraining, fine-tuning, and case-studies (denovo molecule design & lead optimization) experiments reported in our paper, including support for both standard Trajectory Balance and Relative Trajectory Balance (RTB).

## ⚙️ Installation

```bash
conda create -n agfn python=3.12 -y
conda activate agfn
cd agfn
pip install -r requirements_dev.txt
pip install -e .
```

> ℹ️ The dependency stack targets **PyTorch 2.8.0 + CUDA 12.9 wheels** and supports
> Blackwell GPUs (RTX 50 series, B100/B200, compute capability `sm_120`). Older
> hardware (Ampere/Hopper) remains supported by the same wheels.

## 🔬 Pretraining

To begin pretraining the AGFN model, follow these steps:

```bash
cd ./AGFN
conda activate agfn
python ./src/pretrainSrc/driver.py
```

This uses the default training configuration. To modify default settings update `./src/config/pretrain.yml`.

⚠️ Note:
By default, the script utilizes **all available GPUs** on the node. To restrict the number of GPUs, update the `world_size` parameter in the `if __name__ == '__main__'` block of `driver.py`.

The code has been tested on up to **8 NVIDIA A100 GPUs** and verified on a **single NVIDIA RTX 5090** (Blackwell, `sm_120`) with PyTorch 2.8.0+cu129.

Model checkpoints will be saved to: `./AGFN_logs/[wandb_run_name]/*.pt`. All configurable pretraining hyperparameters are located in: `.agfn/config/pretrain.yml`

### 🧠 Pretrained Weights

We provide pretrained weights for **AGFN-large**
(Trained on 6.75 million Enamine compounds, ~9.36 million parameters)

📥 [Download pretrained model](https://drive.google.com/file/d/1BmHF7gskOIKkFLn5KA0gfrdvxf2yNrY7/view?usp=sharing)

Place the downloaded checkpoint file at ./saved_models/pretrained/[pretrained_model].pt

🔜 Weights for smaller AGFN models will be released soon.

## 🧪 Finetuning

Similar to the pretraining setup, the tunable hyperparameters for finetuning are placed in:
`./src/config/finetune.yml`

The following fields of finetune.yml should be sufficient to recreate the experiments reported in our paper:

•	type: 'finetuning' / 'rtb' for vanilla finetuning with Trajectory Balance or Relative Trajectory Balance (RTB)

•	objective: property_constrained_optimization / property_targeting / property_optimization (We expect most users to be interested in property_constrained_optimization. See the Experiments section of our paper for more details.)

•	**subtype**: preserved / DRA (Dynamic Range Adjustment)

•	**task**: currently supports the following tasks out of the box:

◦	`Caco2`

◦	`LD50`

◦	`Lipophilicity`

◦	`Solubility`

◦	`BindingRate`

◦	`MicroClearance`

◦	`HepatocyteClearance`

•	**task_possible_range**: Refer to Table 8 and 9 in the appendix of our paper.

•	**pref_dir**: Preference direction; Refer to Table 9 in the appendix.

•	**task_model_path**: Path to the Maplight model for the task (e.g. ./saved_models/task_models/modelname.pt)

•	**offline_data**: True / False; whether to finetune with hybrid offline + online data or purely online

•	**offline_df_path**: Path to offline data used when offline_data == True; refer to files in ./data/task_files/ for formatting

•	**saved_model_path**: Path to pretrained AGFN prior

🔜 Support for custom tasks is coming soon!

### 🖥️ Multi-GPU Usage

By default, the script utilizes all available GPUs on the node. To restrict the number of GPUs used during training, update the world_size parameter in the `if __name__ == '__main__'` block of `ft_driver.py`.

### 🚀 Running Fine-Tuning

To run fine-tuning:

```bash
python ./src/finetuneSrc/ft_driver.py ./src/config/finetune.yml
```

## 💊 Sampling Molecules

To sample molecules using a fine-tuned GFlowNet model, run:

```bash
python ./src/agfn/sampling.py [finetuned_model_path] [n_samples] --bs [batch_size]
```

### 🔧 Arguments:

•	finetuned_model_path: Path to your trained .pt model file.

•	n_samples: Total number of SMILES to sample.

•	--bs: (Optional) Batch size used during sampling. Default is 32.

### 📁 Output:

•	Sampled SMILES will be saved to: ./data/gfn_samples/smiles_checkpoints/

•	If n_samples > 1000, intermediate checkpoints (10%, 20%, ..., 90%) will be saved incrementally.

•	The final SMILES list (100%) is always saved as smiles_final.pkl.


### 📊 Metrics

To sample molecules from a fine-tuned model and compute evaluation metrics:

```bash
python ./src/agfn/metrics.py [finetuned_model_path] --ntrajs [#samples]
```

- `--ntrajs`: Number of trajectories/molecules to sample

> ⚠️ **Note**:  
> Hypervolume calculation is disabled by default due to slowness.  
> To enable, pass `--do_hyp_vol True`.

```bash
python ./src/agfn/metrics.py [finetuned_model_path] --ntrajs [#samples] --do_hyp_vol True
```


## 🔬 Applications

### 1. De Novo Design of Target-Specific Binders with Molecular Docking

This pipeline enables the generation of de novo molecules for a target protein, with molecular docking evaluation using **[Uni-Dock](https://github.com/dptech-corp/Uni-Dock)** (GPU-accelerated) as rewards. Docking runs **in-process** in the training session via the `unidock_tools` API.

Docking needs the `unidock` engine available in-env. The canonical **`agfn`** env already ships it (the Uni-Dock binary + the `unidock_tools` wrapper) alongside the AutoGluon stack for the BBB/solubility gates, so there is no second env to manage. If you ever need to (re)install the engine into a fresh AGFN env:

```bash
# 1. Install ONLY Uni-Dock's native libs (these are pip-stack-safe). Pinning
#    cuda-version=12.9 matches the cuda129 unidock build below.
conda install -n agfn -c conda-forge \
  "cuda-version=12.9" "libcurand>=10.3.10.19,<11" \
  "libboost>=1.86,<1.87" "libboost-devel>=1.86" "libboost-headers>=1.86" "icu>=78"

# 2. Install just the unidock BINARY with --no-deps. The conda `unidock` package
#    bundles a Python wrapper that depends on numpy/pandas/rdkit/openmm; --no-deps
#    avoids conda overwriting this env's pip-installed torch/rdkit/numpy stack.
conda install -n agfn -c conda-forge "unidock=1.1.3=cuda129_h10d1193_2" --no-deps

# 3. (Re)install the lean unidock_tools 1.1.2 wrapper that drives the binary.
#    Tag 1.1.2 avoids the openmm import the bundled 1.1.3 wrapper pulls in.
/path/to/envs/agfn/bin/pip install --force-reinstall --no-deps \
  "unidock_tools @ git+https://github.com/dptech-corp/Uni-Dock.git@1.1.2#subdirectory=unidock_tools"
```

`openbabel-wheel` and the `unidock_tools` wrapper are already part of the AGFN requirements. With the env activated, `unidock` resolves from `$CONDA_PREFIX/bin`. For notebooks that don't activate the env, the docking code prepends the running interpreter's own `$CONDA_PREFIX/bin` to `PATH` and the `agfn` Jupyter kernelspec sets it too, so `shutil.which("unidock")` still resolves. Run the docking pipeline from the `agfn` env.

### 🔧 Configuration

Update the following fields in `./config/denovo.yml`:

- **`target_name`**: Specifies the docking target. Supported targets out of the box:

  - `"5ht1b"`
  - `"fa7"`
  - `"parp1"`
  - `"jak2"`
  - `"braf"`
- **`saved_model_path`**:
  Path to the pretrained AGFN prior model.
- **`unidock_search_mode`**:
  Docking exhaustiveness: `"fast"` (default), `"balance"`, or `"detail"`.
- **`unidock_num_workers`**:
  Number of worker processes for ETKDG 3D conformer generation. `1` runs in-process; `>1` uses a multiprocessing pool.
- For a **custom target**, create an entry under the `target_grid` field.`target_grid` is a dictionary where:

  - The **key** is your custom target name.
  - The **value** is another dictionary with the docking box parameters (`receptor`, `center_x/y/z`, `size_x/y/z`).

  🔍 Refer to existing targets in `denovo.yml` for formatting examples.

  📌 Additionally, place the prepared receptor file for the custom target at: `./data/docking/\custom\_target.pdbqt`. Uni-Dock takes `.pdbqt` receptors directly; convert a `.pdb` with Uni-Dock's helper, e.g. `unidocktools proteinprep -r receptor.pdb -o receptor.pdbqt`.

Other hyperparameters can be left at their default values or customized based on your use case.

### 🚀 Running Training

```bash
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo.yml
````

> ⚠️ **Note**:
> This setup is optimized for a **2-GPU** configuration.
> For **single-GPU** setups, if you encounter docking-related CUDA errors, consider **reducing** the `training_batch_size` in `denovo.yml`.

### Sampling proceeds similarly to sampling molecules for fine-tuning.

## 📖 Citation

If you use this codebase or find it helpful in your research, please cite our paper:

```bibtex
@misc{pandey2025pretraininggenerativeflownetworks,
      title={Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation}, 
      author={Mohit Pandey and Gopeshh Subbaraj and Artem Cherkasov and Martin Ester and Emmanuel Bengio},
      year={2025},
      eprint={2503.06337},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.06337}, 
}
```

## 📬 Contact

For questions or issues, please open an issue in the GitHub repository or contact the authors listed in the paper.
