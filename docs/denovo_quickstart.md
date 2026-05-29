# Denovo UniDock Training Quickstart

Run commands from the repository root. Use the `agfn` environment — it has the
Uni-Dock stack **and** the AutoGluon inference stack needed by the BBB constraint:

```bash
conda activate agfn
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo.yml
```

`agfn-no-vina` also works for runs **without** the BBB constraint, but it has no
AutoGluon installed; with `bbb_constraint: true` it now aborts at startup with an
`ImportError` (it previously degraded silently, logging `0.0` / `nan` BBB metrics).

## 1. Default UniDock Training

Use `src/config/denovo.yml` as-is for the default UniDock reward path. The important fields are:

```yaml
finetuning:
  task: "QedxSaxDock"
  target_name: "braf"
  unidock_search_mode: "fast"
  unidock_num_workers: 1
```

`target_name` must have a matching entry under `target_grid`.

## 2. UniDock Training From A Seed Molecule

To bias generation around a seed without freezing the exact molecule, set `seed_smiles`. If `seed_scaffold` is null, AGFN starts from the Murcko scaffold of `seed_smiles`.

```yaml
finetuning:
  task: "QedxSaxDock"
  seed_smiles: "O=C(Nc1ccc(Cl)nc1)c1cncc(OCc2ccn[nH]2)n1"
  seed_scaffold: null
```

To provide the editable scaffold explicitly:

```yaml
finetuning:
  task: "QedxSaxDock"
  seed_smiles: "O=C(Nc1ccc(Cl)nc1)c1cncc(OCc2ccn[nH]2)n1"
  seed_scaffold: "O=C(Nc1ccc(Cl)nc1)c1cncc(OCc2ccn[nH]2)n1"
```

## 3. UniDock Training With Frozen Seed Atoms

Use `initial_scaffold` when every trajectory should start from the exact seed molecule and keep its original atoms/bonds fixed. Atom indices are 0-based RDKit atom indices for `initial_scaffold`.

Set exactly one of `allowed_growth_atoms` or `frozen_atoms`.

Allow expansion only at selected atoms:

```yaml
finetuning:
  task: "QedxSaxDock"
  initial_scaffold: "O=C(Nc1ccc(Cl)nc1)c1cncc(OCc2ccn[nH]2)n1"
  allowed_growth_atoms: [0, 5, 12]
  frozen_atoms: null
```

Freeze selected atoms and allow expansion from all other seed atoms:

```yaml
finetuning:
  task: "QedxSaxDock"
  initial_scaffold: "O=C(Nc1ccc(Cl)nc1)c1cncc(OCc2ccn[nH]2)n1"
  allowed_growth_atoms: null
  frozen_atoms: [1, 2, 3, 4]
```

If `initial_scaffold` is set, `seed_smiles` and `seed_scaffold` are ignored. Offline data is disabled because offline molecules cannot guarantee the frozen core is preserved.

## 4. UniDock Training With BBB Constraint

The `agfn` environment already ships the minimal AutoGluon inference stack. If you
need to recreate it (e.g. from a fresh clone of `agfn-no-vina`):

```bash
python -m pip install \
  autogluon.tabular==1.5.0 \
  catboost lightgbm xgboost
python -m pip check
```

This downgrades `pandas` (3.x → 2.3.3), `numpy`, `scipy`, and `scikit-learn` to
versions compatible with AutoGluon 1.5.0. `torch` (`2.8.0+cu129`) is **not**
affected, so the generator/docking stack is untouched.

The saved models were trained under Python 3.13 while the env runs Python 3.12;
`AutoGluonBBBScorer` loads them with `require_py_version_match=False`, which is safe
for these tree-based tabular ensembles (CatBoost/LightGBM/XGBoost).

Then add the BBB constraint fields to your denovo config:

```yaml
finetuning:
  task: "QedxSaxDock"
  bbb_constraint: true
  bbb_model_path: "./saved_models/autogluon/bbb_maccs_descriptors"
  bbb_threshold: 0.9
  bbb_fail_reward: 1.e-30
  bbb_positive_class: 1
```

With this setup, UniDock still provides the reward among molecules that pass the BBB threshold. Molecules with predicted positive-class probability below `bbb_threshold` receive `bbb_fail_reward`, which is the same numerical floor used before log-reward conversion.

You can combine the BBB constraint with a frozen seed:

```yaml
finetuning:
  task: "QedxSaxDock"
  initial_scaffold: "O=C(Nc1ccc(Cl)nc1)c1cncc(OCc2ccn[nH]2)n1"
  allowed_growth_atoms: [0, 5, 12]
  frozen_atoms: null
  bbb_constraint: true
  bbb_model_path: "./saved_models/autogluon/bbb_maccs_descriptors"
  bbb_threshold: 0.9
  bbb_fail_reward: 1.e-30
  bbb_positive_class: 1
```
