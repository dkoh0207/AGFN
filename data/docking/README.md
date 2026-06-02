# Docking targets

Prepared receptors (`.pdbqt`) and docking boxes for the five example targets used by the de novo
docking finetuning task (`task: QedxSaxDock`). Docking is performed by the **Uni-Dock**
GPU backend (`src/apps/docking/unidock.py`), run **in-process** during training via the
`unidock_tools` API. The reward is the existing `molecular_rewards` composite
(`tpsa·num_rings·sas·qed`) multiplied by the Uni-Dock-derived affinity reward.

## Targets

Each receptor is paired with a search box (center + size, in Å) defined under `target_grid` in
the configs. The boxes follow the standard de novo docking benchmark used in the GFlowNet /
RxnFlow molecular-docking literature.

| Target  | Protein                                              | Receptor            | Box center (x, y, z)        | Box size (x, y, z)        |
|---------|------------------------------------------------------|---------------------|-----------------------------|---------------------------|
| `fa7`   | Coagulation factor VIIa (serine protease)            | `fa7.pdbqt`         | (10.131, 41.879, 32.097)    | (20.673, 20.198, 21.362)  |
| `parp1` | Poly(ADP-ribose) polymerase 1                        | `parp1.pdbqt`       | (26.413, 11.282, 27.238)    | (18.521, 17.479, 19.995)  |
| `5ht1b` | 5-hydroxytryptamine (serotonin) receptor 1B (GPCR)   | `5ht1b.pdbqt`       | (-26.602, 5.277, 17.898)    | (22.5, 22.5, 22.5)        |
| `jak2`  | Janus kinase 2 (tyrosine kinase)                     | `jak2.pdbqt`        | (114.758, 65.496, 11.345)   | (19.033, 17.929, 20.283)  |
| `braf`  | B-Raf proto-oncogene serine/threonine kinase         | `braf.pdbqt`        | (84.194, 6.949, -7.081)     | (22.032, 19.211, 14.106)  |

Lower (more negative) affinities are better; affinities are clamped to ≤ 0 and scaled to a
reward via `(affinity - 10) / -11 - 1` (an affinity of −10 → reward 0, −1 → reward −1), matching
`UniDockGPU` defaults (`reward_scale_min=-10`, `reward_scale_max=-1`).

## Running de novo docking for a target

Prerequisite: the `agfn` conda env with the `unidock` engine installed in-env (see the
"De Novo Design" section of the repo `README.md`). Activate it so `unidock` is on `PATH`:

```bash
conda activate agfn
```

There is one ready-to-run config per target in `src/config/` (`denovo_<target>.yml`). From the
repo root:

```bash
# fa7
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo_fa7.yml
# parp1
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo_parp1.yml
# 5ht1b
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo_5ht1b.yml
# jak2
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo_jak2.yml
# braf
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo_braf.yml
```

Each config differs only in `target_name`; they share one `target_grid` (all five boxes above)
and identical training hyperparameters. To dock a different target with the generic
`denovo.yml`, just change its `target_name` to one of the keys above.

### Useful knobs (in each config)

- `unidock_search_mode`: `fast` (default) | `balance` | `detail` — docking exhaustiveness vs. speed.
- `unidock_num_workers`: `1` runs ETKDG conformer generation in-process; `>1` uses a
  multiprocessing pool.
- `sampling_batch_size` / `training_batch_size`: reduce these if you hit GPU OOM (each iteration
  docks `sampling_batch_size` molecules).
- `saved_model_path`: the pretrained AGFN prior checkpoint to finetune from.

Metrics (including `avg_task_score` = mean affinity and `avg_task_reward`) are written per
iteration to `<log_dir>/<run_name>/metrics.csv`.

## Adding a custom target

1. Place the prepared receptor at `./data/docking/<name>.pdbqt`. Uni-Dock takes `.pdbqt`
   directly; convert a `.pdb` with `unidocktools proteinprep -r receptor.pdb -o receptor.pdbqt`.
2. Add a `target_grid` entry with `receptor` + `center_x/y/z` + `size_x/y/z` (copy an existing
   entry as a template) and set `target_name` to your new key.
