# Vina-GPU 255 failure cascade — analysis

Investigation of the bug where `python ./src/apps/docking/denovo/denovo_driver.py
./src/config/denovo.yml` runs normally for a few iterations, then hits
`Vina failed with return code 255` and from then on returns
affinity = 0 for every generated molecule.

Every claim below is followed by the exact file:line range that backs it.

---

## 1. The call path: how a docking batch is run

The training loop samples molecules and asks for a task reward:

- `src/iterators/samp_iter_finetune.py:246-255` — every iteration: serialises
  the online `mols` to `{gfn_samples_path}/sampled_mols.pkl`, computes
  non-affinity rewards via `self.reward.molecular_rewards(mols)`, then calls
  `self.reward.task_reward(self.hps.task, mols)` (line 255, since
  `'QedxSaxDock'` is not in `task_model_reward_funcs` defined at
  `samp_iter_finetune.py:61-69`).

The task reward shells out to a fresh Python subprocess per iteration:

- `src/apps/docking/denovo/reward_dock.py:22-39` — `task_reward` builds
  `vina_docking_cmd = ["python", "./src/apps/docking/gpuvina.py",
  self.hps.target_name, self.gfn_samples_path, self.vina_path]` (line 33) and
  invokes it with `subprocess.run(vina_docking_cmd, check=True)` (line 34).

The subprocess is `gpuvina.py` run as `__main__`:

- `src/apps/docking/gpuvina.py:467-484` — instantiates
  `QuickVina2GPU(vina_path=..., target=...)` (line 474), loads
  `sampled_mols.pkl`, converts to SMILES, calls
  `vina.calculate_rewards(smiles_list)` (line 479), writes the result back
  to disk and exits.

**Consequence:** every docking call is a brand-new Python process with its
own `tempfile.mkdtemp()` input directory (next section), its own OpenCL
context, and no shared in-memory state with prior subprocesses.

---

## 2. Per-invocation isolation rules out stateful corruption of Vina itself

Each subprocess creates a fresh tempdir in `__init__`:

- `src/apps/docking/gpuvina.py:279-284` — `if input_dir is None: input_dir =
  tempfile.mkdtemp()` (line 280), then `self.out_dir = input_dir + "_out"`
  (line 284). `mkdtemp()` returns a unique path that is never reused.

The current-batch tempdir is cleared at the start of every batch:

- `src/apps/docking/gpuvina.py:313-318` —
  `_clear_dir_contents(self.input_dir)` and `_clear_dir_contents(self.out_dir)`.
- `src/apps/docking/gpuvina.py:131-139` — `_clear_dir_contents` only touches
  the path it was handed; it does **not** clean up tempdirs created by prior
  subprocesses.

**Consequence:** the persistence of the failure cannot live inside the docking
subprocess. Each call is a fresh process. If state survives across iterations,
it lives in (a) the parent GFN process, (b) the GPU, or (c) the filesystem.

---

## 3. How a Vina-GPU failure becomes "all zeros"

When Vina-GPU fails, no `_out` dir is created and the parser falls through:

- `src/apps/docking/gpuvina.py:412-415` — `_check_outputs` returns False if
  `self.out_dir` does not exist.
- `src/apps/docking/gpuvina.py:427-431` — in `calculate_rewards`:
  ```python
  if self._check_outputs():
      affinties = self._parse_results()
  else:
      affinties = [0.0] * self.batch_size
  ```
  Every molecule in the batch gets affinity 0.0.

Affinities are then rescaled into rewards:

- `src/apps/docking/gpuvina.py:433-435` —
  `rewards = (affinties + self.reward_scale_min) /
  (self.reward_scale_min + self.reward_scale_max) - 1`.
- `src/apps/docking/gpuvina.py:240-241, 273-274` — defaults
  `reward_scale_max = -1.0`, `reward_scale_min = -10.0`, and the `__main__`
  call at line 474 does not override them.

With affinity = 0 for every molecule, every reward collapses to the **same
constant**: `(0 + (-10)) / ((-10) + (-1)) - 1 = -0.0909…`. No
per-molecule discrimination remains.

The constant reward then propagates back through:

- `src/apps/docking/denovo/reward_dock.py:35-39` — `task_reward` reads the
  `_docked` file, returns `outs[2]` (the constant reward array) as
  `flat_rewards_task` and `outs[1]` (the zero affinities) as
  `true_task_score`.
- `src/iterators/samp_iter_finetune.py:255-259` —
  `normalized_task_rew, true_task_score = self.reward.task_reward(...)`,
  then `flat_rewards = rew * normalized_task_rew` (line 259, since
  `task_rewards_only: False` in `src/config/denovo.yml:58`).

So the affinity term degenerates to a constant multiplier over the batch.

---

## 4. Why the policy cannot recover from the constant reward

The non-affinity reward terms keep producing per-molecule gradients:

- `src/agfn/reward.py:394-450` — `_compute_flat_rewards` iterates over
  every property in `self.cond_range` (qed, sas, num_rings, tpsa,
  zinc-radius via `searchAtomEnvironments_fraction` at line 439, etc.) and
  appends a per-molecule flat reward for each.
- `src/agfn/reward.py:452-473` — `molecular_rewards` aggregates them
  (config `reward_aggergation: "mul"` per
  `src/config/denovo.yml:75`, hitting line 461-462).

So once the affinity factor flattens, the *only* gradient signal left is from
QED, SAS, zinc-radius, num_rings, tpsa. None of those favour Vina-friendly
chemistry — they push toward QED-pretty, drug-like-but-arbitrarily-shaped
structures, which is the same direction that produced the macrocycle problem
in the first place (see `ca09fc0`'s commit message: "the probability of [a
macrocyclic mol per batch] climbs toward 1 across training").

**Consequence:** the policy continues drifting *deeper* into Vina-hostile
chemistry while getting zero corrective feedback. Within a few iterations
most or all generated mols can't be successfully converted to valid pdbqts,
and the input dir ends up empty.

---

## 5. Empirical evidence of policy collapse: leaked tempdirs in `/tmp`

`/tmp` accumulates leaked input/out dirs because `_teardown` is never called
from `calculate_rewards`:

- `src/apps/docking/gpuvina.py:346-356` — `_teardown` is defined but not
  invoked.
- `src/apps/docking/gpuvina.py:417-442` — `calculate_rewards` does not call
  `_teardown` (compare with `dock_mols` at line 444, which also does not).

This is incidentally useful for debugging: the exact pdbqt files Vina-GPU saw
on failed batches are still on disk. Two patterns dominate among failed
batches (those with `input_*` files but no `_out` directory):

- **Pattern A — input dir contains only PyTorch artifacts, no pdbqts.**
  Most failed leaked dirs (e.g. `/tmp/tmp_q5ly3ha`) contain *only*
  `__pycache__/` and `_remote_module_non_scriptable.py`. Re-running
  Vina-GPU on such a dir reproduces the failure, with stderr:
  > `Parse error on line 0 in file ".../__pycache__": No atoms in the ligand`
  > `Parse error on line 1 in file ".../_remote_module_non_scriptable.py": Unknown or inappropriate tag`
  > `No valid ligands in the input directory`

  This is the normal end-state of policy collapse: every ligand fails the
  conformer or atom-type check, the input dir is empty of pdbqts, Vina-GPU
  exits non-zero on an empty batch. (The PyTorch scratch files are a
  *separate* polluting issue — see §7 — that compounds the same outcome.)

- **Pattern B — input dir contains valid pdbqts and Vina-GPU still failed.**
  One example: `/tmp/tmp_k259cq_` holds 8 input files, each passes the
  `_pdbqt_atom_types_valid` check
  (`src/apps/docking/gpuvina.py:118-128`), and each has a normal
  branch/atom count (5–11 BRANCH lines, 25–33 atoms). Re-running
  Vina-GPU on this directory **standalone, after the training session
  ended**, returns code 0 with valid affinities for all 8 molecules
  (-7.x to -8.x kcal/mol). The failure at training-time was therefore
  environmental, not ligand-content.

The most plausible candidate for the environmental failure is GPU/OpenCL
contention with the parent GFN process — the existing source comment at
`src/apps/docking/gpuvina.py:18` already flags this:
```python
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Enable this with correct ID to let docking run on seperate GPU. Enables larger batch size training
```

---

## 6. Why the `ca09fc0` fix doesn't break the cascade

The commit added an atom-type validator that drops bad ligands instead of
letting one bad ligand kill the batch:

- `src/apps/docking/gpuvina.py:104-115` — `_VALID_AD_ATOM_TYPES` set.
- `src/apps/docking/gpuvina.py:118-128` — `_pdbqt_atom_types_valid`.
- `src/apps/docking/gpuvina.py:339-344` — files failing validation are
  removed before Vina-GPU is invoked.

This works correctly for the case in the commit message (one macrocycle with
`CG0`/`G0` pseudo-atoms among otherwise-fine ligands). But it does not handle
either of the two persistence patterns above:

- **Pattern A (empty input dir).** The validator did its job *too* well —
  every ligand was dropped, leaving nothing for Vina-GPU to dock. Vina-GPU
  returns nonzero. The protective "drop the bad one and keep going" branch
  in `_write_pdbqt_files` never runs because the issue is no longer
  *one* bad ligand.

- **Pattern B (valid pdbqts, environmental failure).** The validator passes
  the files cleanly; Vina-GPU still exits nonzero for reasons unrelated to
  ligand content. The fix has no leverage here.

In both cases `_check_outputs` returns False and the reward signal collapses
exactly as described in §3.

---

## 7. The diagnostic blocker

Right now there is no way to tell Pattern A from Pattern B from the training
log because `_run_vina` discards Vina-GPU's stdout on failure:

- `src/apps/docking/gpuvina.py:358-373`:
  ```python
  result = subprocess.run(
      [self.vina_path, "--config", os.path.join(self.input_dir, "../config.txt")],
      capture_output=True, text=True,
      cwd=os.path.dirname(self.vina_path),
  )
  if self.print_time:
      print(result.stdout.split("\n")[-2])
  if self.print_logs:
      print(result.stdout.split("\n"))

  if result.returncode != 0:
      print(f"Vina failed with return code {result.returncode}")
      print(result.stderr)
      return False
  ```

  When `returncode != 0`, only `result.stderr` is printed. Vina-GPU writes
  its real error messages — `"No valid ligands in the input directory"`,
  ligand parse errors, OpenCL build/alloc failures — to **stdout**, which
  is being thrown away.

That is why the user-facing log just says `"Vina failed with return code
255"` and nothing else: the explanation was on stdout the whole time.

---

## 8. The polluting PyTorch artifacts (open thread)

Separately, many failed tempdirs contain `__pycache__/` and
`_remote_module_non_scriptable.py`. That file name is generated by
`torch/distributed/nn/api/remote_module.py` via
`os.path.join(tempfile.gettempdir(), ...)`. The mechanism by which it ends
up inside a `tempfile.mkdtemp()`-created subdirectory rather than at `/tmp/`
itself has not been pinned down in this investigation, but the empirical
behaviour matters: if it is present at the moment Vina-GPU runs, Vina-GPU
sees a non-pdbqt file in `ligand_directory` and aborts the whole batch.

This is a compounding factor on top of policy collapse, not the primary
cause. It is listed here so it isn't lost.

---

## Summary

- The first 255 has many possible triggers (macrocycle pseudo-atoms not yet
  caught by the validator; GPU/OpenCL contention with the parent process; a
  polluting non-pdbqt file in the input dir).
- Once it happens, `calculate_rewards` returns 16 zeros
  (`gpuvina.py:427-431`), which after rescaling
  (`gpuvina.py:433-435`) gives an **identical** reward for every molecule
  in the batch.
- A constant per-molecule reward kills the affinity gradient. The remaining
  reward terms (QED, SAS, zinc-radius, num_rings, tpsa) at
  `reward.py:394-450` keep pushing the policy in directions that don't
  recover Vina-friendly chemistry.
- The policy collapses into a regime where most/all generated ligands fail
  validation or conformer generation, the input dir ends up empty (Pattern A
  in §5), Vina-GPU returns 255 for "no valid ligands", rewards collapse
  again, and the cycle is self-sustaining.

**Immediate diagnostic fix:** print `result.stdout` (not just `result.stderr`)
when `result.returncode != 0` at `gpuvina.py:370-373`. That single change
makes the difference between Pattern A and Pattern B visible per iteration.
