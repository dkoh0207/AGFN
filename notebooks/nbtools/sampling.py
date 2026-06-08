"""Model loading + trajectory sampling — wraps the repo's own Sampler / DockingFineTuner.

These are thin wrappers: they reuse the *exact* constructors training uses, so model architecture
and checkpoint loading match, and they hide the two torch>=2.6 / fine-tuned-checkpoint gotchas
(``add_safe_globals`` and a lenient ``load_state_dict``) behind a single call. All repo imports are
lazy so this module is importable before :func:`nbtools.setup_repo` has wired up ``sys.path``.
"""

import os


def build_sampler(ckpt, initial_scaffold=None, allowed_growth_atoms=None, frozen_atoms=None,
                  seed_graph=None, save_path="./data/gfn_samples/"):
    """Construct an :class:`agfn.sampling.Sampler` from a checkpoint (frozen-core aware).

    ``initial_scaffold`` + ``allowed_growth_atoms`` / ``frozen_atoms`` seed every trajectory from a
    fixed core, exactly as training does. Returns the Sampler (``.seed_graph``, ``.ctx``,
    ``.model``, ``.graph_sampler``, ``.cond_info_cmp``, ``.device``, ``.hps``).
    """
    from agfn.sampling import Sampler

    return Sampler(
        model_load_path=ckpt,
        seed_graph=seed_graph,
        save_path=save_path,
        initial_scaffold=initial_scaffold,
        allowed_growth_atoms=allowed_growth_atoms,
        frozen_atoms=frozen_atoms,
    )


def build_finetuner(hps, conditional_range_dict, cond_prop_var, checkpoint,
                    rank=0, world_size=1, gfn_samples_path=None, lenient=True):
    """Construct a ``DockingFineTuner`` with the same constructor training uses.

    Two checkpoint-loading wrinkles are handled here so notebooks don't have to:

    * ``torch.serialization.add_safe_globals([EasyDict])`` — our checkpoints pickle an EasyDict in
      ``hps``; torch>=2.6 defaults ``weights_only=True`` and would refuse it.
    * ``lenient`` (default True) — a *fine-tuned* checkpoint is saved from the pruned, forward-only
      model and lacks the full model's backward-policy heads. FineTunerRTB builds the full model and
      ``load_state_dict``s strictly, which would error on those missing P_B heads; we temporarily
      relax it to ``strict=False`` (the heads are pruned away and never used for forward sampling).

    Returns the finetuner with ``.gfn_trainer.model`` set to eval mode.
    """
    import torch
    import torch.nn as nn
    from easydict import EasyDict
    from denovo_trainer import DockingFineTuner

    torch.serialization.add_safe_globals([EasyDict])
    if gfn_samples_path:
        os.makedirs(gfn_samples_path, exist_ok=True)

    _orig_lsd = nn.Module.load_state_dict
    if lenient:
        nn.Module.load_state_dict = lambda self, sd, strict=True, **kw: _orig_lsd(self, sd, strict=False, **kw)
    try:
        finetuner = DockingFineTuner(
            hps, conditional_range_dict, cond_prop_var,
            checkpoint, rank=rank, world_size=world_size, gfn_samples_path=gfn_samples_path,
        )
    finally:
        nn.Module.load_state_dict = _orig_lsd

    finetuner.gfn_trainer.model.train(False)
    if hasattr(finetuner.gfn_trainer, "model_prior"):
        finetuner.gfn_trainer.model_prior.train(False)
    return finetuner


def _sampler_components(obj):
    """Return ``(cond_info, model, graph_sampler, device, default_stop_prob, default_action_prob)``
    for either a Sampler (``cond_info_cmp`` / ``model``) or a DockingFineTuner (``cond_info_task`` /
    ``gfn_trainer.model``), so :func:`sample_trajectories` works with both."""
    if hasattr(obj, "gfn_trainer"):                          # DockingFineTuner
        return obj.cond_info_task, obj.gfn_trainer.model, obj.graph_sampler, obj.device, 0.0, 0.0
    # agfn.sampling.Sampler — defaults come from its hps, matching check_frozen_core.ipynb.
    return (obj.cond_info_cmp, obj.model, obj.graph_sampler, obj.device,
            obj.hps["random_stop_prob"], obj.hps["random_action_prob"])


def sample_trajectories(obj, n_samples, seed_graph=None, random_stop_action_prob=None,
                        random_action_prob=None, require_logprob=False):
    """Sample ``n_samples`` trajectories from a Sampler or DockingFineTuner and keep the valid ones.

    ``seed_graph`` pins frozen-core sampling (pass ``sampler.seed_graph``); leave ``None`` for free
    generation. ``random_stop_action_prob`` / ``random_action_prob`` default to the object's own
    settings. ``require_logprob=True`` additionally drops trajectories without ``fwd_logprob`` (the
    filter check_frozen_core.ipynb uses).
    """
    import torch

    ci, model, graph_sampler, device, def_stop, def_act = _sampler_components(obj)
    stop_prob = def_stop if random_stop_action_prob is None else random_stop_action_prob
    act_prob = def_act if random_action_prob is None else random_action_prob

    cond_info = ci.compute_cond_info_forward(n_samples)
    cond_info_enc = ci.thermometer_encoding(cond_info).to(device)
    with torch.no_grad():
        trajs = graph_sampler.sample_from_model(
            model, n_samples, cond_info_enc, device,
            random_stop_action_prob=stop_prob, random_action_prob=act_prob,
            seed_graph=seed_graph,
        )
    if require_logprob:
        return [t for t in trajs if t["is_valid"] and "fwd_logprob" in t]
    return [t for t in trajs if t["is_valid"]]


def trajectory_to_mols(trajs, ctx, final_only=True):
    """Convert trajectories to RDKit molecules via ``ctx.graph_to_mol``, dropping any that fail.

    ``final_only`` (default) returns one mol per trajectory (the final graph ``traj[-1][0]``);
    otherwise returns the list-of-lists of every renderable intermediate.
    """
    out = []
    for t in trajs:
        if final_only:
            try:
                m = ctx.graph_to_mol(t["traj"][-1][0])
            except Exception:
                m = None
            if m is not None:
                out.append(m)
        else:
            steps = []
            for g, _action in t["traj"]:
                if len(g.nodes) == 0:
                    continue
                try:
                    m = ctx.graph_to_mol(g)
                except Exception:
                    m = None
                if m is not None:
                    steps.append(m)
            out.append(steps)
    return out


def rank_batch_by_reward(finetuner, trajs, task):
    """Score + rank a sampled batch the way training does — this DOCKS the batch (needs a GPU).

    Mirrors samp_iter_finetune.py: overall reward = molecular reward * normalized docking reward,
    and ``affinity`` is the raw docking score. Returns a dict with the valid trajectories whose
    final graph renders, their ``mols``, the per-mol ``flat`` reward and ``affinity`` arrays, and
    ``order`` (indices best-first).
    """
    import numpy as np

    valid, mols = [], []
    for t in trajs:
        if not t["is_valid"]:
            continue
        try:
            m = finetuner.ctx.graph_to_mol(t["traj"][-1][0])
        except Exception:
            m = None
        if m is not None:
            valid.append(t)
            mols.append(m)
    if not mols:
        return {"valid": [], "mols": [], "flat": np.array([]), "affinity": np.array([]),
                "order": np.array([], dtype=int)}

    rew = np.asarray(finetuner.reward.molecular_rewards(mols)[2]).reshape(-1)
    normalized_task_rew, true_task_score = finetuner.reward.task_reward(task, mols)
    flat = rew * np.asarray(normalized_task_rew).reshape(-1)
    affinity = np.asarray(true_task_score).reshape(-1)
    order = np.argsort(-flat)
    return {"valid": valid, "mols": mols, "flat": flat, "affinity": affinity, "order": order}
