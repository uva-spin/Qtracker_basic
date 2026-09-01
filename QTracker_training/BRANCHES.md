# Branch Map

Status snapshot of active branches, reconstructed from `git log` / `git diff` against
merge-base, not from memory. Regenerate the "ahead of merge-base" counts with:

```
git merge-base <branch> feat/independent-model-ensemble
git log --oneline <merge-base>..<branch>
```

Last verified: 2026-09-01.

## Active

### `feat/independent-model-ensemble` — current active approach
**Status:** actively developed (last commit 2026-08-21, same day as this doc).

The tip of the active work. Contains **two coexisting model implementations** — worth
being deliberate about which one "the current approach" means:

- **`models/MultiTrackFinder.py`** — joint multi-head U-Net++: one shared denoising
  backbone + one segmentation backbone whose output is reshaped into `max_pairs=3`
  slots, trained with a permutation-invariant `min_perm_loss` (Hungarian-style matching
  over slot order) and a low/med/high curriculum. This is the file that has kept
  receiving commits most recently — `min_perm_loss` (63a9501, 3429f21, 4a7625a),
  `Viz3DCallback` wiring (f312730), and the persistent-storage fix (721b993), both from
  today (2026-08-21).
- **`models/EnsembleTrackFinder.py`** — a *literally* different architecture added later
  (aeeb416, 2026-07-15): `n_models` fully independent single-pair U-Net++ sub-models
  (separate, halved-capacity weights each), combined via an `EnsembleWrapper`, trained
  with a permutation loss plus a diversity penalty (λ=0.05) pushing sub-models toward
  predicting different tracks. This is presumably what the branch name refers to.
  **Resolved, not just parked:** once the eval path was debugged (through 2026-08-07 —
  importlib fix for `AxialAttention` under ROOT's namespace corruption, apptainer
  path/workdir fixes, a broken `refine_hit_arrays` call removed), eval numbers came in
  conclusively worse than the joint model on every metric (18%/12% accuracy vs 49%/46%,
  34/27-channel residual vs 27/19-channel) — halved capacity plus the diversity penalty
  fighting reconstruction quality together made it a worse tradeoff, not just an
  under-iterated one. No commits since 2026-08-07 for that reason. Full detail in
  [EXPERIMENTS.md](EXPERIMENTS.md). Would only be worth revisiting with full-capacity
  sub-models and a softer diversity penalty — meaningfully more GPU memory/compute per
  job than this attempt used.

Recent notable fixes on this branch: Rivanna scratch git corruption recovery (e1817dc —
second occurrence of this issue; fresh clone fixed it both times), apptainer
`sys.path`/workdir fixes so `models.layers` resolves inside the container
(31fe9b2, dd3c1fd), and the fix to write checkpoints/viz outputs to
`/project/ptgroup/spinquest/Anvesh` (via `/mnt/data`) instead of Rivanna scratch, which
risked being purged (721b993). Most recent: (5727dbf, 2026-08-30) ported the curriculum
rehearsal fix validated on `feat/axial-fno-pair-classifier` into `MultiTrackFinder.py`'s
curriculum loop — mixes ~15% of low/med data into the high phase. Validated locally via
integration smoke test only; not yet run on Rivanna with real data. See EXPERIMENTS.md.

MLflow: experiment `multi_track_v2` (MultiTrackFinder.py path) / `ensemble_track_v1`
(EnsembleTrackFinder.py path), tracking URI `/project/ptgroup/spinquest/Anvesh/mlruns`,
run names `slurm_${SLURM_JOB_ID}`.

### `feat/axial-fno-pair-classifier` — Stage-A classifier, config settled
**Status:** active (last commit 2026-08-30). Branches off
`feat/independent-model-ensemble` at e1817dc, then merges it back in at 18eb2e3
(2026-08-18) to pick up shared infra, so it tracks the ensemble branch's shared code.

Isolated under `models/experiments/pair_count_fno/`. Builds "Stage A" of a proposed
two-stage classifier+router design: predict `n` (number of dimuon pairs in an event),
then route to a dedicated n-track segmenter — sidestepping the permutation-invariant +
presence-detection loss the joint model currently solves in one shot. Classifier
architecture is new to this codebase: `FourierBlock1D` (spectral convolution along the
elementID axis — a real physical wire/channel coordinate) alternated with the existing
`AxialAttention` layer (along the categorical detectorID axis).

**Settled config** (validated, see EXPERIMENTS.md for the full trail): `max_pairs=3`,
`base=32, fno_depth=4, k_max=32, num_heads=4` (~296K params), `replay_fraction=0.15`
rehearsal into the high phase. Gets val_acc≈0.52-0.53 (4-way), confirmed stable across
two independent runs. Bigger (`base=64, k_max=64`, 2.2M params) was tried and made
things worse (overfits at every phase transition); `max_pairs=5` was tried and confirmed
the "3+" bucket wasn't hiding accuracy, but 3 is correct anyway since the router's
`min_perm_loss` segmenters only support up to 3 pairs.

15 commits ahead of the ensemble branch's merge-base — key ones:
- bf45d18 — initial Axial-FNO classifier prototype
- bc8411c — fix: Keras `add_weight()` defaults `autocast=True`, silently downcasting
  float32 spectral weights to float16 under `mixed_precision`, which broke
  `tf.complex()`. Only showed up on Rivanna (GPU, `mixed_float16`), not in local smoke
  tests (float32) — smoke test extended to run both policies.
- 18eb2e3 — merge in ensemble-branch infra
- 45a3429 — `inspect_data.py` diagnostic (curriculum-forgetting finding)
- 9d079ec — rehearsal fix in `train.py` — **validated**, see EXPERIMENTS.md
- 7a91978, 18b2ad5 — confusion-matrix + hit-matrix-heatmap visualization callbacks,
  GIF assembly, run naming, persistent-storage output paths
- 32d1f4d, b768534 — widened-model capacity experiment (result: worse, reverted)
- bae52c8, 5ef3b3b — `--reverse_curriculum` flag (built, not yet tried)
- b4abdce, 2f38dd2 — max_pairs 5 experiment then revert to 3 (final scope)

Untried: `--reverse_curriculum` (train high→med→low instead) — built specifically to
address the widened model's low-phase overfitting, but that config was abandoned before
it got used.

MLflow: experiment `pair_count_fno`, same tracking URI, run names `slurm_${SLURM_JOB_ID}`.

## Merged / superseded (0 commits ahead of `feat/independent-model-ensemble`)

These still exist as branch refs but are fully contained in the active branch — nothing
to reconstruct, safe to ignore or delete:

- `feat/perm-invariant-training-loss` (merge-base = tip, last commit 2026-06-17)
- `feat/simultaneous-multi-track-eval` (merge-base = tip, last commit 2026-06-02)

## Dormant / historical

- `TrackFinder` — old single-model U-Net approach, last commit 2025-08-07 by a
  different author (Donghwa Shin). Predates the multi-track work; 76 commits ahead of
  the common ancestor with the active branch but on a different track entirely
  (single-track finder, distance-based loss). Not part of current work.
- `main` — ~129 commits behind the active branch as of this writing. Do not base new
  work on it; it does not reflect the current multi-track effort.
- Remote-only branches (`origin/feat/ensemble-fixed-n-track-finder`,
  `origin/feat/autoregressive-track-finder`, `origin/feat/plan-01-multi-track-loss-data`,
  `origin/QMetric-Single`, `origin/dev`, `origin/feature/multi-track-qtracker`,
  `origin/MultiTrackTrackFinder`, `origin/Data-Team`, `origin/testing`,
  `origin/Devin-Testing`) all last touched 2026-06-07 or earlier — predate the current
  ensemble/curriculum work and weren't inspected in depth for this doc. Worth a fresh
  look only if one of their names becomes relevant again (e.g. the fixed-N or
  autoregressive framing resurfaces).

## This branch (`AnveshTrackFinderWork`)

Direct ancestor of `feat/independent-model-ensemble` (11 commits behind, no divergence —
`git log AnveshTrackFinderWork..feat/independent-model-ensemble` is empty in the other
direction). Used here only as a stable home for these two tracking docs, per Anvesh's
instruction not to write them on `main` (stale) or `feat/independent-model-ensemble`
(the model branch itself, keep it focused on model code).
