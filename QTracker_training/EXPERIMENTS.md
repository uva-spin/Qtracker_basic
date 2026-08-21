# Experiment Log

Narrative record of what was tried, why, and what happened — the reasoning and dead
ends that raw MLflow metrics don't capture. See [BRANCHES.md](BRANCHES.md) for the
current branch/status map.

MLflow tracking URI: `/project/ptgroup/spinquest/Anvesh/mlruns` (on Rivanna — not
reachable from a local machine, so run IDs below are referenced by experiment name +
`slurm_${SLURM_JOB_ID}` run name + date; fill in exact run IDs from the MLflow UI/CLI on
Rivanna when you have a terminal there).

How to add an entry: date, branch, what you tried and why, what happened, and the
MLflow run it corresponds to. Newest first.

---

## 2026-08-21 — `feat/axial-fno-pair-classifier` — curriculum rehearsal fix (not yet run)

**What:** Added rehearsal/replay to `train.py`: mix ~15% of low-phase data and ~15% of
med-phase data into the high-complexity phase's training set, instead of training the
high phase on exclusively-high data.

**Why:** The validation set spans the full low-through-high complexity range, but
high-phase training exposes the model to nothing but 34-50-background-track events for
many epochs — the model drifts away from low/med patterns it had already learned.
val_loss climbs even as train_loss keeps falling: **val_loss went 3.45 → 3.8+ across
epochs 71-80**, the same divergence pattern seen in the multi-head joint model, which is
what motivated treating this as a curriculum-forgetting problem rather than a
loss/architecture problem (see the diagnostic entry below). Rehearsal keeps low/med
gradient signal present during high-phase training without a separate rehearsal pass.
The 15% figure is not empirically tuned — it's a reasonable starting point sized to add
a meaningful rehearsal signal without dominating the high-phase batch composition.

Alternatives considered and not taken: shuffling all three phases together from the
start (rejected — prior experience shows exposing the model to the hardest events before
it can handle any of the task prevents convergence entirely, which is the whole reason a
curriculum exists in the first place); loss-weighting by phase instead of data-mixing
(more complex to implement for the same rehearsal effect that data-mixing gets more
simply).

**What happened:** No Rivanna run yet — `checkpoints/` is empty and the smoke test
passed locally today, but no training job has been submitted. Next `pair_count_fno`
run is the one to check for whether val_loss holds through the high phase this time.

**MLflow:** experiment `pair_count_fno`, run — TBD, check for the first `slurm_*` run
after 2026-08-21 09:27 (commit 9d079ec).

---

## 2026-08-21 — `feat/axial-fno-pair-classifier` — curriculum data diagnostic

**What:** Built `inspect_data.py` to inspect the multi-track curriculum ROOT files
directly (low/med/high/val) rather than trusting the documented complexity ramp,
checking: whether the `nPairs` truth branch agrees with the pair-count label actually
derived from `HitArray` occupancy, what fraction of events have out-of-range
`HitArray` values (silently dropped by `load_data_denoise`), whether track/hit density
really escalates low→med→high, and whether the validation set is genuinely uniform
across the full complexity range or concentrated at low/med.

**Why:** The Axial-FNO classifier — a ~296K-param model with *zero* segmentation loss,
about as simple as this problem gets — was showing val_loss/val_acc collapsing hard the
instant training switched to the "high" phase (34-50 background tracks/event). That's
suspicious for a model this small and this task-simple: if a tiny classifier alone falls
over on the phase transition, loss complexity probably isn't the real culprit for the
divergence previously seen in the big joint model either. Wanted to rule out a genuine
train/val distribution mismatch or data-generation bug before assuming it's a training
dynamics problem.

**What happened:** **Not yet run on Rivanna** — the script needs the ROOT files that
only exist there, so there are no actual printed numbers yet. **The rehearsal fix
above was written on the working assumption that this is curriculum-forgetting, not
loss/architecture complexity — but that assumption hasn't actually been confirmed by
this script's output. Run it before trusting that framing further.**

Separately, from working with the ground-truth branch while writing this: the true
`nPairs` branch actually ranges **0-5**, nearly uniformly distributed (~16.6% each
class) — not 0-3 as assumed. Both this classifier and the joint model's `min_perm_loss`
cap at `max_pairs=3`, silently folding true pair-counts of 3, 4, and 5 into a single "3+"
bucket. Real architecture-scope gap, separate from the curriculum question — open,
not yet addressed anywhere.

**MLflow:** diagnostic script, not a training run — no MLflow entry. Capture the printed
summary in this entry (or as a linked file) once it's actually run on Rivanna.

---

## 2026-08-18 to 2026-08-21 — `feat/independent-model-ensemble` — Viz3DCallback + persistent storage fix

**What:** Added `models/viz_3d_callback.py` (`Viz3DCallback`, f312730), which saves
three plots every 10 epochs to `checkpoints/plots/3d/`:
- **Event scatter** — one event in 3D: detector layer on x, element ID (channel) on y,
  pair index stacked on z. Background hits in gray (subsampled 25% for readability),
  truth tracks as circles colored per pair, predicted tracks as crosses in a distinct
  palette. Shows at a glance whether predictions land in the right region of the
  detector or somewhere else entirely.
- **Residual waterfall** — 3D bar chart of residual (predicted − true element ID) by
  detector layer, to surface whether error concentrates in specific layers (e.g. prior
  evaluations showed elevated ~50-channel residuals concentrated in Station 2,
  detectors 19-30).
- **Per-layer accuracy** — 3D bar chart, detector layer × (pair, charge) series ×
  exact-match accuracy, to show whether specific layers or specific pairs are
  consistently harder than others.

3D specifically because the problem's natural axes (layer, channel, pair) don't
collapse to 2D without losing a dimension that actually carries information about
*where* the model is failing.

Then fixed checkpoints and viz outputs to write to persistent project storage
(`/project/ptgroup/spinquest/Anvesh` via the `/mnt/data` apptainer bind) instead of
Rivanna scratch (721b993) — scratch has already caused problems twice (see the
git-corruption entries below) and is subject to purge policies.

**Why:** Loss curves alone don't show *where* in the detector the model is failing;
needed training artifacts to survive rather than risk loss to a scratch purge.

**What happened:** Both landed 2026-08-21 (today); not yet exercised in an actual
training run. Next `slurm_*` run under `multi_track_v2` is the first real test of both
the callback's plots and the new persistent-storage path.

**MLflow:** experiment `multi_track_v2` — check for the next run after 2026-08-21 09:52
(commit 721b993).

---

## 2026-08-18 — `feat/axial-fno-pair-classifier` — branch started, Keras autocast bug

**What:** Started the Axial-FNO Stage-A classifier prototype (bf45d18): a two-stage
classifier+router design, where Stage A predicts `n` (dimuon pair count) so a later
stage can route to a dedicated n-track segmenter, rather than making one model solve
permutation-invariance and presence-detection jointly. The classifier itself alternates
a new `FourierBlock1D` (spectral convolution along the elementID axis, a real physical
wire/channel coordinate) with the existing `AxialAttention` layer (along the categorical
detectorID axis).

**Why:** The joint model's loss (`min_perm_loss` handling permutation-invariant
matching + presence/absence of tracks all at once) is a lot to ask one training signal
to solve simultaneously. Splitting "how many tracks" from "where are they" is a bet that
each sub-problem becomes easier in isolation.

Axial-FNO specifically (over a plain CNN classifier, an MLP on pooled features, or a
U-Net++ with a classification head) because the pair-count task needs evidence
integrated across the *entire* 62×201 hit matrix — a muon track is a coherent stripe
spanning all 62 layers, which a CNN's local receptive fields can miss without going very
deep. FNO operates in frequency space and captures global correlations in
O(N log N) vs O(N²) for full attention. The axial decomposition (1D FNO along detector
layers, then along channels) keeps parameter count manageable while still covering the
full spatial extent.

**What happened:** Hit a real Keras bug same day (bc8411c). Diagnosis path: the smoke
test passed under `float32` but crashed under `mixed_float16` with `tf.complex()`
rejecting float16 inputs. Root cause — Keras's `add_weight()` defaults to
`autocast=True`, which silently downcasts stored weights to the compute dtype
(float16) inside `call()`; the FNO spectral weights are complex-valued, constructed via
`tf.complex(real, imag)`, and float16 complex isn't supported by TF's FFT ops. Fix was
two lines: `autocast=False` on both `add_weight()` calls in `FourierBlock1D`. Smoke
test was then extended to explicitly run both policies and assert the spectral weights
stay float32 regardless of global policy — float32-only testing would have hidden this
bug completely, which is exactly what happened the first time (it only showed up on
Rivanna, under the GPU run's `mixed_float16` policy).

**MLflow:** no run yet at this point — smoke-test-only stage.

---

## 2026-08-05 to 2026-08-08 — `feat/independent-model-ensemble` — ensemble eval debugging + scratch corruption

**What:** Debugged the eval path for `EnsembleTrackFinder.py` (the literal
independent-sub-models architecture, added 2026-07-15): removed a broken
`refine_hit_arrays` call (6b74ac6), added `sys.path` insert + apptainer workdir fix so
`models.layers` resolves correctly inside the container (31fe9b2, dd3c1fd), then
switched to loading `AxialAttention` via `importlib` (4cfb387) to avoid ROOT's import
corrupting the `models` namespace.

**Why:** Getting the ensemble checkpoints (from the 2026-07-15/07-22 standalone eval
SLURM script) to actually evaluate inside the apptainer container kept surfacing
environment/import issues specific to running ROOT + the `models` package together
under apptainer.

**What happened:** Fixed one issue at a time, each surfacing the next:
1. `TypeError: refine_hit_arrays() missing 2 required positional arguments` — eval
   called it as `(y_mup_pred_raw[:, p, :], X_hits)`, but the actual signature needs
   `(hit_array_mup, hit_array_mum, detectorIDs, elementIDs)` from ROOT. Fixed by
   removing the refinement block from ensemble eval entirely (6b74ac6) — only raw +
   Hungarian-matched metrics are kept there now.
2. `ModuleNotFoundError: No module named 'models.layers'` — importing ROOT has the side
   effect of corrupting `sys.modules['models']`, breaking the subsequent
   `from models.layers import AxialAttention`. `sys.path.insert` (31fe9b2) and an
   apptainer `--pwd /mnt/code` flag (dd3c1fd) were both tried and didn't fix it. Actual
   fix: load `layers.py` by absolute path via `importlib.util.spec_from_file_location`
   (4cfb387), bypassing the corrupted module namespace entirely.
3. `FileNotFoundError: '/mnt/code/models/layers.py'` — turned out `layers.py` was
   genuinely missing from the branch on Rivanna scratch; earlier training runs had only
   succeeded because a cached `.pyc` in `__pycache__/` was being used. `ls models/`
   confirmed only `data_loader.py`, `EnsembleTrackFinder.py`, `losses.py`,
   `MultiTrackFinder.py`, `__pycache__` — no `layers.py` on disk.
4. Root cause of #3: git object-database corruption on the Rivanna scratch clone.
   `git restore .` printed
   `error: unable to read sha1 file of QTracker_training/models/layers.py (ea0662f...)`;
   `git pull` failed against the same object with "unresolved deltas". Standard repair
   (`git fsck`, re-committing `layers.py` under a new SHA — e1817dc, 2026-08-08) didn't
   fully resolve it since the pack corruption meant even the new object couldn't be
   fetched cleanly. Actual fix: shallow clone
   (`git clone --depth=1 --branch feat/independent-model-ensemble`) to a fresh location
   on scratch, then `cp layers.py` manually into the corrupted working tree. This is
   the second time this scratch clone has corrupted (see
   [project_rivanna_scratch_git_corruption.md] in memory).

No commits to `EnsembleTrackFinder.py` or its eval path since 2026-08-07 — see the
entry below for why (eval numbers came in conclusively worse, so focus shifted back to
`MultiTrackFinder.py`).

**MLflow:** experiment `ensemble_track_v1` — runs from this period, if any, would be
tagged around 2026-07-15 to 2026-08-07.

---

## 2026-07-15 to 2026-07-22 — `feat/independent-model-ensemble` — independent-model ensemble added

**What:** Added `models/EnsembleTrackFinder.py` (aeeb416): `n_models` fully independent
single-pair U-Net++ sub-models (own weights each, not a shared backbone), combined via
an `EnsembleWrapper` that averages the denoised output and stacks per-model segmentation
predictions. Loss (`ensemble_seg_loss`) is a permutation-invariant matching term (same
idea as `min_perm_loss`, applied across model outputs instead of slots) plus a
diversity penalty that discourages sub-models from converging on the same track. Also
added an ensemble smoke test + data summary utility same day (296864c), and a standalone
eval SLURM script for ensemble checkpoints a week later (7c24752).

**Why:** Hypothesis: 3 independent full-capacity models, each responsible for one pair,
would outperform a single model splitting its capacity across 3 shared output heads —
each sub-model could specialize, and predictions would be more diverse / less
correlated. The diversity penalty (λ=0.05, cosine-similarity-style overlap between
sub-model softmax outputs) was added specifically to stop all three sub-models from
collapsing onto the same prediction. Sub-model capacity was halved (`base=32` vs the
joint model's `base=64`) to keep total parameter count roughly comparable to the
multi-head model.

**What happened — resolved, not just parked:** Eval-path debugging continued through
early August (see entry above); once it actually ran, results were **conclusively worse
on every metric**: 18%/12% accuracy vs the multi-head model's 49%/46%, and 34/27-channel
residual vs 27/19-channel. The capacity reduction plus the diversity penalty were
counterproductive together — halved-capacity sub-models couldn't match single-pair
accuracy on their own, and actively penalizing prediction overlap fought reconstruction
quality rather than helping it. No commits since 2026-08-07 because of this result, not
because it's queued behind something else — this is presumably what gives the branch
its name, even though the two-stage FNO classifier is now the active direction and
`MultiTrackFinder.py` (the pre-existing joint model, inherited from
`AnveshTrackFinderWork`) is the file still receiving commits. Worth revisiting only with
full-capacity sub-models and a softer diversity penalty — which would need
significantly more GPU memory/compute per job than this attempt used.

**MLflow:** experiment `ensemble_track_v1`.

---

## Earlier history (pre-2026-07-15, on `AnveshTrackFinderWork` / shared ancestor)

Not itemized commit-by-commit — 120 commits on `AnveshTrackFinderWork` ahead of `main`.
Two decisions from this period worth keeping:

- **`min_perm_loss`** was added specifically because slot-to-slot cross-entropy was
  punishing *correct* predictions in the *wrong slot* — a model that found all three
  pairs correctly but assigned them to different slots than the ground-truth ordering
  would score near-zero accuracy and high loss despite being right. The fix enumerates
  every permutation of the `max_pairs` slots, computes loss under each, and takes the
  per-event minimum. Hungarian matching at eval time was added for the same underlying
  reason — without it, eval numbers were artificially deflated and not comparable to
  any single-track baseline.
- The **low/med/high curriculum** was carried over directly from the single-track
  `TrackFinder.py` infrastructure, which already had it — same three ROOT files, same
  phase boundaries, same LR schedule, extended to multi-track with no structural
  change. The curriculum-forgetting issue found later (2026-08-21, on the FNO branch)
  is sharper for multi-track than it apparently was for single-track — plausibly
  because finding 3 pairs among up to 50 background tracks is combinatorially harder,
  so the model hasn't fully saturated low/med before the high-phase transition hits.

Worth backfilling further only if a specific past decision from this period becomes
relevant again — check `git log --oneline main..AnveshTrackFinderWork` for the full
list rather than re-deriving from memory.

**MLflow:** experiment `multi_track_v2` (created in commit bf9beaf, "new MLflow
experiment multi_track_v2 for improved training run").
