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

## 2026-09-01 — `feat/independent-model-ensemble` — MultiTrackFinder eval, per-pair breakdown

**What:** Full evaluation of the shared-backbone/independent-heads architecture
(`MultiTrackFinder.py` — denoising backbone suppresses background first, then a
segmentation backbone with axial attention outputs hit probabilities for each charge and
pair slot simultaneously), trained with the 0-50 background-track curriculum, 60 epochs
on 1x A100, on the 53K-event validation set. Best val_loss 2.346.

Results:
- Primary pair: 99.4% recall, 74.1% precision, F1 84.9%
- Hit accuracy: ~43% exact, ~52-54% within ±2 channels, mean residual 7-14 ch
- Pair 3: 25% recall, mean residual 20-33 ch
- Pair 4: 0.7% recall

(Pair-3/pair-4 breakdown implies this run used `max_pairs=5`, not the `max_pairs=3` used
elsewhere on this branch's `train_multi.slurm` — worth confirming which commit/config
produced this when the exact MLflow run is located.)

**Why:** Checking whether the joint model actually resolves pairs beyond the primary one,
now that the true `nPairs` label is known to range 0-5 (see the 2026-08-21 diagnostic
entry below) rather than the previously-assumed 0-3.

**What happened:** Primary-pair detection is strong (F1 84.9%) and hit-level accuracy
on detected pairs is consistent with prior runs (~43% exact / mean residual 7-14 ch vs.
the 42.2%/42.7% exact-accuracy baseline in `CLAUDE.md`). But pairs 3+ are badly
underdetected — recall falls off a cliff (25% → 0.7%) rather than degrading gradually,
consistent with under-representation in training data (these are the rarest slots:
recall on more-rarely-populated pairs was already known to be lower even under the old
max_pairs=3 framing — see "Pair slot ordering" in `CLAUDE.md`). Oversampling pairs 3+
identified as the next step; momentum integration and QMetric extension are planned
once hit accuracy improves further.

**MLflow:** experiment `multi_track_v2` — exact run ID/commit not yet identified; locate
by best val_loss ≈ 2.346 and `max_pairs=5`.

---

## 2026-08-22 to 2026-08-30 — `feat/axial-fno-pair-classifier` — rehearsal validated, capacity and label-scope ruled out

**What:** Four follow-on experiments on the classifier, run in sequence:

1. **Rehearsal fix validated** (job `rehearsal_v1`): confirmed the fix below actually
   works. val_acc held at 0.51-0.52 and *improved* to 0.522 (val_loss 1.140) through the
   entire high phase — no collapse, versus the pre-fix run's val_acc crashing 0.52→0.40
   the instant the high phase started. Replicated on a rerun (`max_pairs3_v2`, different
   job): val_acc 0.524/val_loss 1.146 — small transient dip right at the med→high
   transition (0.527→0.514) that fully recovers, consistent with GPU non-determinism
   rather than a real regression.
2. **Widened model tested capacity hypothesis, result: capacity was not the bottleneck.**
   `base=64, k_max=64, num_heads=8` (2.2M params, ~7.5x bigger) overfit hard on the clean
   low-phase data (train acc 83%, val_loss ballooned 1.29→4.72) and, despite the same 15%
   rehearsal mix, *also* overfit through the high phase this time (train acc climbing
   0.51→0.57 while val_loss climbed 1.18→1.64) — a genuinely different failure mode
   (overfitting) from the small model's clean improvement. Best checkpoint (val_loss
   1.181, val_acc 0.509 via EarlyStopping) was worse than the small model's 0.522.
   Reverted to the small config (`base=32, k_max=32, num_heads=4`).
3. **max_pairs=5 tested the label-fold hypothesis, result: not hiding real accuracy.**
   Re-derived labels over the true 0-5 range (6 classes, confirmed uniform ~16.6% each
   in the actual data) instead of folding 3/4/5 into one "3+" bucket. Raw val_acc dropped
   to 34.8% — looks worse, but normalized against each task's random baseline
   (6-way: 16.7%, 4-way: 25%), both land at ~2.08-2.09x chance. The model's actual
   discriminative skill didn't change; max_pairs=3's "3+" bucket wasn't hiding easy
   accuracy the fold was suppressing.
4. **Reverted to max_pairs=3 as final scope**, not because of the accuracy finding above
   but because the router's downstream segmenters (`min_perm_loss`) only support
   `1 <= n_pairs <= 3` — a finer 0-5 classification is precision the router can't act on
   regardless of the classifier's real capability. `max_pairs=3, base=32, k_max=32,
   replay_fraction=0.15` (the validated small-model + rehearsal config) is the config
   going forward.

**Why:** After the rehearsal fix worked, wanted to know whether the ~52% ceiling was a
capacity problem (try a bigger model) or a labeling problem (the folded "3+" bucket)
before concluding the classifier was done. Both came back negative — useful in that they
close off two plausible-looking levers rather than leaving them as open guesses.

**What happened:** See above — rehearsal fix holds under replication; the ~52% (4-way,
correctly-scoped) ceiling appears to be a real property of this classification task given
the current architecture and input representation (global-average-pooled classification
over a merged noisy occupancy grid), not a capacity or label-scope artifact. Not yet
explored: whether a pooling mechanism that preserves more spatial structure than GAP
would move the ceiling.

**MLflow:** experiment `pair_count_fno`, runs `rehearsal_v1_*`, `widened_v1_*` (two —
one killed by the original 12h walltime, one completed at 24h), `max_pairs5_v1_*`,
`max_pairs3_v2_*`.

---

## 2026-08-30 — `feat/independent-model-ensemble` — rehearsal fix ported to MultiTrackFinder

**What:** Ported the same rehearsal/replay fix (5727dbf) from the classifier to
`MultiTrackFinder.py`'s curriculum loop: `_sample_replay()` now samples matching
`(X, X_clean, y)` triples from the low and med phases before they're deleted, and mixes
~15% of each (`--replay_fraction`, same default as the classifier) into the high phase's
training set alongside the original high-complexity data.

**Why:** This is the actual production model with the real project-level divergence
problem the classifier work was undertaken to understand cheaply first. Everything
learned on the classifier (curriculum-forgetting diagnosis, rehearsal as the fix,
capacity and label-scope ruled out as confounds) pointed back to this as the next step.

**What happened:** Validated with a local integration smoke test only (stubbed
`load_data_denoise`, tiny synthetic data, `denoise_base=8/base=8`, 6 epochs across
low/med/high) — confirmed the "High phase rehearsal: mixing in N low/med events" path
runs end-to-end with correct array shapes through all three phases and saves a model.
**Not yet run on Rivanna with real curriculum data** — that's the real test of whether
this fixes the joint model's documented divergence the way it fixed the classifier's.

**MLflow:** experiment `multi_track_v2` — no run yet as of this entry; next `slurm_*`
run after commit 5727dbf is the one to check.

---

## 2026-08-21 — `feat/axial-fno-pair-classifier` — curriculum rehearsal fix (validated — see 2026-08-22 to 2026-08-30 entry above)

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

## 2026-08-21 — `feat/axial-fno-pair-classifier` — curriculum data diagnostic (confirmed)

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

**What happened:** Run on Rivanna against all four ROOT files — confirmed the
curriculum-forgetting framing rather than a data bug. Low/med/high share identical
`nPairs` truth distribution and event count (133,243 each) — same underlying signal
events, only background injection differs, exactly as `messy_gen.py`'s design implies.
Track/hit density escalates cleanly (mean tracks/event 14.2→31.2→48.2 across
low/med/high, matching the documented 0-16/17-33/34-50 ranges). Zero events dropped by
`load_data_denoise`'s range check. Critically, `mc_events_val.root`'s track count is
genuinely `~Uniform(0,50)` (mean 30.0, std 15.1 — matching the theoretical mean/std of a
uniform draw almost exactly), confirming val is representative across the full
complexity range the whole time, not concentrated at low/med. That's what makes the
high-phase-only training a genuine curriculum-forgetting setup: the model spends the
final phase seeing exclusively the hardest third of what val actually contains.

Separately, from working with the ground-truth branch while writing this: the true
`nPairs` branch actually ranges **0-5**, nearly uniformly distributed (~16.6% each
class) — not 0-3 as assumed. Both this classifier and the joint model's `min_perm_loss`
cap at `max_pairs=3`, silently folding true pair-counts of 3, 4, and 5 into a single "3+"
bucket. Addressed on the classifier in the 2026-08-22 to 2026-08-30 entry below
(tested max_pairs=5, found it doesn't hide real accuracy, reverted to 3 anyway since
that's what the router can act on) — the joint model still caps at max_pairs=3 in
`train_multi.slurm`, unchanged; the 2026-09-01 entry above (max_pairs=5 eval) used a
different/uncommitted config.

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

**What happened:** Both landed 2026-08-21. Exercised same day in job `18750604`
(`train_multi.slurm`): confirmed checkpoints correctly saved to
`/mnt/data/checkpoints/multi_track_finder_best.keras` (not scratch) through the first
few epochs observed. Full curriculum run (through the high phase) not confirmed
complete as of this doc.

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
