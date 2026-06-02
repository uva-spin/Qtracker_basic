# QMetric Results Summary

1. Current Definition of QMetric

The current Quality Metric (QMetric) is a post-TrackFinder, pre-momentum, per-track reconstruction-worthiness score

```text
one event -> one mu+ candidate track + one mu- candidate track
```

The final use case is:

```text
TrackFinder candidate tracks
-> QMetric score per candidate track
-> user-chosen threshold
-> selected tracks sent to momentum reconstruction
```

The QMetric must only use information available at runtime before momentum reconstruction. It should not use truth information or already-reconstructed momentum as input.

Runtime-allowed inputs:

```text
- candidate hit pattern
- candidate elementID pattern
- candidate driftDistance pattern, when available
- missing-hit pattern
- detector/station coverage
- event occupancy and local candidate density
- TrackFinder softmax confidence, margin, and entropy
```

Calibration-only information, not allowed as runtime QMetric input:

```text
- ground truth HitArray
- true momentum
- reconstructed momentum error
- exact hit residuals
- exact_fraction_on_truth_hits
- residual_leq_2_fraction_on_truth_hits
- mean_abs_residual_on_truth_hits
```

The candidate representation used for downstream momentum validation is:

```text
candidate_track.shape == (62, 2)
candidate_track[:, 0] = elementID
candidate_track[:, 1] = driftDistance
```

This matches the existing momentum-model convention, where the momentum model expects an input track array with 62 detector slots and two features per detector slot.

---

## 2. Main QMetric Scripts Added So Far

The work was developed as separate scripts rather than by modifying `QTracker.py`, because the current active workflow evaluates `.keras` / `.h5` models directly and `QTracker.py` is not yet the active integration point.

### Pattern Table Builder

```text
scripts/build_quality_pattern_table.py
```

Purpose:

```text
ROOT file -> TrackFinder inference -> candidate-level feature table
```

This first version generated feature-only candidate rows with:

```text
- event_id
- pair_index
- charge
- hit pattern features
- event/local occupancy features
- softmax features
- truth residual calibration columns
- qmetric_score_v0
```

### Array Pattern Table Builders

```text
scripts/build_quality_pattern_table_arrays.py
scripts/build_quality_pattern_table_arrays_v2.py
```

Purpose:

```text
ROOT file -> TrackFinder inference -> candidate-level feature table
+ save candidate arrays needed for momentum evaluation
```

These versions add columns such as:

```text
raw_elem_00 ... raw_elem_61
candidate_elem_00 ... candidate_elem_61
candidate_drift_00 ... candidate_drift_61
true_px
true_py
true_pz
true_p
true_charge
qmetric_score_v0
qmetric_score_v2
```

The `candidate_elem_*` and `candidate_drift_*` columns are needed to reconstruct the `(62, 2)` momentum-model input for each candidate track.

### Residual Summary

```text
scripts/summarize_quality_patterns.py
```

Purpose:

```text
candidate pattern table -> residual-quality summaries
```

Main outputs:

```text
threshold_sweep.csv
threshold_sweep_by_charge.csv
quantile_summary.csv
feature_correlations.csv
summary.txt
plots/*.png
```

### Momentum Evaluation

```text
scripts/evaluate_qmetric_momentum.py
```

Purpose:

```text
candidate pattern table with arrays
-> rebuild candidate_track = (62, 2)
-> run mom_mup.h5 / mom_mum.h5
-> compare predicted momentum to true momentum
-> threshold-level momentum quality summary
```

Main outputs:

```text
momentum_summary.txt
momentum_threshold_sweep.csv
momentum_threshold_sweep_by_charge.csv
optional candidate-level predictions CSV
```

### Component Comparison

```text
scripts/compare_qmetric_components.py
scripts/compare_qmetric_components_v2.py
```

Purpose:

```text
compare qmetric_score_v0, qmetric_score_v2, and diagnostic score variants
at the same kept-track fractions
```

This was necessary because different score definitions have different score distributions. Fixed threshold comparison alone is not fair. The more meaningful comparison is at the same approximate kept-track fraction, such as 50%, 60%, or 70% of candidate tracks.

---

## 3. QMetric Score Definitions

### Common Helper Definitions

Let:

```text
clip01(x) = min(max(x, 0), 1)
```

Feature names are the exact column names used in the pattern table.

```text
m = missing_fraction_active
s = station_coverage_fraction
g = min(max_missing_gap_active / 20.0, 1.0)
o = min(event_mean_layer_occupancy / 20.0, 1.0)
ld = min(local_density_mean / 5.0, 1.0)
c = softmax_conf_mean
r = softmax_margin_mean
h = softmax_entropy_mean
```

Interpretation:

```text
m  : fraction of active detector slots missing from the candidate track
s  : fraction of stations with at least one active hit
g  : normalized maximum missing-hit gap, capped at 1
o  : normalized event-level occupancy, capped at 1
ld : normalized local candidate density, capped at 1
c  : mean selected-element softmax confidence
r  : mean top1-top2 softmax margin
h  : mean normalized softmax entropy
```

### QMetric v0 Equation

`qmetric_score_v0` is the first interpretable baseline score.

```text
qmetric_score_v0 = clip01(
    1.0
    - 0.55 * missing_fraction_active
    - 0.20 * (1.0 - station_coverage_fraction)
    - 0.10 * min(max_missing_gap_active / 20.0, 1.0)
    - 0.10 * min(event_mean_layer_occupancy / 20.0, 1.0)
    + 0.15 * softmax_conf_mean
    + 0.15 * softmax_margin_mean
    - 0.15 * softmax_entropy_mean
)
```

Equivalently:

```text
qmetric_score_v0 = clip01(
    1.0
    - 0.55*m
    - 0.20*(1.0 - s)
    - 0.10*g
    - 0.10*o
    + 0.15*c
    + 0.15*r
    - 0.15*h
)
```

Main design idea:

```text
v0 penalizes missing hits, poor station coverage, long missing-hit gaps,
and high global event occupancy, while rewarding sharp TrackFinder softmax output.
```

### QMetric v2 Equation

`qmetric_score_v2` is the current preferred candidate. It keeps the main v0 structure, but replaces the global event-level occupancy penalty with a local-density penalty around the candidate track.

```text
qmetric_score_v2 = clip01(
    1.0
    - 0.55 * missing_fraction_active
    - 0.20 * (1.0 - station_coverage_fraction)
    - 0.10 * min(max_missing_gap_active / 20.0, 1.0)
    - 0.06 * min(local_density_mean / 5.0, 1.0)
    + 0.15 * softmax_conf_mean
    + 0.15 * softmax_margin_mean
    - 0.15 * softmax_entropy_mean
)
```

Equivalently:

```text
qmetric_score_v2 = clip01(
    1.0
    - 0.55*m
    - 0.20*(1.0 - s)
    - 0.10*g
    - 0.06*ld
    + 0.15*c
    + 0.15*r
    - 0.15*h
)
```

Main design idea:

```text
v2 should not strongly punish a candidate just because the whole event is busy.
Instead, it penalizes the candidate if its own local neighborhood is confused.
This is more consistent with the fact that QMetric selects tracks, not events.
```

---

## 4. v0 Residual Validation Results

The initial feature-only single-track validation result had:

```text
rows: 17142
events: 8571
charges: {'mup': 8571, 'mum': 8571}
```

This is correct for single-track mode:

```text
8571 events * 2 candidate tracks/event = 17142 candidate tracks
```

Overall residual quality for this validation table:

```text
mean_abs_residual_on_truth_hits: mean ~= 0.903
exact_fraction_on_truth_hits: mean ~= 0.819
residual_leq_2_fraction_on_truth_hits: mean ~= 0.904
qmetric_score_v0: mean ~= 0.918, median ~= 0.942
```

v0 threshold sweep:

```text
threshold >= 0.90
    kept_track_fraction ~= 66.5%
    mean_abs_residual ~= 0.188
    exact_fraction ~= 0.917
    residual <= 2 fraction ~= 0.974

threshold >= 0.94
    kept_track_fraction ~= 51.1%
    mean_abs_residual ~= 0.117
    exact_fraction ~= 0.938
    residual <= 2 fraction ~= 0.983

threshold >= 0.98
    kept_track_fraction ~= 32.7%
    mean_abs_residual ~= 0.070
    exact_fraction ~= 0.958
    residual <= 2 fraction ~= 0.990
```

v0 score quantile separation:

```text
lowest score quartile:
    mean_abs_residual ~= 2.86
    exact_fraction ~= 0.571
    residual <= 2 fraction ~= 0.715

highest score quartile:
    mean_abs_residual ~= 0.054
    exact_fraction ~= 0.966
    residual <= 2 fraction ~= 0.993
```

Conclusion from residual validation:

```text
qmetric_score_v0 clearly separates low-quality candidate tracks from high-quality candidate tracks.
```

---

## 5. Momentum Validation Results

Residual validation alone is not enough, because the actual goal is not only hit-level accuracy. The real question is whether selected candidate tracks are worth sending to momentum reconstruction.

Momentum validation was performed by:

```text
1. Reading the candidate pattern table with candidate_elem_* and candidate_drift_* columns.
2. Reconstructing candidate_track with shape (62, 2).
3. Running mom_mup.h5 or mom_mum.h5 depending on candidate charge.
4. Comparing predicted momentum to true momentum.
5. Sweeping QMetric thresholds.
```

Overall conclusion:

```text
QMetric thresholds reduce downstream momentum error and reduce the bad momentum-error tail.
```

For v0, representative full-validation momentum results:

```text
v0 threshold >= 0.90
    kept_track_fraction ~= 66.5%
    momentum_relative_l2_error_mean ~= 0.0590
    bad_rel_l2_gt_0p10_fraction ~= 14.8%

v0 threshold >= 0.94
    kept_track_fraction ~= 51.2%
    momentum_relative_l2_error_mean ~= 0.0553
    bad_rel_l2_gt_0p10_fraction ~= 13.3%

v0 threshold >= 0.98
    kept_track_fraction ~= 32.6%
    momentum_relative_l2_error_mean ~= 0.0527
    bad_rel_l2_gt_0p10_fraction ~= 11.9%
```

Important interpretation:

```text
Increasing the threshold removes more candidate tracks.
This improves momentum quality, but after about 50% kept tracks the improvement becomes smaller.
Therefore, threshold selection must balance reconstruction cleanliness against statistics.
```

---

## 6. Component Comparison Results

Component comparison was used to answer:

```text
Which feature groups actually help select momentum-reconstruction-worthy tracks?
```

The following diagnostic score variants were compared:

```text
qmetric_score_v0
score_v0_no_occupancy
score_softmax_only
score_shape_only
score_occupancy_only
score_smoothness_only
score_v0_no_softmax
score_v0_no_shape
score_v1_balanced
score_v2_low_occupancy
score_v2_local_density
score_v2_low_occ_local
score_v2_softmax_min_guard
score_v2_softmax_shape_local
```

Important result at approximately 50% kept tracks:

```text
score_v2_local_density:
    kept_fraction = 50.0%
    mean_abs_residual ~= 0.1127
    momentum_relative_l2_error_mean ~= 0.0535
    bad_rel_l2_gt_0p10_fraction ~= 12.39%

qmetric_score_v0:
    kept_fraction = 50.0%
    mean_abs_residual ~= 0.1092
    momentum_relative_l2_error_mean ~= 0.0551
    bad_rel_l2_gt_0p10_fraction ~= 13.23%

score_softmax_only:
    kept_fraction = 50.0%
    mean_abs_residual ~= 0.0786
    momentum_relative_l2_error_mean ~= 0.0687
    bad_rel_l2_gt_0p10_fraction ~= 14.69%
```

Interpretation:

```text
softmax-only gives very good hit residuals, but worse momentum performance.
Therefore, TrackFinder confidence alone is not enough to define reconstruction-worthiness.
```

Rejected directions:

```text
score_v1_balanced:
    rejected; worse momentum behavior than v0/v2

score_softmax_only:
    rejected as final metric; useful diagnostic but not enough for momentum quality

score_shape_only:
    rejected as standalone score

score_occupancy_only:
    rejected as standalone score

score_smoothness_only:
    rejected as standalone score
```

Important insight:

```text
Global event occupancy is useful information, but penalizing it too strongly can reject tracks that are still locally clean and momentum-reconstructable.
Local candidate density is a better replacement for the current single-track validation setting.
```

---

## 7. v2 Definition and Validation Results

`qmetric_score_v2` is the current preferred candidate.

Main change from v0:

```text
v0 uses global event_mean_layer_occupancy penalty.
v2 replaces this with a local_density_mean penalty around the candidate track.
```

This is conceptually better because QMetric selects a track candidate, not the whole event.

v2 full validation sanity check:

```text
rows: 400000
events: 200000
charges: {'mup': 200000, 'mum': 200000}
```

This is correct:

```text
200000 events * 2 candidate tracks/event = 400000 candidate tracks
```

v2 residual threshold sweep:

```text
v2 threshold >= 0.94
    kept_track_fraction ~= 71.8%
    mean_abs_residual ~= 0.225
    residual <= 2 fraction ~= 0.972

v2 threshold >= 0.98
    kept_track_fraction ~= 54.6%
    mean_abs_residual ~= 0.129
    residual <= 2 fraction ~= 0.984

v2 threshold >= 0.99
    kept_track_fraction ~= 48.7%
    mean_abs_residual ~= 0.108
    residual <= 2 fraction ~= 0.987

v2 threshold >= 0.995
    kept_track_fraction ~= 45.5%
    mean_abs_residual ~= 0.099
    residual <= 2 fraction ~= 0.988
```

v2 momentum threshold sweep:

```text
v2 threshold >= 0.94
    kept_track_fraction ~= 71.8%
    momentum_relative_l2_error_mean ~= 0.0591
    bad_rel_l2_gt_0p10_fraction ~= 15.1%

v2 threshold >= 0.98
    kept_track_fraction ~= 54.6%
    momentum_relative_l2_error_mean ~= 0.0542
    bad_rel_l2_gt_0p10_fraction ~= 12.8%

v2 threshold >= 0.99
    kept_track_fraction ~= 48.7%
    momentum_relative_l2_error_mean ~= 0.0533
    bad_rel_l2_gt_0p10_fraction ~= 12.3%

v2 threshold >= 0.995
    kept_track_fraction ~= 45.5%
    momentum_relative_l2_error_mean ~= 0.0529
    bad_rel_l2_gt_0p10_fraction ~= 12.1%
```

Fair v0 vs v2 comparison must be done at similar kept-track fraction, not at the same threshold number.

Representative fair comparison:

```text
v0 @ threshold ~= 0.94
    kept_track_fraction ~= 51.2%
    momentum_relative_l2_error_mean ~= 0.0553
    bad_rel_l2_gt_0p10_fraction ~= 13.3%

v2 @ threshold ~= 0.99
    kept_track_fraction ~= 48.7%
    momentum_relative_l2_error_mean ~= 0.0533
    bad_rel_l2_gt_0p10_fraction ~= 12.3%
```

Conclusion:

```text
v2 is slightly better for downstream momentum reconstruction-worthiness while preserving the interpretable structure of v0.
```

---

## 8. Recommended Thresholds

Because v2 scores are more concentrated near the high-score region than v0 scores, v0 and v2 threshold numbers are not directly comparable.

### v0 Threshold Interpretation

```text
Loose:
    qmetric_score_v0 >= 0.90
    keeps about 66% of tracks

Balanced:
    qmetric_score_v0 >= 0.94
    keeps about 51% of tracks

Strict:
    qmetric_score_v0 >= 0.98
    keeps about 33% of tracks
```

### v2 Threshold Interpretation

```text
Loose:
    qmetric_score_v2 >= 0.94
    keeps about 72% of tracks

Balanced / recommended default:
    qmetric_score_v2 >= 0.98
    keeps about 55% of tracks

Strict:
    qmetric_score_v2 >= 0.99 or 0.995
    keeps about 49% or 45% of tracks
```

Current recommended default:

```text
qmetric_score_v2 >= 0.98
```

Reason:

```text
This keeps slightly more than half of candidate tracks, reduces the bad momentum-error tail,
and avoids throwing away as much statistics as the stricter 0.99 or 0.995 cuts.
```

When discussing thresholds, always report:

```text
score version + threshold + kept-track fraction
```

Example:

```text
v0 @ 0.94 keeps about 51% of tracks.
v2 @ 0.98 keeps about 55% of tracks.
```

This avoids misleading comparisons between differently scaled scores.

---

## 9. Current Conclusion

Current status:

```text
single-track validated preferred QMetric candidate
```

Not yet:

```text
final production QMetric
```

Summary judgment:

```text
qmetric_score_v0 is the initial interpretable baseline.
qmetric_score_v2 is the current preferred candidate because it keeps the same interpretable structure but replaces global event occupancy punishment with local candidate-confusion information.
This improves downstream momentum selection at similar kept-track fractions.
```

The main evidence is:

```text
1. QMetric threshold improves TrackFinder-vs-truth residual quality.
2. QMetric threshold improves downstream momentum reconstruction quality.
3. QMetric threshold reduces high-error momentum outliers.
4. Component ablation shows softmax-only is not enough.
5. v2 local-density score gives better momentum behavior than v0 at similar kept fraction.
```

---

## 10. What Not To Do Yet

Do not call v2 the final production QMetric yet.

Reasons:

```text
1. The current validation is single-track based.
2. Multi-track introduces false candidates, duplicate candidates, empty candidates, and candidate overlap.
3. The v2 score scale is concentrated near 1.0, so user-facing calibration may still be needed.
4. The current threshold recommendation is empirical and validation-set based.
5. The current QMetric is still hand-designed, not yet trained or calibrated as a probability.
```

Do not use truth-based columns as runtime features.

Forbidden runtime inputs:

```text
mean_abs_residual_on_truth_hits
max_abs_residual_on_truth_hits
exact_fraction_on_truth_hits
residual_leq_2_fraction_on_truth_hits
true_px, true_py, true_pz
momentum error columns
```

Do not rely on fixed threshold numbers across different score versions.

```text
v0 @ 0.94 and v2 @ 0.94 do not mean the same thing.
Always compare by kept-track fraction or calibrated confidence.
```

Do not over-tune v2 on the current single-track validation set.

```text
The next real test is multi-track candidate selection.
Further single-track weight tuning risks overfitting to this validation setting.
```

Do not integrate directly into QTracker.py yet.

```text
QTracker.py is not currently the active development/integration point for the latest TrackFinder/QMetric workflow.
The current workflow is direct model evaluation and candidate-level table analysis.
```

---

## 11. Immediate Next Steps

### Step 1: Keep this document updated

The result set is now large enough that v0/v2, residual/momentum, and debug/normal results can easily become confusing. This summary should be updated whenever a new score version or multi-track result is added.

### Step 2: Prepare Multi-Track Candidate Adapter

The next major development target should be multi-track support.

Expected MultiTrackFinder output:

```text
seg_output.shape == (N, P, 2, 62, 201)
```

Needed adapter output:

```text
one row per candidate track:
    event_id
    pair_index
    charge
    candidate_elem_00 ... candidate_elem_61
    candidate_drift_00 ... candidate_drift_61
    softmax features
    hit pattern features
    local density features
    qmetric_score_v2
```

Multi-track-specific features likely needed:

```text
- candidate existence score
- empty-pair detection
- duplicate-hit conflict
- overlap with other candidates in the same event
- candidate multiplicity context
- pair-level confidence
```

### Step 3: Validate v2 on Multi-Track Candidates

The same validation structure should be repeated:

```text
multi-track candidate table
-> residual summary
-> momentum validation
-> false candidate / duplicate candidate analysis
-> threshold recommendation
```

Key multi-track metrics:

```text
kept candidate tracks
kept track fraction
truth residual quality
momentum relative error
bad momentum tail fraction
false candidate rejection
duplicate candidate rejection
```

### Step 4: Consider Score Calibration Later

Because v2 scores cluster near 1.0, a future calibrated confidence score may be more user-friendly.

Possible future representation:

```text
qmetric_score_v2_raw = current formula score
qmetric_confidence = calibrated empirical probability or monotonic mapping
```

This should not be done before multi-track validation.

---

## One-Sentence Handoff Summary

The QMetric work has successfully progressed from candidate-level pattern-table generation to residual validation, momentum validation, component ablation, and selection of `qmetric_score_v2` as the current preferred single-track validated score; the next major step is to apply the same candidate-level QMetric framework to multi-track TrackFinder outputs.
