# QMetric v0 development starter

This patch starts the Quality Metric development outside `QTracker.py`.
It builds a candidate-level pattern table directly from `.keras` / `.h5`
TrackFinder or MultiTrackFinder outputs.

## Add these folders to `QTracker_training/`

```text
quality_metric/
scripts/build_quality_pattern_table.py
```

## Single-track example

```bash
python3 scripts/build_quality_pattern_table.py \
    data/processed_files/mc_events_val.root \
    checkpoints/single_track_finder.keras \
    --output outputs/qmetric_single_val.csv \
    --multi_track 0 \
    --use_denoise_loader 1 \
    --chunk_size 128
```

## Multi-track example

```bash
python3 scripts/build_quality_pattern_table.py \
    data/multi_track/processed_files/mc_events_val.root \
    checkpoints/multi_track_finder.keras \
    --output outputs/qmetric_multi_val.csv \
    --multi_track 1 \
    --max_pairs 5 \
    --use_denoise_loader 1 \
    --chunk_size 64
```

## Output meaning

Each row is one candidate track:

```text
event_id, pair_index, charge
runtime-observable features
calibration-only truth residual columns
qmetric_score_v0
```

`qmetric_score_v0` is intentionally a simple first-pass score. It is not the
final physics metric. The important first product is the pattern table, which
lets us see which candidate observables correlate with reconstruction quality.

## Important constraints

- This does not modify `QTracker.py`.
- This does not use the old `Qmetric_training.py` chi2 model.
- Runtime features do not use ground truth or reconstructed momentum.
- Truth columns are included only for calibration/pattern-study analysis.
