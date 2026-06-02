#!/usr/bin/env python3
"""
Build a candidate-level Quality Metric pattern table from TrackFinder output.

This file follows the existing QTracker_training evaluation convention:
    - use TensorFlow / Keras through tf.keras
    - load full .keras/.h5 models with custom_objects and compile=False
    - use rebuild + load_weights only when explicitly requested
    - do not modify QTracker.py, refine.py, or old Qmetric_training.py

The output CSV is a pattern-study table, not the final QMetric model.
Truth residual columns are calibration-only and must not be used as runtime input.
"""

import argparse
import csv
import math
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

if os.environ.get("STRIP_USER_SITE", "0") == "1":
    home_local = os.path.abspath(os.path.expanduser("~/.local"))
    sys.path[:] = [
        path for path in sys.path
        if not (path and os.path.abspath(path).startswith(home_local))
    ]

import numpy as np

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201
NO_HIT_VALUE = 0

INACTIVE_SLICES = (
    (7, 12),
    (55, 58),
    (59, 62),
)

STATION_RANGES = (
    (0, 6),
    (12, 18),
    (18, 30),
    (30, 42),
    (42, 55),
    (58, 59),
)

FRONT_RANGE = (0, 18)
MIDDLE_RANGE = (18, 42)
BACK_RANGE = (42, 62)
TAIL_RANGE = (54, 62)


def make_active_mask():
    mask = np.ones(NUM_DETECTORS, dtype=bool)
    for start, stop in INACTIVE_SLICES:
        mask[start:stop] = False
    return mask


ACTIVE_MASK = make_active_mask()


class ModelLoadError(RuntimeError):
    pass


def print_environment():
    print("[env] python:", sys.executable)
    print("[env] project_root:", PROJECT_ROOT)
    print("[env] models_dir:", MODELS_DIR)
    try:
        import tensorflow as tf
        print("[env] tensorflow:", tf.__version__, tf.__file__)
    except Exception as exc:
        print("[env] tensorflow import failed:", exc)
    try:
        import keras
        print("[env] keras:", keras.__version__, keras.__file__)
    except Exception as exc:
        print("[env] keras import failed:", exc)
    try:
        import ROOT
        print("[env] ROOT:", ROOT.gROOT.GetVersion())
    except Exception as exc:
        print("[env] ROOT import failed:", exc)


def build_custom_objects(model_kind):
    custom_objects = {}

    try:
        from models.losses import custom_loss, weighted_bce, multi_track_loss
        custom_objects["custom_loss"] = custom_loss
        custom_objects["weighted_bce"] = weighted_bce
        custom_objects["multi_track_loss"] = multi_track_loss
        custom_objects["loss"] = multi_track_loss()
    except Exception as exc:
        print("[warn] Could not import models.losses custom objects:", exc)

    try:
        from models.layers import AxialAttention
        custom_objects["AxialAttention"] = AxialAttention
        custom_objects["Custom>AxialAttention"] = AxialAttention
    except Exception as exc:
        print("[warn] Could not import AxialAttention:", exc)

    if model_kind in ("heavy", "auto"):
        heavy_modules = ("models.TrackFinder_UGNN_heavy_v2", "TrackFinder_UGNN_heavy_v2")
        for module_name in heavy_modules:
            try:
                module = __import__(module_name, fromlist=["dummy"])
                for name in (
                    "PositionalChannels",
                    "WeightedLogitFusion",
                    "LightweightGNN",
                    "StableDetectorGraphConv",
                    "LayerScale",
                ):
                    if hasattr(module, name):
                        obj = getattr(module, name)
                        custom_objects[name] = obj
                        custom_objects["Custom>" + name] = obj
                if hasattr(module, "custom_loss"):
                    custom_objects["custom_loss"] = getattr(module, "custom_loss")
                break
            except Exception:
                pass

    return custom_objects


def load_full_model(model_path, model_kind):
    import tensorflow as tf

    custom_objects = build_custom_objects(model_kind)
    print("[load] Full model:", model_path)
    print("[load] model_kind:", model_kind)
    print("[load] custom_objects:", sorted(custom_objects.keys()))
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
    except Exception as exc:
        raise ModelLoadError(
            "Failed to load full model using the same convention as existing "
            "evaluation scripts: tf.keras.models.load_model(..., "
            "custom_objects=..., compile=False).\n"
            "This usually means the checkpoint was saved with a different "
            "Keras/TensorFlow serialization format than the current runtime.\n"
            "Original error:\n" + str(exc)
        )


def rebuild_single_model(args):
    from models import TrackFinder

    print("[load] Rebuilding models.TrackFinder.build_model and loading weights from:", args.model_path)
    model = TrackFinder.build_model(
        use_bn=bool(args.batch_norm),
        dropout_bn=args.dropout_bn,
        dropout_enc=args.dropout_enc,
        denoise_base=args.denoise_base,
        base=args.base,
        use_attn=bool(args.use_attn),
        use_attn_ffn=bool(args.use_attn_ffn),
        dropout_attn=args.dropout_attn,
    )
    model.load_weights(args.model_path)
    return model


def rebuild_multi_model(args):
    from models import MultiTrackFinder

    print("[load] Rebuilding models.MultiTrackFinder.build_model and loading weights from:", args.model_path)
    model = MultiTrackFinder.build_model(
        use_bn=bool(args.batch_norm),
        dropout_bn=args.dropout_bn,
        dropout_enc=args.dropout_enc,
        denoise_base=args.denoise_base,
        base=args.base,
        use_attn=bool(args.use_attn),
        use_attn_ffn=bool(args.use_attn_ffn),
        dropout_attn=args.dropout_attn,
        max_pairs=args.max_pairs,
    )
    model.load_weights(args.model_path)
    return model


def rebuild_unetpp_model(args):
    import TrackFinder_unetpp

    print("[load] Rebuilding TrackFinder_unetpp.build_model and loading weights from:", args.model_path)
    model = TrackFinder_unetpp.build_model(
        use_bn=bool(args.batch_norm),
        deep_supervision=bool(args.deep_supervision),
    )
    model.load_weights(args.model_path)
    return model


def load_trained_model(args):
    if args.load_mode == "full":
        return load_full_model(args.model_path, args.model_kind)
    if args.load_mode == "rebuild_single":
        return rebuild_single_model(args)
    if args.load_mode == "rebuild_multi":
        return rebuild_multi_model(args)
    if args.load_mode == "rebuild_unetpp":
        return rebuild_unetpp_model(args)
    if args.load_mode == "auto":
        try:
            return load_full_model(args.model_path, args.model_kind)
        except ModelLoadError as exc:
            print("[warn] Full model load failed in auto mode.")
            print(str(exc))
            if args.multi_track:
                return rebuild_multi_model(args)
            return rebuild_single_model(args)
    raise ValueError("Unsupported load_mode: " + str(args.load_mode))


def load_root_arrays(root_file, multi_track, max_pairs, use_denoise_loader, max_events):
    from models import data_loader

    print("[data] Loading ROOT file:", root_file)
    if use_denoise_loader:
        X, X_clean, y_mup, y_mum = data_loader.load_data_denoise(
            root_file,
            multi_track=multi_track,
            max_pairs=max_pairs if multi_track else None,
        )
    else:
        X, y_mup, y_mum = data_loader.load_data(
            root_file,
            multi_track=multi_track,
            max_pairs=max_pairs if multi_track else None,
        )
        X_clean = None

    if X is None:
        raise RuntimeError("data_loader returned None. Check ROOT tree/branches.")

    if max_events > 0:
        X = X[:max_events]
        y_mup = y_mup[:max_events]
        y_mum = y_mum[:max_events]
        if X_clean is not None:
            X_clean = X_clean[:max_events]

    print("[data] X shape:", X.shape)
    print("[data] y_mup shape:", y_mup.shape)
    print("[data] y_mum shape:", y_mum.shape)
    if X_clean is not None:
        print("[data] X_clean shape:", X_clean.shape)

    return X, X_clean, y_mup, y_mum


def load_event_metadata(root_file, max_events):
    import ROOT

    print("[data] Loading event metadata for candidate arrays/momentum truth:", root_file)
    fin = ROOT.TFile.Open(root_file, "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError("Could not open ROOT file for metadata: " + str(root_file))
    tree = fin.Get("tree")
    if not tree:
        fin.Close()
        raise RuntimeError("Could not find tree named 'tree' in ROOT file: " + str(root_file))

    detector_ids = []
    element_ids = []
    drift_distances = []
    true_momenta = []

    n_entries = tree.GetEntries()
    if max_events > 0:
        n_entries = min(n_entries, max_events)

    for index in range(n_entries):
        tree.GetEntry(index)
        detector_ids.append(np.array(tree.detectorID, dtype=np.int32))
        element_ids.append(np.array(tree.elementID, dtype=np.int32))
        drift_distances.append(np.array(tree.driftDistance, dtype=np.float32))

        entry = {
            "gpx": np.array(tree.gpx, dtype=np.float32) if hasattr(tree, "gpx") else np.array([], dtype=np.float32),
            "gpy": np.array(tree.gpy, dtype=np.float32) if hasattr(tree, "gpy") else np.array([], dtype=np.float32),
            "gpz": np.array(tree.gpz, dtype=np.float32) if hasattr(tree, "gpz") else np.array([], dtype=np.float32),
            "gCharge": np.array(tree.gCharge, dtype=np.int32) if hasattr(tree, "gCharge") else np.array([], dtype=np.int32),
        }
        true_momenta.append(entry)

    fin.Close()
    print("[data] metadata events:", len(detector_ids))
    return detector_ids, element_ids, drift_distances, true_momenta


def infer_detector_id_base(detector_ids):
    values = []
    for item in detector_ids[:min(200, len(detector_ids))]:
        if len(item) > 0:
            values.append(np.asarray(item, dtype=np.int32))
    if not values:
        print("[warn] Could not infer detectorID base from empty detectorID arrays. Falling back to 1-based.")
        return 1
    all_values = np.concatenate(values)
    if np.any(all_values == 0):
        print("[data] inferred detectorID base: zero-based ROOT detectorID")
        return 0
    print("[data] inferred detectorID base: one-based ROOT detectorID")
    return 1


def choose_detector_id_base(mode, detector_ids):
    if mode == "zero":
        return 0
    if mode == "one":
        return 1
    if mode == "auto":
        return infer_detector_id_base(detector_ids)
    raise ValueError("Unsupported detector_id_base: " + str(mode))


def snap_candidate_to_real_hits(hit_array, detector_ids_event, element_ids_event, drift_distances_event, detector_id_base):
    hit_array = np.asarray(hit_array, dtype=np.int32)
    detector_ids_event = np.asarray(detector_ids_event, dtype=np.int32)
    element_ids_event = np.asarray(element_ids_event, dtype=np.int32)
    drift_distances_event = np.asarray(drift_distances_event, dtype=np.float32)

    candidate_elem = np.zeros(NUM_DETECTORS, dtype=np.int32)
    candidate_drift = np.zeros(NUM_DETECTORS, dtype=np.float32)

    for detector in range(NUM_DETECTORS):
        pred_elem = int(hit_array[detector])
        if pred_elem == NO_HIT_VALUE:
            continue

        root_detector_id = detector if detector_id_base == 0 else detector + 1
        matches = np.where(detector_ids_event == root_detector_id)[0]
        if len(matches) == 0:
            continue

        actual_elems = element_ids_event[matches]
        best_local = int(np.argmin(np.abs(actual_elems - pred_elem)))
        best_index = int(matches[best_local])
        candidate_elem[detector] = int(element_ids_event[best_index])
        candidate_drift[detector] = float(drift_distances_event[best_index])

    return candidate_elem, candidate_drift


def exact_candidate_drift(hit_array, detector_ids_event, element_ids_event, drift_distances_event, detector_id_base):
    hit_array = np.asarray(hit_array, dtype=np.int32)
    detector_ids_event = np.asarray(detector_ids_event, dtype=np.int32)
    element_ids_event = np.asarray(element_ids_event, dtype=np.int32)
    drift_distances_event = np.asarray(drift_distances_event, dtype=np.float32)

    candidate_elem = hit_array.astype(np.int32).copy()
    candidate_drift = np.zeros(NUM_DETECTORS, dtype=np.float32)

    for detector in range(NUM_DETECTORS):
        pred_elem = int(hit_array[detector])
        if pred_elem == NO_HIT_VALUE:
            candidate_elem[detector] = 0
            continue

        root_detector_id = detector if detector_id_base == 0 else detector + 1
        matches = np.where((detector_ids_event == root_detector_id) & (element_ids_event == pred_elem))[0]
        if len(matches) > 0:
            candidate_drift[detector] = float(drift_distances_event[int(matches[0])])

    return candidate_elem, candidate_drift


def add_candidate_array_columns(row, raw_hit_array, candidate_elem, candidate_drift):
    for detector in range(NUM_DETECTORS):
        key = "%02d" % detector
        row["raw_elem_" + key] = int(raw_hit_array[detector])
        row["candidate_elem_" + key] = int(candidate_elem[detector])
        row["candidate_drift_" + key] = float(candidate_drift[detector])


def add_truth_momentum_columns(row, true_momentum_entry, charge_index):
    if true_momentum_entry is None:
        row["true_px"] = float("nan")
        row["true_py"] = float("nan")
        row["true_pz"] = float("nan")
        row["true_p"] = float("nan")
        row["true_charge"] = 0
        return

    gpx = true_momentum_entry.get("gpx", np.array([], dtype=np.float32))
    gpy = true_momentum_entry.get("gpy", np.array([], dtype=np.float32))
    gpz = true_momentum_entry.get("gpz", np.array([], dtype=np.float32))
    gcharge = true_momentum_entry.get("gCharge", np.array([], dtype=np.int32))

    if len(gpx) <= charge_index or len(gpy) <= charge_index or len(gpz) <= charge_index:
        row["true_px"] = float("nan")
        row["true_py"] = float("nan")
        row["true_pz"] = float("nan")
        row["true_p"] = float("nan")
        row["true_charge"] = 0
        return

    px = float(gpx[charge_index])
    py = float(gpy[charge_index])
    pz = float(gpz[charge_index])
    row["true_px"] = px
    row["true_py"] = py
    row["true_pz"] = pz
    row["true_p"] = float(math.sqrt(px * px + py * py + pz * pz))
    if len(gcharge) > charge_index:
        row["true_charge"] = int(gcharge[charge_index])
    else:
        row["true_charge"] = 0


def get_segmentation_output(prediction):
    if isinstance(prediction, (list, tuple)):
        if len(prediction) < 2:
            raise ValueError("Model returned a list/tuple but no segmentation output was found.")
        return prediction[1]
    return prediction


def validate_prediction_shape(pred, multi_track, max_pairs):
    if multi_track:
        if pred.ndim != 5:
            raise ValueError("Expected multi-track prediction shape (N,P,2,62,201), got " + str(pred.shape))
        if pred.shape[1] != max_pairs or pred.shape[2] != 2:
            raise ValueError("Bad multi-track pair/charge axes: " + str(pred.shape))
        if pred.shape[3] != NUM_DETECTORS or pred.shape[4] != NUM_ELEMENT_IDS:
            raise ValueError("Bad multi-track detector/element axes: " + str(pred.shape))
    else:
        if pred.ndim != 4:
            raise ValueError("Expected single-track prediction shape (N,2,62,201), got " + str(pred.shape))
        if pred.shape[1] != 2:
            raise ValueError("Bad single-track charge axis: " + str(pred.shape))
        if pred.shape[2] != NUM_DETECTORS or pred.shape[3] != NUM_ELEMENT_IDS:
            raise ValueError("Bad single-track detector/element axes: " + str(pred.shape))


def safe_fraction(num, den):
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def longest_false_run(mask):
    best = 0
    current = 0
    for value in mask:
        if not value:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return int(best)


def compute_hit_pattern_features(hit_array):
    hit_array = np.asarray(hit_array, dtype=np.int32)
    active_hits = (hit_array != NO_HIT_VALUE) & ACTIVE_MASK
    active_total = int(np.sum(ACTIVE_MASK))
    hit_count = int(np.sum(active_hits))
    missing_count = active_total - hit_count

    features = {
        "hit_count_active": hit_count,
        "missing_count_active": missing_count,
        "missing_fraction_active": safe_fraction(missing_count, active_total),
        "max_missing_gap_active": longest_false_run(active_hits[ACTIVE_MASK]),
    }

    covered = 0
    station_fractions = []
    for index, (start, stop) in enumerate(STATION_RANGES):
        station_mask = ACTIVE_MASK[start:stop]
        denom = int(np.sum(station_mask))
        count = int(np.sum(active_hits[start:stop]))
        fraction = safe_fraction(count, denom)
        if count > 0:
            covered += 1
        features["station_" + str(index) + "_hit_count"] = count
        features["station_" + str(index) + "_hit_fraction"] = fraction
        station_fractions.append(fraction)

    features["covered_station_count"] = covered
    features["station_coverage_fraction"] = safe_fraction(covered, len(STATION_RANGES))
    features["min_station_hit_fraction"] = float(min(station_fractions)) if station_fractions else 0.0

    for name, span in (("front", FRONT_RANGE), ("middle", MIDDLE_RANGE), ("back", BACK_RANGE), ("tail", TAIL_RANGE)):
        start, stop = span
        region_mask = ACTIVE_MASK[start:stop]
        denom = int(np.sum(region_mask))
        count = int(np.sum(active_hits[start:stop]))
        features[name + "_hit_count"] = count
        features[name + "_hit_fraction"] = safe_fraction(count, denom)

    nonzero_detectors = np.where(active_hits)[0]
    if len(nonzero_detectors) >= 2:
        elems = hit_array[nonzero_detectors].astype(np.float32)
        diffs = np.diff(elems)
        features["mean_abs_element_step"] = float(np.mean(np.abs(diffs)))
        features["max_abs_element_step"] = float(np.max(np.abs(diffs)))
    else:
        features["mean_abs_element_step"] = 0.0
        features["max_abs_element_step"] = 0.0

    return features


def compute_event_features(event_matrix):
    mat = np.asarray(event_matrix)
    if mat.ndim == 3:
        mat = mat[:, :, 0]
    hit_counts = np.sum(mat > 0, axis=1)
    active_counts = hit_counts[ACTIVE_MASK]
    return {
        "event_total_occupancy": int(np.sum(hit_counts)),
        "event_active_occupancy": int(np.sum(active_counts)),
        "event_mean_layer_occupancy": float(np.mean(active_counts)) if len(active_counts) else 0.0,
        "event_max_layer_occupancy": int(np.max(active_counts)) if len(active_counts) else 0,
        "event_nonempty_active_layers": int(np.sum(active_counts > 0)),
    }


def compute_local_density(event_matrix, hit_array, radius):
    mat = np.asarray(event_matrix)
    if mat.ndim == 3:
        mat = mat[:, :, 0]
    densities = []
    for det in range(NUM_DETECTORS):
        elem = int(hit_array[det])
        if not ACTIVE_MASK[det] or elem == NO_HIT_VALUE:
            continue
        lo = max(0, elem - radius)
        hi = min(NUM_ELEMENT_IDS, elem + radius + 1)
        densities.append(int(np.sum(mat[det, lo:hi] > 0)))
    if not densities:
        return {"local_density_mean": 0.0, "local_density_max": 0}
    return {
        "local_density_mean": float(np.mean(densities)),
        "local_density_max": int(np.max(densities)),
    }


def normalized_entropy(prob):
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / math.log(len(p)))


def compute_softmax_features(softmax, hit_array):
    sm = np.asarray(softmax, dtype=np.float32)
    hit_array = np.asarray(hit_array, dtype=np.int32)
    confs = []
    margins = []
    entropies = []
    presence_probs = []

    for det in range(NUM_DETECTORS):
        if not ACTIVE_MASK[det]:
            continue
        probs = sm[det]
        pred_elem = int(hit_array[det])
        pred_elem = max(0, min(NUM_ELEMENT_IDS - 1, pred_elem))
        confs.append(float(probs[pred_elem]))
        top2 = np.partition(probs, -2)[-2:]
        top2.sort()
        margins.append(float(top2[-1] - top2[-2]))
        entropies.append(normalized_entropy(probs))
        presence_probs.append(float(1.0 - probs[NO_HIT_VALUE]))

    if not confs:
        return {
            "softmax_conf_mean": 0.0,
            "softmax_conf_min": 0.0,
            "softmax_margin_mean": 0.0,
            "softmax_margin_min": 0.0,
            "softmax_entropy_mean": 0.0,
            "presence_prob_mean": 0.0,
            "presence_prob_min": 0.0,
        }

    return {
        "softmax_conf_mean": float(np.mean(confs)),
        "softmax_conf_min": float(np.min(confs)),
        "softmax_margin_mean": float(np.mean(margins)),
        "softmax_margin_min": float(np.min(margins)),
        "softmax_entropy_mean": float(np.mean(entropies)),
        "presence_prob_mean": float(np.mean(presence_probs)),
        "presence_prob_min": float(np.min(presence_probs)),
    }


def compare_to_truth(pred_hit_array, truth_hit_array):
    pred = np.asarray(pred_hit_array, dtype=np.int32)
    truth = np.asarray(truth_hit_array, dtype=np.int32)
    truth_hit = (truth != NO_HIT_VALUE) & ACTIVE_MASK
    pred_hit = (pred != NO_HIT_VALUE) & ACTIVE_MASK
    both = truth_hit & pred_hit
    abs_residuals = np.abs(pred[both] - truth[both])

    if len(abs_residuals) > 0:
        mean_abs = float(np.mean(abs_residuals))
        max_abs = float(np.max(abs_residuals))
        exact = int(np.sum(abs_residuals == 0))
        leq1 = int(np.sum(abs_residuals <= 1))
        leq2 = int(np.sum(abs_residuals <= 2))
    else:
        mean_abs = 0.0
        max_abs = 0.0
        exact = 0
        leq1 = 0
        leq2 = 0

    truth_count = int(np.sum(truth_hit))
    pred_count = int(np.sum(pred_hit))
    return {
        "truth_hit_count_active": truth_count,
        "pred_hit_count_active": pred_count,
        "matched_truth_pred_count": int(np.sum(both)),
        "exact_match_count": exact,
        "residual_leq_1_count": leq1,
        "residual_leq_2_count": leq2,
        "mean_abs_residual_on_truth_hits": mean_abs,
        "max_abs_residual_on_truth_hits": max_abs,
        "missing_truth_hit_count": int(np.sum(truth_hit & ~pred_hit)),
        "extra_pred_hit_count": int(np.sum(pred_hit & ~truth_hit)),
        "exact_fraction_on_truth_hits": safe_fraction(exact, truth_count),
        "residual_leq_2_fraction_on_truth_hits": safe_fraction(leq2, truth_count),
    }


def score_v0(row):
    score = 1.0
    score -= 0.55 * float(row.get("missing_fraction_active", 0.0))
    score -= 0.20 * (1.0 - float(row.get("station_coverage_fraction", 0.0)))
    score -= 0.10 * min(float(row.get("max_missing_gap_active", 0.0)) / 20.0, 1.0)
    score -= 0.10 * min(float(row.get("event_mean_layer_occupancy", 0.0)) / 20.0, 1.0)
    score += 0.15 * float(row.get("softmax_conf_mean", 0.0))
    score += 0.15 * float(row.get("softmax_margin_mean", 0.0))
    score -= 0.15 * float(row.get("softmax_entropy_mean", 0.0))
    return max(0.0, min(1.0, score))


def make_candidate_row(event_id, pair_index, charge_name, hit_array, softmax, event_matrix, truth_array, local_radius):
    row = {
        "event_id": int(event_id),
        "pair_index": int(pair_index),
        "charge": charge_name,
    }
    row.update(compute_hit_pattern_features(hit_array))
    row.update(compute_event_features(event_matrix))
    row.update(compute_local_density(event_matrix, hit_array, local_radius))
    row.update(compute_softmax_features(softmax, hit_array))
    row.update(compare_to_truth(hit_array, truth_array))
    row["qmetric_score_v0"] = score_v0(row)
    return row


def build_rows_for_chunk(
    pred,
    X_chunk,
    y_mup_chunk,
    y_mum_chunk,
    multi_track,
    max_pairs,
    local_radius,
    event_offset,
    detector_ids,
    element_ids,
    drift_distances,
    true_momenta,
    detector_id_base,
    args,
):
    rows = []
    n_events = pred.shape[0]
    if multi_track:
        for local_event in range(n_events):
            event_matrix = X_chunk[local_event]
            event_id = event_offset + local_event
            det_evt = detector_ids[event_id] if detector_ids is not None else None
            elem_evt = element_ids[event_id] if element_ids is not None else None
            drift_evt = drift_distances[event_id] if drift_distances is not None else None
            true_evt = true_momenta[event_id] if true_momenta is not None else None
            for pair_index in range(max_pairs):
                for charge_index, charge_name in ((0, "mup"), (1, "mum")):
                    softmax = pred[local_event, pair_index, charge_index]
                    hit_array = np.argmax(softmax, axis=-1).astype(np.int32)
                    if charge_index == 0:
                        truth = y_mup_chunk[local_event, pair_index]
                    else:
                        truth = y_mum_chunk[local_event, pair_index]
                    row = make_candidate_row(event_id, pair_index, charge_name, hit_array, softmax, event_matrix, truth, local_radius)
                    if args.save_candidate_arrays:
                        if args.snap_to_real_hits:
                            candidate_elem, candidate_drift = snap_candidate_to_real_hits(
                                hit_array, det_evt, elem_evt, drift_evt, detector_id_base
                            )
                        else:
                            candidate_elem, candidate_drift = exact_candidate_drift(
                                hit_array, det_evt, elem_evt, drift_evt, detector_id_base
                            )
                        row["detector_id_base_for_drift"] = int(detector_id_base)
                        row["candidate_snap_to_real_hits"] = int(args.snap_to_real_hits)
                        add_candidate_array_columns(row, hit_array, candidate_elem, candidate_drift)
                    if args.save_truth_momentum:
                        add_truth_momentum_columns(row, true_evt, charge_index)
                    rows.append(row)
    else:
        for local_event in range(n_events):
            event_matrix = X_chunk[local_event]
            event_id = event_offset + local_event
            det_evt = detector_ids[event_id] if detector_ids is not None else None
            elem_evt = element_ids[event_id] if element_ids is not None else None
            drift_evt = drift_distances[event_id] if drift_distances is not None else None
            true_evt = true_momenta[event_id] if true_momenta is not None else None
            for charge_index, charge_name in ((0, "mup"), (1, "mum")):
                softmax = pred[local_event, charge_index]
                hit_array = np.argmax(softmax, axis=-1).astype(np.int32)
                truth = y_mup_chunk[local_event] if charge_index == 0 else y_mum_chunk[local_event]
                row = make_candidate_row(event_id, 0, charge_name, hit_array, softmax, event_matrix, truth, local_radius)
                if args.save_candidate_arrays:
                    if args.snap_to_real_hits:
                        candidate_elem, candidate_drift = snap_candidate_to_real_hits(
                            hit_array, det_evt, elem_evt, drift_evt, detector_id_base
                        )
                    else:
                        candidate_elem, candidate_drift = exact_candidate_drift(
                            hit_array, det_evt, elem_evt, drift_evt, detector_id_base
                        )
                    row["detector_id_base_for_drift"] = int(detector_id_base)
                    row["candidate_snap_to_real_hits"] = int(args.snap_to_real_hits)
                    add_candidate_array_columns(row, hit_array, candidate_elem, candidate_drift)
                if args.save_truth_momentum:
                    add_truth_momentum_columns(row, true_evt, charge_index)
                rows.append(row)
    return rows


def open_csv_writer(output_file, first_rows):
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fout = open(output_file, "w", newline="")
    fieldnames = []
    seen = set()
    for row in first_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    return fout, writer


def write_pattern_table(model, X, y_mup, y_mum, args):
    import tensorflow as tf

    detector_ids = None
    element_ids = None
    drift_distances = None
    true_momenta = None
    detector_id_base = 1
    if args.save_candidate_arrays or args.save_truth_momentum:
        detector_ids, element_ids, drift_distances, true_momenta = load_event_metadata(args.root_file, args.max_events)
        detector_id_base = choose_detector_id_base(args.detector_id_base, detector_ids)

    fout = None
    writer = None
    total_rows = 0
    n_events = len(X)

    for start in range(0, n_events, args.chunk_size):
        stop = min(start + args.chunk_size, n_events)
        X_chunk = X[start:stop]
        y_mup_chunk = y_mup[start:stop]
        y_mum_chunk = y_mum[start:stop]
        prediction = model.predict(tf.cast(X_chunk, tf.float32), verbose=0)
        pred = np.asarray(get_segmentation_output(prediction), dtype=np.float32)
        validate_prediction_shape(pred, args.multi_track, args.max_pairs)
        rows = build_rows_for_chunk(
            pred,
            X_chunk,
            y_mup_chunk,
            y_mum_chunk,
            args.multi_track,
            args.max_pairs,
            args.local_radius,
            start,
            detector_ids,
            element_ids,
            drift_distances,
            true_momenta,
            detector_id_base,
            args,
        )
        if writer is None:
            if not rows:
                raise RuntimeError("No candidate rows were produced in the first chunk.")
            fout, writer = open_csv_writer(args.output, rows)
        writer.writerows(rows)
        total_rows += len(rows)
        print("[predict] processed", str(stop) + "/" + str(n_events), "events; rows so far:", total_rows)

    if fout is not None:
        fout.close()
    if total_rows == 0:
        raise RuntimeError("No candidate rows were produced.")
    print("[done] Wrote", total_rows, "candidate rows to", args.output)


def parse_args():
    parser = argparse.ArgumentParser(description="Build candidate-level QMetric pattern table from TrackFinder output.")
    parser.add_argument("root_file", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--output", type=str, default="outputs/qmetric_pattern_table.csv")
    parser.add_argument("--multi_track", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=5)
    parser.add_argument("--use_denoise_loader", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--local_radius", type=int, default=2)
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--print_env", type=int, default=1)
    parser.add_argument("--model_kind", type=str, default="auto", choices=["auto", "single", "multi", "heavy", "unetpp"])
    parser.add_argument("--load_mode", type=str, default="full", choices=["full", "auto", "rebuild_single", "rebuild_multi", "rebuild_unetpp"])
    parser.add_argument("--save_candidate_arrays", type=int, default=0)
    parser.add_argument("--save_truth_momentum", type=int, default=0)
    parser.add_argument("--snap_to_real_hits", type=int, default=1)
    parser.add_argument("--detector_id_base", type=str, default="auto", choices=["auto", "zero", "one"])

    # Rebuild knobs. Defaults match current train.slurm / train_multi.slurm conventions.
    parser.add_argument("--batch_norm", type=int, default=1)
    parser.add_argument("--use_attn", type=int, default=1)
    parser.add_argument("--use_attn_ffn", type=int, default=0)
    parser.add_argument("--denoise_base", type=int, default=32)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--dropout_bn", type=float, default=0.5)
    parser.add_argument("--dropout_enc", type=float, default=0.4)
    parser.add_argument("--dropout_attn", type=float, default=0.1)
    parser.add_argument("--deep_supervision", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.multi_track = bool(args.multi_track)
    args.use_denoise_loader = bool(args.use_denoise_loader)
    args.save_candidate_arrays = bool(args.save_candidate_arrays)
    args.save_truth_momentum = bool(args.save_truth_momentum)
    args.snap_to_real_hits = bool(args.snap_to_real_hits)

    if args.print_env:
        print_environment()

    if args.multi_track and args.max_pairs <= 0:
        raise ValueError("--max_pairs must be positive for multi-track mode.")

    X, X_clean, y_mup, y_mum = load_root_arrays(
        args.root_file,
        args.multi_track,
        args.max_pairs,
        args.use_denoise_loader,
        args.max_events,
    )

    model = load_trained_model(args)
    write_pattern_table(model, X, y_mup, y_mum, args)


if __name__ == "__main__":
    main()
