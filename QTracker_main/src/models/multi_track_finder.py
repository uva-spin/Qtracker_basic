import numpy as np
import tensorflow as tf

from src.config import MAX_STEPS, SINGLE_TRACK_FINDER_PATH
from src.layers.axial_attention import AxialAttention
from src.models.refiner import Refiner
from src.preprocessing.data_loader import (
    load_data_with_labels,
    load_data_without_labels,
    load_detector_element_data,
)


class MultiTrackFinder:
    """Auto-regressive multi-track finder.

    Iteratively invokes the single track finder on a (possibly modified)
    event representation.  After each successful extraction the identified
    track's contribution is *softly subtracted* from the hit matrix so that
    subsequent iterations can discover additional tracks.

    The loop terminates when a learned **confidence score** (produced by the
    single track finder's confidence head) drops below a configurable
    threshold, or when ``max_steps`` iterations have been exhausted.

    If the loaded model does **not** contain a confidence head (i.e. it only
    produces two outputs – denoise + segment), the finder falls back to
    running for the full ``max_steps`` iterations.  This ensures backward
    compatibility with legacy single-track checkpoints.
    """

    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        mode: Literal["evaluation", "production"] = "evaluation",
        confidence_threshold: float = 0.5,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the MultiTrackFinder.

        Args:
            max_steps: Maximum number of auto-regressive iterations.  Should
                match the value used during data generation.
            mode: ``"evaluation"`` runs all steps regardless of early exit
                (so that output shapes are uniform for metric computation).
                ``"production"`` allows early termination when no active
                events remain.
            confidence_threshold: Events whose confidence score drops below
                this value are marked inactive (their predicted tracks are
                zeroed out in subsequent iterations).
            model_path: Path to the single track finder checkpoint.  When
                ``None`` the default from ``src.config`` is used.
        """
        self.track_finder = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={"AxialAttention": AxialAttention},
        )
        self.refiner = Refiner()
        self.max_steps = max_steps
        self.mode = mode
        self.confidence_threshold = confidence_threshold

        # Detect whether the model has a confidence head by inspecting the
        # number of outputs.  Legacy models produce 2 (denoise, segment);
        # models with a confidence head produce 3.
        if isinstance(self.track_finder.output, (list, tuple)):
            self._num_outputs = len(self.track_finder.output)
        else:
            self._num_outputs = 1
        self._has_confidence = self._num_outputs >= 3

        if not self._has_confidence:
            print(
                "[MultiTrackFinder] WARNING: loaded model has no confidence "
                "head – falling back to fixed-iteration mode "
                f"(max_steps={self.max_steps})."
            )

    # --- Public interface ------------------------------------------------- #

    def run(
        self, input_root_file: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the multi-track finding algorithm on the input ROOT file.

        Uses an iterative approach to find multiple tracks, continuing until
        the confidence score drops below the threshold or ``max_steps``
        iterations are exhausted.

        Args:
            input_root_file: Path to the input ROOT file.

        Returns:
            A 4-tuple of arrays:
              * ``mu_plus_tracks``   – shape ``(N, T, 62)``
              * ``mu_minus_tracks``  – shape ``(N, T, 62)``
              * ``mu_plus_softmax``  – shape ``(N, T, 62, 201)``
              * ``mu_minus_softmax`` – shape ``(N, T, 62, 201)``
            where *N* is the number of events and *T* = ``max_steps``.
        """
        X = load_data_without_labels(input_root_file)
        if X is None:
            raise ValueError("No data found in the provided ROOT file.")

        n_events = X.shape[0]

        # Pre-load detector-level information for the refiner (done once).
        detector_id, element_id, _, _ = load_detector_element_data(input_root_file)

        mu_plus_softmax: list[np.ndarray] = []
        mu_minus_softmax: list[np.ndarray] = []
        mu_plus_tracks: list[np.ndarray] = []
        mu_minus_tracks: list[np.ndarray] = []

        active_events = np.ones(n_events, dtype=bool)

        for step in range(self.max_steps):
            # Early exit allowed only in production mode.
            if self.mode != "evaluation" and not np.any(active_events):
                # Pad remaining steps with zeros so output shape is uniform.
                n_remaining = self.max_steps - step
                num_det = X.shape[1]
                num_elem = X.shape[2]
                for _ in range(n_remaining):
                    mu_plus_tracks.append(np.zeros((n_events, num_det), dtype=np.int32))
                    mu_minus_tracks.append(
                        np.zeros((n_events, num_det), dtype=np.int32)
                    )
                    mu_plus_softmax.append(
                        np.zeros((n_events, num_det, num_elem), dtype=np.float32)
                    )
                    mu_minus_softmax.append(
                        np.zeros((n_events, num_det, num_elem), dtype=np.float32)
                    )
                break

            # ---- Forward pass ----
            raw_outputs = self.track_finder.predict(tf.cast(X, tf.float32), verbose=0)

            # Unpack outputs – handle both 2-output and 3-output models.
            if self._has_confidence:
                _, y_pred, confidence_score = (
                    raw_outputs[0],
                    raw_outputs[1],
                    raw_outputs[2],
                )
                # ``model.predict`` returns numpy arrays; squeeze the
                # trailing unit dimension so that shape is (N,).
                if confidence_score.ndim > 1:
                    confidence_score = confidence_score.squeeze(axis=-1)

                active_events = active_events & (
                    confidence_score >= self.confidence_threshold
                )
            else:
                _, y_pred = raw_outputs[0], raw_outputs[1]
                # No confidence head – all events remain active until
                # max_steps is reached.

            # Softmax probabilities per muon charge.
            mp_softmax = y_pred[:, 0, :, :]  # (N, 62, 201)
            mm_softmax = y_pred[:, 1, :, :]  # (N, 62, 201)

            # Predicted element-IDs via argmax.
            mu_plus_pred = np.argmax(mp_softmax, axis=-1).astype(np.int32)  # (N, 62)
            mu_minus_pred = np.argmax(mm_softmax, axis=-1).astype(np.int32)  # (N, 62)

            # ---- Refinement ----
            refined_mu_plus_pred, refined_mu_minus_pred = (
                self.refiner.refine_hit_arrays(
                    mu_plus_pred, mu_minus_pred, detector_id, element_id
                )
            )

            # ---- Mask inactive events ----
            refined_mu_plus_pred[~active_events] = 0
            refined_mu_minus_pred[~active_events] = 0
            mp_softmax_copy = mp_softmax.copy()
            mm_softmax_copy = mm_softmax.copy()
            mp_softmax_copy[~active_events] = 0.0
            mm_softmax_copy[~active_events] = 0.0

            mu_plus_softmax.append(mp_softmax_copy)
            mu_minus_softmax.append(mm_softmax_copy)
            mu_plus_tracks.append(refined_mu_plus_pred)
            mu_minus_tracks.append(refined_mu_minus_pred)

            # ---- Soft hit subtraction ----
            X = self._subtract_found_tracks(
                X,
                refined_mu_plus_pred,
                refined_mu_minus_pred,
                mp_softmax,
                mm_softmax,
                active_events=active_events,
            )

        # Stack along the track (step) axis.
        mu_plus_softmax_arr = np.stack(mu_plus_softmax, axis=1)  # (N, T, 62, 201)
        mu_minus_softmax_arr = np.stack(mu_minus_softmax, axis=1)
        mu_plus_tracks_arr = np.stack(mu_plus_tracks, axis=1)  # (N, T, 62)
        mu_minus_tracks_arr = np.stack(mu_minus_tracks, axis=1)

        return (
            mu_plus_tracks_arr,
            mu_minus_tracks_arr,
            mu_plus_softmax_arr,
            mu_minus_softmax_arr,
        )

    def evaluate(self, input_root_file: str) -> dict[str, Union[float, list[float]]]:
        """
        Evaluate the multi-track finder model on the input ROOT file with
        known labels.

        Args:
            input_root_file: Path to the input ROOT file.

        Returns:
            A dictionary containing evaluation metrics.
        """
        X, y_mu_plus, y_mu_minus = load_data_with_labels(input_root_file)
        if any(v is None for v in [X, y_mu_plus, y_mu_minus]):
            raise ValueError("No data found in the provided ROOT file.")

        # Predict tracks using the multi-track finder model
        refined_mu_plus_pred, refined_mu_minus_pred, _, _ = self.run(input_root_file)

        # Prepare true labels – shape (N, 2, 62) or (N, T, 2, 62)
        y_true = np.stack([y_mu_plus, y_mu_minus], axis=1)
        mu_plus_true = y_true[:, 0, :].astype(np.int32)
        mu_minus_true = y_true[:, 1, :].astype(np.int32)

        # Residuals
        mu_plus_residuals = mu_plus_true - refined_mu_plus_pred
        mu_minus_residuals = mu_minus_true - refined_mu_minus_pred

        # Calculate evaluation metrics
        mask = y_mu_plus != 0

        mu_plus_accuracy = np.mean(mu_plus_residuals[mask] == 0)
        mu_minus_accuracy = np.mean(mu_minus_residuals[mask] == 0)

        mu_plus_within_two = np.mean(np.abs(mu_plus_residuals)[mask] <= 2)
        mu_minus_within_two = np.mean(np.abs(mu_minus_residuals)[mask] <= 2)

        mu_plus_mean = np.mean(np.abs(mu_plus_residuals)[mask])
        mu_minus_mean = np.mean(np.abs(mu_minus_residuals)[mask])

        mu_plus_std = np.std(np.abs(mu_plus_residuals)[mask])
        mu_minus_std = np.std(np.abs(mu_minus_residuals)[mask])

        # Per-event metrics
        mu_plus_accuracy_per_track = []
        mu_minus_accuracy_per_track = []

        mu_plus_within_two_per_track = []
        mu_minus_within_two_per_track = []

        mu_plus_mean_per_track = []
        mu_minus_mean_per_track = []

        mu_plus_std_per_track = []
        mu_minus_std_per_track = []

        for evt in range(mu_plus_residuals.shape[0]):
            evt_mask = mask[evt]

            mu_plus_evt = mu_plus_residuals[evt]
            mu_minus_evt = mu_minus_residuals[evt]

            if np.any(evt_mask):
                mu_plus_accuracy_per_track.append(np.mean(mu_plus_evt[evt_mask] == 0))
                mu_minus_accuracy_per_track.append(np.mean(mu_minus_evt[evt_mask] == 0))

                mu_plus_within_two_per_track.append(
                    np.mean(np.abs(mu_plus_evt[evt_mask]) <= 2)
                )
                mu_minus_within_two_per_track.append(
                    np.mean(np.abs(mu_minus_evt[evt_mask]) <= 2)
                )

                mu_plus_mean_per_track.append(np.mean(np.abs(mu_plus_evt[evt_mask])))
                mu_minus_mean_per_track.append(np.mean(np.abs(mu_minus_evt[evt_mask])))

                mu_plus_std_per_track.append(np.std(np.abs(mu_plus_evt[evt_mask])))
                mu_minus_std_per_track.append(np.std(np.abs(mu_minus_evt[evt_mask])))
            else:
                # no real hits in this event
                mu_plus_accuracy_per_track.append(np.nan)
                mu_minus_accuracy_per_track.append(np.nan)
                mu_plus_within_two_per_track.append(np.nan)
                mu_minus_within_two_per_track.append(np.nan)
                mu_plus_mean_per_track.append(np.nan)
                mu_minus_mean_per_track.append(np.nan)
                mu_plus_std_per_track.append(np.nan)
                mu_minus_std_per_track.append(np.nan)

        evaluation_results = {
            "mu_plus_accuracy": mu_plus_accuracy,
            "mu_minus_accuracy": mu_minus_accuracy,
            "mu_plus_within_two": mu_plus_within_two,
            "mu_minus_within_two": mu_minus_within_two,
            "mu_plus_mean_residual": mu_plus_mean,
            "mu_minus_mean_residual": mu_minus_mean,
            "mu_plus_std_residual": mu_plus_std,
            "mu_minus_std_residual": mu_minus_std,
            "mu_plus_accuracy_per_track": mu_plus_accuracy_per_track,
            "mu_minus_accuracy_per_track": mu_minus_accuracy_per_track,
            "mu_plus_within_two_per_track": mu_plus_within_two_per_track,
            "mu_minus_within_two_per_track": mu_minus_within_two_per_track,
            "mu_plus_mean_residual_per_track": mu_plus_mean_per_track,
            "mu_minus_mean_residual_per_track": mu_minus_mean_per_track,
            "mu_plus_std_residual_per_track": mu_plus_std_per_track,
            "mu_minus_std_residual_per_track": mu_minus_std_per_track,
        }
        return evaluation_results

    # --- Private helpers -------------------------------------------------- #

    @staticmethod
    def _subtract_found_tracks(
        X: np.ndarray,
        mu_plus_pred: np.ndarray,
        mu_minus_pred: np.ndarray,
        mu_plus_softmax: np.ndarray,
        mu_minus_softmax: np.ndarray,
        active_events: np.ndarray,
    ) -> np.ndarray:
        """Subtract found tracks from the hit matrix via soft subtraction.

        For each detector in each active event the softmax probability of the
        predicted element-ID is subtracted from the corresponding position in
        the hit matrix.  A lower bound of zero is enforced so that
        overlapping μ⁺ / μ⁻ tracks sharing the same detector element cannot
        drive the hit value negative.

        The implementation is **fully vectorised** using numpy advanced
        indexing (no Python loops over events or detectors).

        Args:
            X: Input hit matrix.  Accepted shapes:
                * ``(N, D, E, 1)`` – with trailing channel dimension.
                * ``(N, D, E)``    – without channel dimension.
            mu_plus_pred: Predicted element-IDs for μ⁺, shape ``(N, D)``.
            mu_minus_pred: Predicted element-IDs for μ⁻, shape ``(N, D)``.
            mu_plus_softmax: Softmax scores for μ⁺, shape ``(N, D, E)``.
            mu_minus_softmax: Softmax scores for μ⁻, shape ``(N, D, E)``.
            active_events: Boolean mask of shape ``(N,)`` indicating which
                events should be updated.  Inactive events are left
                untouched.

        Returns:
            Updated hit matrix with the same shape as *X*.
        """

        # Handle the optional trailing channel dimension transparently.
        has_channel = X.ndim == 4
        if has_channel:
            X_work = X[..., 0].copy()  # (N, D, E)
        else:
            X_work = X.copy()

        num_events, num_detectors, num_elements = X_work.shape

        # Build index arrays for advanced indexing:
        #   event_idx : (N, D) – broadcast event index
        #   det_idx   : (N, D) – broadcast detector index
        event_idx = np.arange(num_events)[:, None]  # (N, 1)
        det_idx = np.arange(num_detectors)[None, :]  # (1, D)

        # Expand active_events to (N, 1) for broadcasting with (N, D).
        active_mask = active_events[:, None]  # (N, 1)

        # --- μ⁺ subtraction ---
        # Clamp element indices to valid range (safety net).
        mp_elem = np.clip(mu_plus_pred, 0, num_elements - 1)  # (N, D)

        # Gather the softmax value at the predicted element for each
        # (event, detector) pair.
        mp_sub = mu_plus_softmax[event_idx, det_idx, mp_elem]  # (N, D)

        # Only subtract for active events and where the prediction is
        # non-zero (element-ID 0 encodes "no hit").
        mp_valid = active_mask & (mu_plus_pred > 0) & (mu_plus_pred < num_elements)

        # Apply subtraction.
        X_work[event_idx, det_idx, mp_elem] = np.where(
            mp_valid,
            np.maximum(0.0, X_work[event_idx, det_idx, mp_elem] - mp_sub),
            X_work[event_idx, det_idx, mp_elem],
        )

        # --- μ⁻ subtraction ---
        mm_elem = np.clip(mu_minus_pred, 0, num_elements - 1)
        mm_sub = mu_minus_softmax[event_idx, det_idx, mm_elem]
        mm_valid = active_mask & (mu_minus_pred > 0) & (mu_minus_pred < num_elements)

        X_work[event_idx, det_idx, mm_elem] = np.where(
            mm_valid,
            np.maximum(0.0, X_work[event_idx, det_idx, mm_elem] - mm_sub),
            X_work[event_idx, det_idx, mm_elem],
        )

        # Restore the channel dimension if the input had one.
        if has_channel:
            X_out = np.empty_like(X)
            X_out[..., 0] = X_work
            return X_out

        return X_work


if __name__ == "__main__":
    multi_track_finder = MultiTrackFinder()
    results = multi_track_finder.evaluate("path/to/your/test_file.root")
    print(results)
