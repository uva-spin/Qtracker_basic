import numpy as np
import tensorflow as tf
from src.config import SINGLE_TRACK_FINDER_PATH
from src.preprocessing.data_loader import (
    load_data_with_labels,
    load_data_without_labels,
    load_detector_element_data,
)
from src.layers.axial_attention import AxialAttention
from src.models.refiner import Refiner
from typing import Union


class MultiTrackFinder:
    def __init__(self):
        self.track_finder = tf.keras.models.load_model(
            SINGLE_TRACK_FINDER_PATH,
            compile=False,
            custom_objects={"AxialAttention": AxialAttention},
        )
        self.refiner = Refiner()

    # --- Public interface --- #
    def run(
        self, input_root_file: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the multi-track finding algorithm on the input ROOT file.
        Uses an iterative approach to find multiple tracks, continuing until the confidence score drops below 0.5.

        Args:
            input_root_file (str): Path to the input ROOT file.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four arrays containing the refined hit predictions and softmax scores for mu+ and mu- tracks.
        """
        X = load_data_without_labels(input_root_file)
        if X is None:
            raise ValueError("No data found in the provided ROOT file.")

        # Predict tracks using the single track finder model
        mu_plus_softmax = []
        mu_minus_softmax = []

        mu_plus_tracks = []
        mu_minus_tracks = []

        confidence_score = tf.constant(1.0)
        while confidence_score.numpy().mean() >= 0.5:
            _, y_pred, confidence_score = self.track_finder.predict(
                tf.cast(X, tf.float32)
            )

            # Extract predicted Hit Arrays and softmax scores
            mp_softmax = y_pred[:, 0, :, :]
            mm_softmax = y_pred[:, 1, :, :]

            mu_plus_pred = tf.cast(tf.argmax(mp_softmax, axis=-1), tf.int32).numpy()
            mu_minus_pred = tf.cast(tf.argmax(mm_softmax, axis=-1), tf.int32).numpy()

            # Refine predicted Hit Arrays
            detector_id, element_id, _, _ = load_detector_element_data(input_root_file)
            (
                refined_mu_plus_pred,
                refined_mu_minus_pred,
            ) = self.refiner.refine_hit_arrays(
                mu_plus_pred, mu_minus_pred, detector_id, element_id
            )

            mu_plus_softmax.append(mp_softmax)
            mu_minus_softmax.append(mm_softmax)
            mu_plus_tracks.append(refined_mu_plus_pred)
            mu_minus_tracks.append(refined_mu_minus_pred)

            # Subtract found tracks from input data
            X = self._subtract_found_tracks(
                X, refined_mu_plus_pred, refined_mu_minus_pred, mp_softmax, mm_softmax
            )

        mu_plus_softmax = np.stack(mu_plus_softmax, axis=1)
        mu_minus_softmax = np.stack(mu_minus_softmax, axis=1)
        mu_plus_tracks = np.stack(mu_plus_tracks, axis=1)
        mu_minus_tracks = np.stack(mu_minus_tracks, axis=1)

        return mu_plus_tracks, mu_minus_tracks, mu_plus_softmax, mu_minus_softmax

    def evaluate(self, input_root_file: str) -> dict[str, Union[float, list[float]]]:
        """
        Evaluate the multi-track finder model on the input ROOT file with known labels.

        Args:
            input_root_file (str): Path to the input ROOT file.
        Returns:
            dict[str, Union[float, list[float]]]: A dictionary containing evaluation metrics.
        """
        X, y_mu_plus, y_mu_minus = load_data_with_labels(input_root_file)
        if any(v is None for v in [X, y_mu_plus, y_mu_minus]):
            raise ValueError("No data found in the provided ROOT file.")

        # Predict tracks using the multi-track finder model
        refined_mu_plus_pred, refined_mu_minus_pred, _, _ = self.run(input_root_file)

        # Prepare true labels
        y_true = np.stack([y_mu_plus, y_mu_minus], axis=1)
        mu_plus_true = y_true[:, 0, :].astype(np.int32)
        mu_minus_true = y_true[:, 1, :].astype(np.int32)

        # Calculate residuals
        mu_plus_residuals = mu_plus_true - refined_mu_plus_pred
        mu_minus_residuals = mu_minus_true - refined_mu_minus_pred

        # Calculate evaluation metrics
        mu_plus_accuracy = np.mean(mu_plus_residuals == 0)
        mu_minus_accuracy = np.mean(mu_minus_residuals == 0)

        mu_plus_within_two = np.mean(np.abs(mu_plus_residuals) <= 2)
        mu_minus_within_two = np.mean(np.abs(mu_minus_residuals) <= 2)

        mu_plus_mean = np.mean(np.abs(mu_plus_residuals))
        mu_minus_mean = np.mean(np.abs(mu_minus_residuals))

        mu_plus_std = np.std(np.abs(mu_plus_residuals))
        mu_minus_std = np.std(np.abs(mu_minus_residuals))

        # Calculate per-track metrics
        mu_plus_accuracy_per_track = [
            np.mean(mu_plus_track_residuals == 0)
            for mu_plus_track_residuals in mu_plus_residuals
        ]
        mu_minus_accuracy_per_track = [
            np.mean(mu_minus_track_residuals == 0)
            for mu_minus_track_residuals in mu_minus_residuals
        ]

        mu_plus_within_two_per_track = [
            np.mean(np.abs(mu_plus_track_residuals) <= 2)
            for mu_plus_track_residuals in mu_plus_residuals
        ]
        mu_minus_within_two_per_track = [
            np.mean(np.abs(mu_minus_track_residuals) <= 2)
            for mu_minus_track_residuals in mu_minus_residuals
        ]

        mu_plus_mean_per_track = [
            np.mean(np.abs(mu_plus_track_residuals))
            for mu_plus_track_residuals in mu_plus_residuals
        ]
        mu_minus_mean_per_track = [
            np.mean(np.abs(mu_minus_track_residuals))
            for mu_minus_track_residuals in mu_minus_residuals
        ]

        mu_plus_std_per_track = [
            np.std(np.abs(mu_plus_track_residuals))
            for mu_plus_track_residuals in mu_plus_residuals
        ]
        mu_minus_std_per_track = [
            np.std(np.abs(mu_minus_track_residuals))
            for mu_minus_track_residuals in mu_minus_residuals
        ]

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

    # --- Private helpers --- #
    def _subtract_found_tracks(
        self,
        X: np.ndarray,
        mu_plus_pred: np.ndarray,
        mu_minus_pred: np.ndarray,
        mu_plus_softmax: np.ndarray,
        mu_minus_softmax: np.ndarray,
    ) -> np.ndarray:
        """
        Subtract the found tracks from the input data to prepare for the next iteration.
        Use soft subtraction to avoid abrupt changes in the data:
        - Instead of zeroing out the hits corresponding to found tracks, reduce the hit values by a certain factor.
        - Example factors: highest softmax value for that detector.

        Args:
            X (np.ndarray): Input data array. Shape: (num_events, num_detectors, num_elements).
            mu_plus_pred (np.ndarray): Predicted hit array for mu+ tracks. Shape: (num_events, num_detectors).
            mu_minus_pred (np.ndarray): Predicted hit array for mu- tracks. Shape: (num_events, num_detectors).
            mu_plus_softmax (np.ndarray): Softmax scores for mu+ tracks. Shape: (num_events, num_detectors, num_elements).
            mu_minus_softmax (np.ndarray): Softmax scores for mu- tracks. Shape: (num_events, num_detectors, num_elements).
        Returns:
            np.ndarray: Updated input data array with found tracks subtracted. Shape: (num_events, num_detectors, num_elements).
        """
        X_updated = X.copy()
        num_events, num_detectors, num_elements = X.shape

        for event_idx in range(num_events):
            for detector_idx in range(num_detectors):
                # Subtract mu+ track hits
                mu_plus_element_idx = mu_plus_pred[event_idx, detector_idx]
                if 0 <= mu_plus_element_idx < num_elements:
                    subtraction_value = mu_plus_softmax[
                        event_idx, detector_idx, mu_plus_element_idx
                    ]
                    X_updated[event_idx, detector_idx, mu_plus_element_idx] = max(
                        0,
                        X_updated[event_idx, detector_idx, mu_plus_element_idx]
                        - subtraction_value,
                    )

                # Subtract mu- track hits
                mu_minus_element_idx = mu_minus_pred[event_idx, detector_idx]
                if 0 <= mu_minus_element_idx < num_elements:
                    subtraction_value = mu_minus_softmax[
                        event_idx, detector_idx, mu_minus_element_idx
                    ]
                    X_updated[event_idx, detector_idx, mu_minus_element_idx] = max(
                        0,
                        X_updated[event_idx, detector_idx, mu_minus_element_idx]
                        - subtraction_value,
                    )

        return X_updated


if __name__ == "__main__":
    multi_track_finder = MultiTrackFinder()
    results = multi_track_finder.evaluate("path/to/your/test_file.root")
    print(results)
