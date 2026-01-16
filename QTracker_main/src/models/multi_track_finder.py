import tensorflow as tf
from src.config import SINGLE_TRACK_FINDER_PATH
from src.layers.axial_attention import AxialAttention


class MultiTrackFinder:
    def __init__(self):
        self.track_finder = tf.keras.models.load_model(
            SINGLE_TRACK_FINDER_PATH,
            compile=False,
            custom_objects={"AxialAttention": AxialAttention},
        )

    # --- Public interface --- #
    def run(self):
        pass

    # --- Private helpers --- #
    def _run_single_track_finder(self):
        pass

    def _subtract_found_tracks(self):
        pass
