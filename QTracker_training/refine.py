import numpy as np


def refine_hit_arrays(hit_array_mup, hit_array_mum, detectorIDs, elementIDs):
    """
    Refines the HitArrays by replacing inferred elementIDs with the closest actual elementID
    using the detectorID and elementID vectors. Returns 0 if no actual hits exist.
    Optimized for speed.
    """

    def find_closest_actual_hit(
        detector_id, inferred_element, detectorIDs_event, elementIDs_event
    ):
        """
        Finds the closest actual hit to the inferred_element for a specific detector_id.
        Returns 0 if no hits exist.
        """
        if inferred_element == 0:
            return 0  # Preserve 0 values (no hit).

        # Filter elementIDs for the given detector_id
        actual_elementIDs = elementIDs_event[detectorIDs_event == detector_id]

        if len(actual_elementIDs) == 0:
            return 0  # Return 0 if no hits exist.

        # Find the closest actual hit elementID using NumPy's vectorized operations
        closest_elementID = actual_elementIDs[
            np.argmin(np.abs(actual_elementIDs - inferred_element))
        ]

        return closest_elementID

    # Initialize refined arrays
    refined_mup = np.zeros_like(hit_array_mup)
    refined_mum = np.zeros_like(hit_array_mum)

    num_events, num_detectors = hit_array_mup.shape

    # Precompute detector IDs (1-based to match detector_id in the ROOT file)
    detector_ids = np.arange(1, num_detectors + 1)

    # Iterate over events
    for event in range(num_events):
        # Convert detectorIDs and elementIDs to NumPy arrays for faster processing
        detectorIDs_event = np.array(detectorIDs[event], dtype=np.int32)
        elementIDs_event = np.array(elementIDs[event], dtype=np.int32)

        # Iterate over detectors
        for detector in range(num_detectors):
            # Get inferred elementIDs for mu+ and mu-
            inferred_mup = hit_array_mup[event, detector]
            inferred_mum = hit_array_mum[event, detector]

            # Find the closest actual hits
            refined_mup[event, detector] = find_closest_actual_hit(
                detector_ids[detector],
                inferred_mup,
                detectorIDs_event,
                elementIDs_event,
            )
            refined_mum[event, detector] = find_closest_actual_hit(
                detector_ids[detector],
                inferred_mum,
                detectorIDs_event,
                elementIDs_event,
            )

    return refined_mup, refined_mum


def refine_hit_arrays_multi(
    hit_array_mup_multi, hit_array_mum_multi, detectorIDs, elementIDs
):
    """
    Wrapper to refine hit arrays for multiple pairs by reshaping.
    This allows batch processing of multiple pairs using the existing single-pair refinement logic.

    Args:
        hit_array_mup_multi (np.ndarray): Hit arrays for mu+ (shape: num_events, max_pairs, 62)
        hit_array_mum_multi (np.ndarray): Hit arrays for mu- (shape: num_events, max_pairs, 62)
        detectorIDs (list of np.ndarray): List of detectorID vectors for each event
        elementIDs (list of np.ndarray): List of elementID vectors for each event

    Returns:
        tuple: (refined_mup_multi, refined_mum_multi) each with shape (num_events, max_pairs, 62)
    """
    num_events, max_pairs, num_detectors = hit_array_mup_multi.shape

    # Reshape to (num_events * max_pairs, 62) for batch processing
    mup_flat = hit_array_mup_multi.reshape(-1, num_detectors)
    mum_flat = hit_array_mum_multi.reshape(-1, num_detectors)

    # Repeat detectorIDs and elementIDs for each pair
    det_repeated = detectorIDs * max_pairs
    elem_repeated = elementIDs * max_pairs

    # Refine using existing function
    refined_mup_flat, refined_mum_flat = refine_hit_arrays(
        mup_flat, mum_flat, det_repeated, elem_repeated
    )

    # Reshape back to (num_events, max_pairs, 62)
    refined_mup = refined_mup_flat.reshape(num_events, max_pairs, num_detectors)
    refined_mum = refined_mum_flat.reshape(num_events, max_pairs, num_detectors)

    return refined_mup, refined_mum
