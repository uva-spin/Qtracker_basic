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
