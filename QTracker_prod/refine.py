import numpy as np
import tensorflow as tf


def refine_hit_arrays(hit_array_mup, hit_array_mum, detectorIDs, elementIDs):
    """
    Refines the HitArrays by replacing inferred elementIDs with the closest actual elementID
    using the detectorID and elementID vectors. Returns 0 if no actual hits exist.
    Optimized for speed.
    """
    def find_closest_actual_hit(detector_id, inferred_element, detectorIDs_event, elementIDs_event):
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
        closest_elementID = actual_elementIDs[np.argmin(np.abs(actual_elementIDs - inferred_element))]

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
                detector_ids[detector], inferred_mup, detectorIDs_event, elementIDs_event
            )
            refined_mum[event, detector] = find_closest_actual_hit(
                detector_ids[detector], inferred_mum, detectorIDs_event, elementIDs_event
            )

    return refined_mup, refined_mum


def refine_hit_arrays_v2(pred_mup, pred_mum, detectorIDs, elementIDs, softmax_mup, softmax_mum, k=1):
    """ David's detector-based refinement function. """
    
    N, NUM_DETECTORS = pred_mup.shape

    refined_mup = np.zeros_like(pred_mup, dtype=np.int32)
    refined_mum = np.zeros_like(pred_mum, dtype=np.int32)

    # Masking unused detectors (index 6~11, 54~61)
    mask = np.ones(NUM_DETECTORS, dtype=bool)
    mask[6:12] = False
    mask[54:62] = False

    # Splitting detectors into multiple stations based on their hit behavior
    region_map = {}
    for d in range(NUM_DETECTORS):
        if not mask[d]:
            region_map[d] = 'masked'
        elif 0 <= d <= 5:
            region_map[d] = 'st1'
        elif 12 <= d <= 17:
            region_map[d] = 'st2'
        elif 18 <= d <= 24:
            region_map[d] = 'st3'
        elif 25 <= d <= 31:
            region_map[d] = 'st4'
        elif 32 <= d <= 33:
            region_map[d] = 'st5'
        elif 36 <= d <= 39:
            region_map[d] = 'st6'
        elif 44 <= d <= 45:
            region_map[d] = 'st7'
        elif 46 <= d <= 53:
            region_map[d] = 'st8'
        elif d in [34, 40, 42]:  # Special (mu+)
            region_map[d] = 'muplus'
        elif d in [35, 41, 43]:  # Special (mu-)
            region_map[d] = 'muminus'
        else:
            region_map[d] = 'etc'

    region2dets = {}
    for d in range(NUM_DETECTORS):
        region = region_map[d]
        region2dets.setdefault(region, []).append(d)

    # Refine
    for i in range(N):
        det_ids_evt = np.array(detectorIDs[i], dtype=np.int32)
        elem_ids_evt = np.array(elementIDs[i], dtype=np.int32)

        # Priority given based on total softmax over each station
        region_assign_order = {}
        for region, det_list in region2dets.items():
            if region in ['masked', 'muplus', 'muminus']:
                continue
            region_softmax_plus = np.sum(softmax_mup[i, det_list, :])
            region_softmax_minus = np.sum(softmax_mum[i, det_list, :])
            if region_softmax_plus >= region_softmax_minus:
                region_assign_order[region] = ['plus', 'minus']
            else:
                region_assign_order[region] = ['minus', 'plus']

        for d in range(NUM_DETECTORS):
            region = region_map[d]
            if region == 'masked':
                refined_mup[i, d] = 0
                refined_mum[i, d] = 0
                continue

            # Special detectors treated separately
            if region == 'muplus':
                pred = pred_mup[i, d]
                if pred == 0:
                    refined_mup[i, d] = 0
                else:
                    actual_elems = elem_ids_evt[det_ids_evt == (d+1)]
                    candidates = [e for e in actual_elems if abs(e - pred) <= k]
                    if len(candidates) == 0:
                        refined_mup[i, d] = 0
                    else:
                        best = candidates[np.argmin(np.abs(np.array(candidates) - pred))]
                        refined_mup[i, d] = best
                refined_mum[i, d] = 0
                continue

            if region == 'muminus':
                pred = pred_mum[i, d]
                if pred == 0:
                    refined_mum[i, d] = 0
                else:
                    actual_elems = elem_ids_evt[det_ids_evt == (d+1)]
                    candidates = [e for e in actual_elems if abs(e - pred) <= k]
                    if len(candidates) == 0:
                        refined_mum[i, d] = 0
                    else:
                        best = candidates[np.argmin(np.abs(np.array(candidates) - pred))]
                        refined_mum[i, d] = best
                refined_mup[i, d] = 0
                continue

            # Normal region (station)
            assign_order = region_assign_order.get(region, ['plus', 'minus'])
            assigned_elem = set()
            pred_elem = {'plus': pred_mup[i, d], 'minus': pred_mum[i, d]}
            for track in assign_order:
                pred = pred_elem[track]
                if pred == 0:
                    if track == 'plus': refined_mup[i, d] = 0
                    else: refined_mum[i, d] = 0
                    continue
                actual_elems = elem_ids_evt[det_ids_evt == (d+1)]
                candidates = [e for e in actual_elems if e not in assigned_elem and abs(e - pred) <= k]
                if len(candidates) == 0:
                    if track == 'plus': refined_mup[i, d] = 0
                    else: refined_mum[i, d] = 0
                    continue
                distances = np.abs(np.array(candidates) - pred)
                best = candidates[np.argmin(distances)]
                assigned_elem.add(best)
                if track == 'plus': refined_mup[i, d] = best
                else: refined_mum[i, d] = best

    # Some statistics for K value, number of hits before and after refinement
    pre_muplus_hits = (pred_mup != 0).sum(axis=0)
    pre_muminus_hits = (pred_mum != 0).sum(axis=0)
    post_muplus_hits = (refined_mup != 0).sum(axis=0)
    post_muminus_hits = (refined_mum != 0).sum(axis=0)

    print(f"\n[k={k}] refine before/after mu+/mu- hit (event):")
    print(f"{'Det':>4} | {'pred_mu+':>9} | {'refined_mu+':>11} | {'pred_mu-':>9} | {'refined_mu-':>11}")
    print("-"*54)
    for d in range(NUM_DETECTORS):
        if not mask[d]:
            continue
        print(f"{d+1:4d} | {pre_muplus_hits[d]:9d} | {post_muplus_hits[d]:11d} | {pre_muminus_hits[d]:9d} | {post_muminus_hits[d]:11d}")

    print("\n[For all events]")
    print(f"mu+ hit (before refine): {pre_muplus_hits.sum()}   After refine: {post_muplus_hits.sum()}")
    print(f"mu- hit (before refine): {pre_muminus_hits.sum()}   After refine: {post_muminus_hits.sum()}")

    return refined_mup, refined_mum


def refine_hit_arrays_v3(hit_array_mup, hit_array_mum, detectorIDs, elementIDs, softmax_mup, softmax_mum, prob_threshold=0.5):
    """
    Donghwa's softmax-based refinement function.
    """
    def find_closest_actual_hit(detector_id, inferred_element, detectorIDs_event, elementIDs_event):
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
        closest_elementID = actual_elementIDs[np.argmin(np.abs(actual_elementIDs - inferred_element))]

        return closest_elementID

    # Get probability arrays for mu+ and mu-
    prob_mup = tf.reduce_max(softmax_mup, axis=-1).numpy()
    prob_mum = tf.reduce_max(softmax_mum, axis=-1).numpy()

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
            # If probability below threshold, set refined hits to 0
            if prob_mup[event, detector] >= prob_threshold:
                inferred_mup = hit_array_mup[event, detector]
                refined_mup[event, detector] = find_closest_actual_hit(
                    detector_ids[detector], inferred_mup, detectorIDs_event, elementIDs_event
                )
            
            if prob_mum[event, detector] >= prob_threshold:
                inferred_mum = hit_array_mum[event, detector]
                refined_mum[event, detector] = find_closest_actual_hit(
                    detector_ids[detector], inferred_mum, detectorIDs_event, elementIDs_event
                )

    return refined_mup, refined_mum
