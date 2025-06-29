import ROOT
import numpy as np
import random
from array import array
import matplotlib.pyplot as plt
import os

def compute_weight_sum(propagation_model, gaussian_sigma, exp_decay_const, max_det=62):
    weight_sum = 0.0
    for det in range(1, max_det + 1):
        if propagation_model == "linear":
            w = max(1.0 - det / 100.0, 0.0)
        elif propagation_model == "gaussian":
            w = np.exp(-0.5 * ((det - 1) / gaussian_sigma) ** 2)
        elif propagation_model == "exponential":
            w = np.exp(-det / exp_decay_const)
        else:
            w = 0.0
        weight_sum += w
    return weight_sum

def calculate_prob_cap(propagation_model,
                       gaussian_sigma,
                       exp_decay_const,
                       num_tracks,
                       desired_max_hits=100,
                       max_det=62):
    weight_sum = compute_weight_sum(propagation_model,
                                    gaussian_sigma,
                                    exp_decay_const,
                                    max_det)
    if weight_sum < 1e-6 or num_tracks < 1:
        return 1.0
    cap = desired_max_hits / (num_tracks * weight_sum)
    return min(cap, 1.0)

def inject_tracks_randomized(
    file1,
    file2,
    file3,
    output_file,
    output_log_txt,
    combinatorial_prob=0.3,  # Probability of injecting from file3
    mu_plus_minus_ratio=1.5,  # Ratio of mu+ to mu- tracks to inject (default 1.5 for 3:2 ratio)
    debug=False,  # Flag to control debug output
):
    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")
    f3 = ROOT.TFile.Open(file3, "READ")
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")
    tree3 = f3.Get("tree")

    # Check which processID branch name is used in each tree
    def get_processID_branch_name(tree):
        branches = [branch.GetName() for branch in tree.GetListOfBranches()]
        if "gProcessID" in branches:
            return "gProcessID"
        elif "processID" in branches:
            return "processID"
        else:
            raise RuntimeError(f"Neither 'gProcessID' nor 'processID' branch found in tree. Available branches: {branches}")
    
    tree1_processID_name = get_processID_branch_name(tree1)
    tree2_processID_name = get_processID_branch_name(tree2)
    tree3_processID_name = get_processID_branch_name(tree3)
    
    print(f"Tree1 (signal) uses: {tree1_processID_name}")
    print(f"Tree2 (background) uses: {tree2_processID_name}")
    print(f"Tree3 (combinatorial) uses: {tree3_processID_name}")

    def assign_occupancy():
        p=random.random()
        if p < 0.5:
            return 0
        elif p < 0.75:
            return random.randint(100,1000) # Shuffle to randomize event order
        else:  # p >= 0.75
            return random.randint(1001,2000) # Default case for remaining 25% probability

    # Pre-scan files to separate mu+ and mu- tracks
    print("Pre-scanning files to separate mu+ and mu- tracks...")
    
    tree2_mu_plus_indices = []
    tree2_mu_minus_indices = []
    tree3_mu_plus_indices = []
    tree3_mu_minus_indices = []
    
    # Scan tree2 - look at all tracks in each event
    for i in range(tree2.GetEntries()):
        tree2.GetEntry(i)
        # Check all tracks in this event
        for track_idx in range(len(tree2.gCharge)):
            if tree2.gCharge[track_idx] == 1:
                tree2_mu_plus_indices.append((i, track_idx))  # (event_index, track_index)
            elif tree2.gCharge[track_idx] == -1:
                tree2_mu_minus_indices.append((i, track_idx))  # (event_index, track_index)
    
    # Scan tree3 - look at all tracks in each event
    for i in range(tree3.GetEntries()):
        tree3.GetEntry(i)
        # Check all tracks in this event
        for track_idx in range(len(tree3.gCharge)):
            if tree3.gCharge[track_idx] == 1:
                tree3_mu_plus_indices.append((i, track_idx))  # (event_index, track_index)
            elif tree3.gCharge[track_idx] == -1:
                tree3_mu_minus_indices.append((i, track_idx))  # (event_index, track_index)
    
    print(f"Tree2: {len(tree2_mu_plus_indices)} mu+ tracks, {len(tree2_mu_minus_indices)} mu- tracks")
    print(f"Tree3: {len(tree3_mu_plus_indices)} mu+ tracks, {len(tree3_mu_minus_indices)} mu- tracks")
    
    # Initialize counters for tracking which indices we've used
    tree2_mu_plus_counter = 0
    tree2_mu_minus_counter = 0
    tree3_mu_plus_counter = 0
    tree3_mu_minus_counter = 0

    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(9)
    output_tree = ROOT.TTree("tree", "Tree with injected tracks and preserved signal hit arrays")

    eventID       = array('i', [0])
    muID          = ROOT.std.vector("int")()
    elementID     = ROOT.std.vector("int")()
    detectorID    = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    tdcTime       = ROOT.std.vector("double")()
    hitID         = ROOT.std.vector("int")()
    hitTrackID    = ROOT.std.vector("int")()
    gProcessID    = ROOT.std.vector("int")()
    gCharge       = ROOT.std.vector("int")()
    gTrackID      = ROOT.std.vector("int")()
    gpx           = ROOT.std.vector("double")()
    gpy           = ROOT.std.vector("double")()
    gpz           = ROOT.std.vector("double")()
    gvx           = ROOT.std.vector("double")()
    gvy           = ROOT.std.vector("double")()
    gvz           = ROOT.std.vector("double")()

    HitArray_mup  = np.zeros(62, dtype=np.int32)
    HitArray_mum  = np.zeros(62, dtype=np.int32)

    output_tree.Branch("eventID",        eventID,       "eventID/I")
    output_tree.Branch("muID",           muID)
    output_tree.Branch("elementID",      elementID)
    output_tree.Branch("detectorID",     detectorID)
    output_tree.Branch("driftDistance",  driftDistance)
    output_tree.Branch("tdcTime",        tdcTime)
    output_tree.Branch("hitID",          hitID)
    output_tree.Branch("hitTrackID",     hitTrackID)
    output_tree.Branch("gProcessID",     gProcessID)
    output_tree.Branch("gTrackID",       gTrackID)
    output_tree.Branch("gCharge",        gCharge)
    output_tree.Branch("gpx",            gpx)
    output_tree.Branch("gpy",            gpy)
    output_tree.Branch("gpz",            gpz)
    output_tree.Branch("gvx",            gvx)
    output_tree.Branch("gvy",            gvy)
    output_tree.Branch("gvz",            gvz)
    output_tree.Branch("HitArray_mup",   HitArray_mup,  "HitArray_mup[62]/I")
    output_tree.Branch("HitArray_mum",   HitArray_mum,  "HitArray_mum[62]/I")

    HitArray_mup_input = np.zeros(62, dtype=np.int32)
    HitArray_mum_input = np.zeros(62, dtype=np.int32)
    tree1.SetBranchAddress("HitArray_mup", HitArray_mup_input)
    tree1.SetBranchAddress("HitArray_mum", HitArray_mum_input)

    occupancies = []

    log_file = open(output_log_txt, "w")
    header = (
        "# eventID  propagation_model  gaussian_sigma  exp_decay_const  "
        "num_tracks  prob_mean  prob_width  total_hits_this_event  "
        "injected_from_file3  file3_tracks  file3_mu_plus  file3_mu_minus  mu_plus_mu_minus_ratio\n"
    )
    log_file.write(header)

    for i in range(tree1.GetEntries()):
        # Check if we've exhausted all tracks from either file
        available_mu_plus_tree2 = len(tree2_mu_plus_indices) - tree2_mu_plus_counter
        available_mu_minus_tree2 = len(tree2_mu_minus_indices) - tree2_mu_minus_counter
        available_mu_plus_tree3 = len(tree3_mu_plus_indices) - tree3_mu_plus_counter
        available_mu_minus_tree3 = len(tree3_mu_minus_indices) - tree3_mu_minus_counter
        
        # Check if either file is completely exhausted
        if available_mu_plus_tree2 <= 0 and available_mu_minus_tree2 <= 0:
            print(f"File2 (background) is exhausted of muons. Stopping at event {i}.")
            break
        if available_mu_plus_tree3 <= 0 and available_mu_minus_tree3 <= 0:
            print(f"File3 (combinatorial) is exhausted of muons. Stopping at event {i}.")
            break
        
        tree1.GetEntry(i)

        sig_eventID     = tree1.eventID
        sig_gCharge     = [tree1.gCharge[0], tree1.gCharge[1]]
        sig_gTrackIDs   = [tree1.gTrackID[0], tree1.gTrackID[1]]
        sig_gpx_list    = list(tree1.gpx)
        sig_gpy_list    = list(tree1.gpy)
        sig_gpz_list    = list(tree1.gpz)
        sig_gvx_list    = list(tree1.gvx)
        sig_gvy_list    = list(tree1.gvy)
        sig_gvz_list    = list(tree1.gvz)

        # Use the correct branch name for tree1
        if tree1_processID_name == "gProcessID":
            sig_gProcessID  = list(tree1.gProcessID)
        else:
            sig_gProcessID  = list(tree1.processID)
            
        sig_elementID   = list(tree1.elementID)
        sig_detectorID  = list(tree1.detectorID)
        sig_driftDist   = list(tree1.driftDistance)
        sig_tdcTime     = list(tree1.tdcTime)
        sig_hitID       = list(tree1.hitID)
        sig_hitTrackID  = list(tree1.hitTrackID)

        sig_HitArray_mup = HitArray_mup_input.copy()
        sig_HitArray_mum = HitArray_mum_input.copy()

        current_max_hitID   = max(tree1.hitID) if len(tree1.hitID) > 0 else 0
        next_gTrackID_base  = max(tree1.gTrackID) + 1 if len(tree1.gTrackID) > 0 else 3

        # Randomly decide whether to inject from file3
        inject_from_file3 = random.random() < combinatorial_prob
        
        if debug:
            print(f"DEBUG: Event {sig_eventID}: inject_from_file3 = {inject_from_file3} (combinatorial_prob = {combinatorial_prob})")

        # Assign occupancy for this specific event
        event_occupancy = assign_occupancy()

        # If occupancy is 0, skip injection and just write signal data
        if event_occupancy == 0:
            # Clear all vectors and add only signal data
            muID.clear()
            elementID.clear()
            detectorID.clear()
            driftDistance.clear()
            tdcTime.clear()
            hitID.clear()
            hitTrackID.clear()
            gProcessID.clear()
            gCharge.clear()
            gTrackID.clear()
            gpx.clear()
            gpy.clear()
            gpz.clear()
            gvx.clear()
            gvy.clear()
            gvz.clear()

            # Add signal data only
            muID.push_back(1)
            muID.push_back(2)
            for v in sig_gCharge:     gCharge.push_back(v)
            for v in sig_gTrackIDs:   gTrackID.push_back(v)
            for v in sig_gpx_list:     gpx.push_back(v)
            for v in sig_gpy_list:     gpy.push_back(v)
            for v in sig_gpz_list:     gpz.push_back(v)
            for v in sig_gvx_list:     gvx.push_back(v)
            for v in sig_gvy_list:     gvy.push_back(v)
            for v in sig_gvz_list:     gvz.push_back(v)
            for v in sig_gProcessID:  gProcessID.push_back(v)
            for v in sig_elementID:   elementID.push_back(v)
            for v in sig_detectorID:  detectorID.push_back(v)
            for v in sig_driftDist:   driftDistance.push_back(v)
            for v in sig_tdcTime:     tdcTime.push_back(v)
            for v in sig_hitID:       hitID.push_back(v)
            for v in sig_hitTrackID:  hitTrackID.push_back(v)

            HitArray_mup[:] = sig_HitArray_mup
            HitArray_mum[:] = sig_HitArray_mum

            occupancies.append(len(sig_hitID))  # Count signal hits only

            log_line = (
                f"{sig_eventID:8d}  "
                f"{'no_injection':11s}  "
                f"{-1:15.5f}  "
                f"{-1:17.5f}  "
                f"{0:10d}  "
                f"{0.0:9.5f}  "
                f"{0.0:10.5f}  "
                f"{len(sig_hitID):10d}  "
                f"{str(False):15s}  "
                f"{0:12d}  "
                f"{0:13d}  "
                f"{0:14d}  "
                f"{mu_plus_minus_ratio:20.3f}\n"
            )
            log_file.write(log_line)

            eventID[0] = sig_eventID
            output_tree.Fill()
            
        else:
            # Original injection logic for non-zero occupancy
            successful = False
            
            while not successful:
                # Pick a random propagation model and its hyperparameter
                propagation_model = random.choice(["linear", "gaussian", "exponential"])
                if propagation_model == "gaussian":
                    gaussian_sigma  = random.uniform(5.0, 20.0)
                    exp_decay_const = None
                elif propagation_model == "exponential":
                    exp_decay_const = random.uniform(10.0, 30.0)
                    gaussian_sigma  = None
                else:  # "linear"
                    gaussian_sigma  = None
                    exp_decay_const = None

                num_tracks = random.randint(10, 200)  # Num_tracks currently selected as between 10, 40, can be changed

                # Compute an upper‐bound on prob_mean so expected background ≤ event_occupancy
                sigma_val = gaussian_sigma if gaussian_sigma is not None else 0.0
                decay_val = exp_decay_const if exp_decay_const is not None else 1.0
                prob_cap = calculate_prob_cap(
                    propagation_model,
                    sigma_val,
                    decay_val,
                    num_tracks,
                    desired_max_hits=event_occupancy,
                    max_det=62
                )

                # Draw prob_mean ∈ [0, prob_cap]
                prob_mean = random.uniform(0.0, prob_cap)

                # Now set prob_width = 0.1 × prob_mean
                prob_width_this = prob_mean * 0.1

                # Calculate how many mu+ and mu- tracks to inject based on the ratio
                total_injected_tracks = num_tracks
                if mu_plus_minus_ratio > 0:
                    mu_plus_tracks = int(round(total_injected_tracks * mu_plus_minus_ratio / (1 + mu_plus_minus_ratio)))
                    mu_minus_tracks = total_injected_tracks - mu_plus_tracks
                else:
                    # If ratio is 0, all tracks are mu-
                    mu_plus_tracks = 0
                    mu_minus_tracks = total_injected_tracks
                
                # Check if we have enough unused tracks available
                available_mu_plus_tree2 = len(tree2_mu_plus_indices) - tree2_mu_plus_counter
                available_mu_minus_tree2 = len(tree2_mu_minus_indices) - tree2_mu_minus_counter
                available_mu_plus_tree3 = len(tree3_mu_plus_indices) - tree3_mu_plus_counter
                available_mu_minus_tree3 = len(tree3_mu_minus_indices) - tree3_mu_minus_counter
                
                # Calculate available tracks based on combinatorial probability
                if inject_from_file3:
                    available_mu_plus = available_mu_plus_tree3 + available_mu_plus_tree2
                    available_mu_minus = available_mu_minus_tree3 + available_mu_minus_tree2
                else:
                    available_mu_plus = available_mu_plus_tree2
                    available_mu_minus = available_mu_minus_tree2
                
                # If we don't have enough tracks, check if files are exhausted
                if mu_plus_tracks > available_mu_plus or mu_minus_tracks > available_mu_minus:
                    # Check if either file is completely exhausted
                    if available_mu_plus_tree2 <= 0 and available_mu_minus_tree2 <= 0:
                        print(f"File2 (background) is exhausted of muons. Stopping at event {i}.")
                        break
                    if available_mu_plus_tree3 <= 0 and available_mu_minus_tree3 <= 0:
                        print(f"File3 (combinatorial) is exhausted of muons. Stopping at event {i}.")
                        break
                    
                    print(f"Event {sig_eventID}: Not enough unused tracks available. mu+ needed: {mu_plus_tracks}, available: {available_mu_plus}; mu- needed: {mu_minus_tracks}, available: {available_mu_minus}. Retrying...")
                    break # Stop the event
                
                print(f"Event {sig_eventID}: Injecting {mu_plus_tracks} mu+ and {mu_minus_tracks} mu- tracks (ratio: {mu_plus_minus_ratio})")

                muID_trial       = ROOT.std.vector("int")()
                elementID_trial  = ROOT.std.vector("int")()
                detectorID_trial = ROOT.std.vector("int")()
                driftDist_trial  = ROOT.std.vector("double")()
                tdcTime_trial    = ROOT.std.vector("double")()
                hitID_trial      = ROOT.std.vector("int")()
                hitTrackID_trial = ROOT.std.vector("int")()
                gProcessID_trial = ROOT.std.vector("int")()
                gCharge_trial    = ROOT.std.vector("int")()
                gTrackID_trial   = ROOT.std.vector("int")()
                gpx_trial        = ROOT.std.vector("double")()
                gpy_trial        = ROOT.std.vector("double")()
                gpz_trial        = ROOT.std.vector("double")()
                gvx_trial        = ROOT.std.vector("double")()
                gvy_trial        = ROOT.std.vector("double")()
                gvz_trial        = ROOT.std.vector("double")()

                muID_trial.push_back(1)
                muID_trial.push_back(2)
                for v in sig_gCharge:     gCharge_trial.push_back(v)
                for v in sig_gTrackIDs:   gTrackID_trial.push_back(v)
                for v in sig_gpx_list:     gpx_trial.push_back(v)
                for v in sig_gpy_list:     gpy_trial.push_back(v)
                for v in sig_gpz_list:     gpz_trial.push_back(v)
                for v in sig_gvx_list:     gvx_trial.push_back(v)
                for v in sig_gvy_list:     gvy_trial.push_back(v)
                for v in sig_gvz_list:     gvz_trial.push_back(v)

                for v in sig_gProcessID:  gProcessID_trial.push_back(v)
                for v in sig_elementID:   elementID_trial.push_back(v)
                for v in sig_detectorID:  detectorID_trial.push_back(v)
                for v in sig_driftDist:   driftDist_trial.push_back(v)
                for v in sig_tdcTime:     tdcTime_trial.push_back(v)
                for v in sig_hitID:       hitID_trial.push_back(v)
                for v in sig_hitTrackID:  hitTrackID_trial.push_back(v)

                local_hitID_counter = current_max_hitID + 1
                next_gTrackID      = next_gTrackID_base
                file3_tracks_injected = 0
                file3_mu_plus_count = 0
                file3_mu_minus_count = 0

                # Create lists to store track indices for this event
                mu_plus_track_indices = []
                mu_minus_track_indices = []
                
                # Collect mu+ tracks - MODIFIED LOGIC: file3 alongside file2
                for track_idx in range(mu_plus_tracks):
                    # When inject_from_file3 is True, distribute tracks between file2 and file3
                    # Use file3 for some tracks and file2 for others
                    # Simple distribution: use file3 for even indices, file2 for odd indices (when available)
                    use_file3_for_this_track = (inject_from_file3 and 
                                               (tree3_mu_plus_counter < len(tree3_mu_plus_indices)) and
                                               (track_idx % 2 == 0))  # Even indices use file3
                    
                    if use_file3_for_this_track:
                        # Use file3 for this mu+ track
                        mu_plus_track_indices.append(('tree3', tree3_mu_plus_indices[tree3_mu_plus_counter]))
                        tree3_mu_plus_counter += 1
                        file3_tracks_injected += 1
                        file3_mu_plus_count += 1
                        if debug:
                            print(f"  DEBUG: Selected mu+ track from tree3 (event {tree3_mu_plus_indices[tree3_mu_plus_counter-1][0]}, track {tree3_mu_plus_indices[tree3_mu_plus_counter-1][1]})")
                    else:
                        # Use file2 for this mu+ track (always available as fallback)
                        if tree2_mu_plus_counter < len(tree2_mu_plus_indices):
                            mu_plus_track_indices.append(('tree2', tree2_mu_plus_indices[tree2_mu_plus_counter]))
                            tree2_mu_plus_counter += 1
                            if debug:
                                print(f"  DEBUG: Selected mu+ track from tree2 (event {tree2_mu_plus_indices[tree2_mu_plus_counter-1][0]}, track {tree2_mu_plus_indices[tree2_mu_plus_counter-1][1]})")
                        else:
                            if debug:
                                print(f"  DEBUG: WARNING: No mu+ tracks available in tree2!")
                
                # Collect mu- tracks - MODIFIED LOGIC: file3 alongside file2
                for track_idx in range(mu_minus_tracks):
                    # When inject_from_file3 is True, distribute tracks between file2 and file3
                    # Use file3 for some tracks and file2 for others
                    # Simple distribution: use file3 for even indices, file2 for odd indices (when available)
                    use_file3_for_this_track = (inject_from_file3 and 
                                               (tree3_mu_minus_counter < len(tree3_mu_minus_indices)) and
                                               (track_idx % 2 == 0))  # Even indices use file3
                    
                    if use_file3_for_this_track:
                        # Use file3 for this mu- track
                        mu_minus_track_indices.append(('tree3', tree3_mu_minus_indices[tree3_mu_minus_counter]))
                        tree3_mu_minus_counter += 1
                        file3_tracks_injected += 1
                        file3_mu_minus_count += 1
                        if debug:
                            print(f"  DEBUG: Selected mu- track from tree3 (event {tree3_mu_minus_indices[tree3_mu_minus_counter-1][0]}, track {tree3_mu_minus_indices[tree3_mu_minus_counter-1][1]})")
                    else:
                        # Use file2 for this mu- track (always available as fallback)
                        if tree2_mu_minus_counter < len(tree2_mu_minus_indices):
                            mu_minus_track_indices.append(('tree2', tree2_mu_minus_indices[tree2_mu_minus_counter]))
                            tree2_mu_minus_counter += 1
                            if debug:
                                print(f"  DEBUG: Selected mu- track from tree2 (event {tree2_mu_minus_indices[tree2_mu_minus_counter-1][0]}, track {tree2_mu_minus_indices[tree2_mu_minus_counter-1][1]})")
                        else:
                            if debug:
                                print(f"  DEBUG: WARNING: No mu- tracks available in tree2!")

                # Combine and shuffle the track indices to randomize the order
                all_track_indices = mu_plus_track_indices + mu_minus_track_indices
                random.shuffle(all_track_indices)

                # Process all tracks in the shuffled order
                for source_tree_name, track_index in all_track_indices:
                    if source_tree_name == 'tree2':
                        tree2.GetEntry(track_index[0])
                        source_tree = tree2
                        source_processID_name = tree2_processID_name
                    else:  # tree3
                        tree3.GetEntry(track_index[0])
                        source_tree = tree3
                        source_processID_name = tree3_processID_name

                    this_gTrackID = next_gTrackID
                    next_gTrackID += 1
                    gTrackID_trial.push_back(this_gTrackID)
                    gCharge_trial.push_back(source_tree.gCharge[track_index[1]])
                    gpx_trial.push_back(source_tree.gpx[track_index[1]])
                    gpy_trial.push_back(source_tree.gpy[track_index[1]])
                    gpz_trial.push_back(source_tree.gpz[track_index[1]])
                    gvx_trial.push_back(source_tree.gvx[track_index[1]])
                    gvy_trial.push_back(source_tree.gvy[track_index[1]])
                    gvz_trial.push_back(source_tree.gvz[track_index[1]])

                    probability = np.clip(
                        np.random.normal(prob_mean, prob_width_this),
                        0.0,
                        1.0
                    )

                    # Get the processID data using the correct branch name
                    if source_processID_name == "gProcessID":
                        source_processID_data = source_tree.gProcessID
                    else:
                        source_processID_data = source_tree.processID

                    for procID, elem, det, dist, tdc in zip(
                        source_processID_data,
                        source_tree.elementID,
                        source_tree.detectorID,
                        source_tree.driftDistance,
                        source_tree.tdcTime
                    ):
                        if propagation_model == "linear":
                            weight = max(1.0 - det / 100.0, 0.0)
                        elif propagation_model == "gaussian":
                            weight = np.exp(-0.5 * ((det - 1) / gaussian_sigma) ** 2)
                        elif propagation_model == "exponential":
                            weight = np.exp(-det / exp_decay_const)
                        else:
                            weight = 0.0

                        if np.random.random() < probability * weight:
                            gProcessID_trial.push_back(procID)
                            elementID_trial.push_back(elem)
                            detectorID_trial.push_back(det)
                            driftDist_trial.push_back(dist)
                            tdcTime_trial.push_back(tdc)
                            hitID_trial.push_back(local_hitID_counter)
                            hitTrackID_trial.push_back(this_gTrackID)
                            local_hitID_counter += 1

                total_hits_trial = elementID_trial.size()
                
                # If total_hits_trial ≤ event_occupancy, commit; otherwise, retry.
                if total_hits_trial <= event_occupancy:
                    muID.clear()
                    elementID.clear()
                    detectorID.clear()
                    driftDistance.clear()
                    tdcTime.clear()
                    hitID.clear()
                    hitTrackID.clear()
                    gProcessID.clear()
                    gCharge.clear()
                    gTrackID.clear()
                    gpx.clear()
                    gpy.clear()
                    gpz.clear()
                    gvx.clear()
                    gvy.clear()
                    gvz.clear()

                    for v in muID_trial:        muID.push_back(v)
                    for v in gCharge_trial:     gCharge.push_back(v)
                    for v in gTrackID_trial:    gTrackID.push_back(v)
                    for v in gpx_trial:         gpx.push_back(v)
                    for v in gpy_trial:         gpy.push_back(v)
                    for v in gpz_trial:         gpz.push_back(v)
                    for v in gvx_trial:         gvx.push_back(v)
                    for v in gvy_trial:         gvy.push_back(v)
                    for v in gvz_trial:         gvz.push_back(v)
                    for v in gProcessID_trial:  gProcessID.push_back(v)
                    for v in elementID_trial:   elementID.push_back(v)
                    for v in detectorID_trial:  detectorID.push_back(v)
                    for v in driftDist_trial:   driftDistance.push_back(v)
                    for v in tdcTime_trial:     tdcTime.push_back(v)
                    for v in hitID_trial:       hitID.push_back(v)
                    for v in hitTrackID_trial:  hitTrackID.push_back(v)

                    HitArray_mup[:] = sig_HitArray_mup
                    HitArray_mum[:] = sig_HitArray_mum

                    occupancies.append(total_hits_trial)

                    log_line = (
                        f"{sig_eventID:8d}  "
                        f"{propagation_model:11s}  "
                        f"{(gaussian_sigma if gaussian_sigma is not None else -1):15.5f}  "
                        f"{(exp_decay_const if exp_decay_const is not None else -1):17.5f}  "
                        f"{num_tracks:10d}  "
                        f"{prob_mean:9.5f}  "
                        f"{prob_width_this:10.5f}  "
                        f"{total_hits_trial:10d}  "
                        f"{str(inject_from_file3):15s}  "
                        f"{file3_tracks_injected:12d}  "
                        f"{file3_mu_plus_count:13d}  "
                        f"{file3_mu_minus_count:14d}  "
                        f"{mu_plus_minus_ratio:20.3f}\n"
                    )
                    log_file.write(log_line)

                    eventID[0] = sig_eventID
                    output_tree.Fill()
                    successful = True
                
                else:
                    print(f"Event {sig_eventID} has {total_hits_trial} hits, which exceeds the desired occupancy of {event_occupancy}. Retrying...")
                    # Reset counters to reuse the same tracks for the next attempt
                    tree2_mu_plus_counter -= len([idx for idx in mu_plus_track_indices if idx[0] == 'tree2'])
                    tree2_mu_minus_counter -= len([idx for idx in mu_minus_track_indices if idx[0] == 'tree2'])
                    tree3_mu_plus_counter -= len([idx for idx in mu_plus_track_indices if idx[0] == 'tree3'])
                    tree3_mu_minus_counter -= len([idx for idx in mu_minus_track_indices if idx[0] == 'tree3'])
                    continue
            
            # If we broke out of the while loop due to file exhaustion, break out of the main loop too
            if not successful:
                break

    fout.Write()
    fout.Close()
    f1.Close()
    f2.Close()
    f3.Close()
    log_file.close()
    if occupancies:
        mean_occ   = float(np.mean(occupancies))
        median_occ = float(np.median(occupancies))
        print(f"\n=== Occupancy summary over {len(occupancies)} events ===")
        print(f"  • mean hits/event    = {mean_occ:.1f}")
        print(f"  • median hits/event  = {median_occ:.1f}")
        print(f"  • min/max hits/event = {min(occupancies)}/{max(occupancies)}")

        plt.figure(figsize=(10, 6))
        plt.hist(occupancies, bins=70,
                 edgecolor='black', align='left')
        plt.xlabel("Number of Hits per Event")
        plt.ylabel("Event Count")
        plt.title("Histogram of Hits per Event")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("occupancy_histogram.png")
    
    else:
        print("No events were processed.")
    
    return occupancies

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inject randomized background tracks with dynamic occupancy control."
    )
    parser.add_argument("file1", type=str, help="Path to the finder_training.root (signal).")
    parser.add_argument("file2", type=str, help="Path to the background file (tree2).")
    parser.add_argument("file3", type=str, help="Path to the combinatorial data file (tree3).")
    parser.add_argument(
        "--output", type=str, default="mc_events_randomized.root",
        help="Output ROOT file name."
    )
    parser.add_argument(
        "--log_txt", type=str, default="hyperparams_per_event.txt",
        help="TXT file recording hyperparameters per event."
    )
    parser.add_argument(
        "--combinatorial_prob", type=float, default=0.1,
        help="Probability of injecting tracks from file3 (combinatorial data)."
    )
    parser.add_argument(
        "--mu_plus_minus_ratio", type=float, default=1.5,
        help="Ratio of mu+ to mu- tracks to inject (default 1.5 for 3:2 ratio)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output to track track selection."
    )
    args = parser.parse_args()

    occupancies = inject_tracks_randomized(
        args.file1,
        args.file2,
        args.file3,
        "temp_output.root",
        args.log_txt,
        combinatorial_prob=args.combinatorial_prob,
        mu_plus_minus_ratio=args.mu_plus_minus_ratio,
        debug=args.debug
    )

    if occupancies:
        output_name = f"mc_events_randomized_{round(len(occupancies), 3)}.root"
        os.rename("temp_output.root", output_name)
        print(f"\nRenamed output file to: {output_name}")
    else:
        print("No events were processed, output file not created.") 