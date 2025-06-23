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
    output_file,
    output_log_txt,
    desired_max_hits=100,
):
    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    def assign_occupancy():
        p=random.random()
        if p< 0.5:
            return 0
        elif p < 0.75:
            return random.randint(50,75)
        else:
            return random.randint(76,100) # Shuffle to randomize event order

   
    occupancy = assign_occupancy()

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

    num_events_tree2 = tree2.GetEntries()
    tree2_index = 0
    occupancies = []

    log_file = open(output_log_txt, "w")
    header = (
        "# eventID  propagation_model  gaussian_sigma  exp_decay_const  "
        "num_tracks  prob_mean  prob_width  total_hits_this_event\n"
    )
    log_file.write(header)

    for i in range(tree1.GetEntries()):
        if tree2_index >= num_events_tree2:
            break

        tree1.GetEntry(i)
        tree2.GetEntry(tree2_index)

        sig_eventID     = tree1.eventID
        sig_gCharge     = [tree1.gCharge[0], tree1.gCharge[1]]
        sig_gTrackIDs   = [tree1.gTrackID[0], tree1.gTrackID[1]]
        sig_gpx_list    = list(tree1.gpx)
        sig_gpy_list    = list(tree1.gpy)
        sig_gpz_list    = list(tree1.gpz)
        sig_gvx_list    = list(tree1.gvx)
        sig_gvy_list    = list(tree1.gvy)
        sig_gvz_list    = list(tree1.gvz)

        sig_gProcessID  = list(tree1.gProcessID)
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

            # Compute an upper‐bound on prob_mean so expected background ≤ desired_max_hits
            sigma_val = gaussian_sigma if gaussian_sigma is not None else 0.0
            decay_val = exp_decay_const if exp_decay_const is not None else 1.0
            prob_cap = calculate_prob_cap(
                propagation_model,
                sigma_val,
                decay_val,
                num_tracks,
                desired_max_hits=occupancy,
                max_det=62
            )

            # Draw prob_mean ∈ [0, prob_cap]
            prob_mean = random.uniform(0.0, prob_cap)

            # Now set prob_width = 0.1 × prob_mean
            prob_width_this = prob_mean * 0.1

            # Simulate injection on a peeked subset of tree2 entries
            temp_index = tree2_index

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

            for _ in range(num_tracks):
                if temp_index >= num_events_tree2:
                    break
                tree2.GetEntry(temp_index)
                temp_index += 1

                this_gTrackID = next_gTrackID
                next_gTrackID += 1
                gTrackID_trial.push_back(this_gTrackID)
                gCharge_trial.push_back(tree2.gCharge[0])
                gpx_trial.push_back(tree2.gpx[0])
                gpy_trial.push_back(tree2.gpy[0])
                gpz_trial.push_back(tree2.gpz[0])
                gvx_trial.push_back(tree2.gvx[0])
                gvy_trial.push_back(tree2.gvy[0])
                gvz_trial.push_back(tree2.gvz[0])

                probability = np.clip(
                    np.random.normal(prob_mean, prob_width_this),
                    0.0,
                    1.0
                )

                for procID, elem, det, dist, tdc in zip(
                    tree2.gProcessID,
                    tree2.elementID,
                    tree2.detectorID,
                    tree2.driftDistance,
                    tree2.tdcTime
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
            
            # If total_hits_trial ≤ desired_max_hits, commit; otherwise, retry.
            if total_hits_trial <= desired_max_hits:
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

                tree2_index = temp_index
                occupancies.append(total_hits_trial)

                log_line = (
                    f"{sig_eventID:8d}  "
                    f"{propagation_model:11s}  "
                    f"{(gaussian_sigma if gaussian_sigma is not None else -1):15.5f}  "
                    f"{(exp_decay_const if exp_decay_const is not None else -1):17.5f}  "
                    f"{num_tracks:10d}  "
                    f"{prob_mean:9.5f}  "
                    f"{prob_width_this:10.5f}  "
                    f"{total_hits_trial:10d}\n"
                )
                log_file.write(log_line)

                eventID[0] = sig_eventID
                output_tree.Fill()
                successful = True
        
            # else: occupancy > desired_max_hits → loop back and retry

    fout.Write()
    fout.Close()
    f1.Close()
    f2.Close()
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
        description="Inject randomized background tracks, capping occupancy under 80."
    )
    parser.add_argument("file1", type=str, help="Path to the finder_training.root (signal).")
    parser.add_argument("file2", type=str, help="Path to the background file (tree2).")
    parser.add_argument(
        "--output", type=str, default="mc_events_randomized_{round(len(occupancies)),2}.root",
        help="Output ROOT file name."
    )
    parser.add_argument(
        "--log_txt", type=str, default="hyperparams_per_event.txt",
        help="TXT file recording hyperparameters per event."
    )
    parser.add_argument(
        "--max_hits", type=int, default=100,
        help="Strict upper bound on total occupancy per event."
    )
    args = parser.parse_args()

    occupancies = inject_tracks_randomized(
        args.file1,
        args.file2,
        "temp_output.root",
        args.log_txt,
        desired_max_hits=args.max_hits
    )

    if occupancies:
        output_name = f"mc_events_randomized_{round(len(occupancies), 3)}.root"
        os.rename("temp_output.root", output_name)
        print(f"\nRenamed output file to: {output_name}")
    else:
        print("No events were processed, output file not created."
    )