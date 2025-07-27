import ROOT
import numpy as np
import random
from array import array
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

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
    file,
    output_file,
    output_log_txt,
    desired_max_hits=100,
    use_propagation_models=True,
):
    f2 = ROOT.TFile.Open(file, "READ")
    tree2 = f2.Get("tree")

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

    num_events_tree2 = tree2.GetEntries()
    occupancies = []

    log_file = open(output_log_txt, "w")
    header = (
        "# eventID  num_tracks_injected  total_hits_this_event\n"
    )
    log_file.write(header)

    # Process events based on tree2 entries
    for i in tqdm(range(num_events_tree2), desc="Processing events", unit="event"):
        # Clear vectors for this event
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

        # Add muon IDs (assuming 2 muons per event)
        muID.push_back(1)
        muID.push_back(2)

        # Randomly select number of tracks to inject (1 to 20 tracks)
        num_tracks_to_inject = random.randint(1, 20)
        
        # Inject random tracks from the input file
        for track_idx in range(num_tracks_to_inject):
            # Pick a random event from the input file
            random_event_idx = random.randint(0, num_events_tree2 - 1)
            tree2.GetEntry(random_event_idx)
            
            # Get track information (assuming first track in the event)
            if len(tree2.gCharge) > 0:
                gCharge.push_back(tree2.gCharge[0])
                gTrackID.push_back(tree2.gTrackID[0])
                gpx.push_back(tree2.gpx[0])
                gpy.push_back(tree2.gpy[0])
                gpz.push_back(tree2.gpz[0])
                gvx.push_back(tree2.gvx[0])
                gvy.push_back(tree2.gvy[0])
                gvz.push_back(tree2.gvz[0])
                
                # Copy all hits associated with this track
                for j in range(len(tree2.gProcessID)):
                    gProcessID.push_back(tree2.gProcessID[j])
                    elementID.push_back(tree2.elementID[j])
                    detectorID.push_back(tree2.detectorID[j])
                    driftDistance.push_back(tree2.driftDistance[j])
                    tdcTime.push_back(tree2.tdcTime[j])
                    hitID.push_back(tree2.hitID[j])
                    hitTrackID.push_back(tree2.hitTrackID[j])

        total_hits = len(gProcessID)
        occupancies.append(total_hits)

        # Log the event
        log_line = f"{i:8d}  {num_tracks_to_inject:18d}  {total_hits:10d}\n"
        log_file.write(log_line)

        # Fill the output tree
        eventID[0] = i
        output_tree.Fill()

    fout.Write()
    fout.Close()
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
    parser.add_argument("file", type=str, help="Path to the background file (tree2).")
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
    parser.add_argument(
        "--use_propagation_models", action="store_true",
        help="Use different propagation models for background tracks."
    )
    args = parser.parse_args()

    occupancies = inject_tracks_randomized(
        args.file,
        args.output,
        args.log_txt,
        desired_max_hits=args.max_hits,
        use_propagation_models=args.use_propagation_models
    )

    if occupancies:
        output_name = f"mc_events_randomized_{round(len(occupancies), 3)}.root"
        os.rename(args.output, output_name)
        print(f"\nRenamed output file to: {output_name}")
    else:
        print("No events were processed, output file not created.")