#!/usr/bin/env python3
import sys
import math
import random
import ROOT

# ============================================================
# Hardcoded configuration (edit these as needed)
# ============================================================
TREE_NAME = "tree"

MMIN  = 0.3
MMAX  = 10.0
NBINS = 200

# Total number of dimuon *events* you want in the output file
N_DIMUON_OUT = 500000

# Random seed (controls which events get picked)
SEED = 42

# If g* branches are per-hit (many entries per same track),
# keep only the entry with largest |p| per gTrackID.
USE_GTRACKID_DEDUP = True

# Optional: write a quick validation histogram into the output file
WRITE_VALIDATION_HIST = True
# ============================================================

MUON_MASS_GEV = 0.1056583745


def enable_branches_for_mass(tree, use_trackid):
    """Enable only the branches needed to compute M_mumu (for speed in pass1/2)."""
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("gCharge", 1)
    tree.SetBranchStatus("gpx", 1)
    tree.SetBranchStatus("gpy", 1)
    tree.SetBranchStatus("gpz", 1)
    if use_trackid and hasattr(tree, "gTrackID"):
        tree.SetBranchStatus("gTrackID", 1)


def get_mu_pair(tree, use_trackid=True):
    """
    Returns (pxp,pyp,pzp, pxm,pym,pzm) for (mu+, mu-) in this event, or None.

    - Uses sign of gCharge to identify mu+/mu-.
    - If use_trackid and gTrackID exists: de-duplicate by gTrackID,
      keeping the entry with the largest |p| for each trackID.
    - If multiple mu+ or mu- tracks exist, pick the one with largest |p|.
    """
    if not (hasattr(tree, "gCharge") and hasattr(tree, "gpx") and hasattr(tree, "gpy") and hasattr(tree, "gpz")):
        return None

    charges = tree.gCharge
    pxs = tree.gpx
    pys = tree.gpy
    pzs = tree.gpz

    n = len(charges)
    if n < 2 or len(pxs) != n or len(pys) != n or len(pzs) != n:
        return None

    candidates = []

    if use_trackid and hasattr(tree, "gTrackID"):
        tids = tree.gTrackID
        if len(tids) != n:
            return None

        best_by_tid = {}  # tid -> (p2, q, px, py, pz)
        for i in range(n):
            q = int(charges[i])
            if q == 0:
                continue
            px = float(pxs[i]); py = float(pys[i]); pz = float(pzs[i])
            p2 = px*px + py*py + pz*pz

            tid = int(tids[i])
            prev = best_by_tid.get(tid)
            if prev is None or p2 > prev[0]:
                best_by_tid[tid] = (p2, q, px, py, pz)

        candidates = list(best_by_tid.values())
    else:
        for i in range(n):
            q = int(charges[i])
            if q == 0:
                continue
            px = float(pxs[i]); py = float(pys[i]); pz = float(pzs[i])
            p2 = px*px + py*py + pz*pz
            candidates.append((p2, q, px, py, pz))

    best_plus = None   # (p2, px, py, pz)
    best_minus = None

    for (p2, q, px, py, pz) in candidates:
        if q > 0:
            if best_plus is None or p2 > best_plus[0]:
                best_plus = (p2, px, py, pz)
        elif q < 0:
            if best_minus is None or p2 > best_minus[0]:
                best_minus = (p2, px, py, pz)

    if best_plus is None or best_minus is None:
        return None

    _, pxp, pyp, pzp = best_plus
    _, pxm, pym, pzm = best_minus
    return pxp, pyp, pzp, pxm, pym, pzm


def compute_mumu_mass(tree, use_trackid=True):
    """Fast invariant mass computation (no TLorentzVector)."""
    pair = get_mu_pair(tree, use_trackid=use_trackid)
    if pair is None:
        return None

    pxp, pyp, pzp, pxm, pym, pzm = pair

    p2p = pxp*pxp + pyp*pyp + pzp*pzp
    p2m = pxm*pxm + pym*pym + pzm*pzm

    e1 = math.sqrt(p2p + MUON_MASS_GEV*MUON_MASS_GEV)
    e2 = math.sqrt(p2m + MUON_MASS_GEV*MUON_MASS_GEV)

    px = pxp + pxm
    py = pyp + pym
    pz = pzp + pzm
    e  = e1 + e2

    m2 = e*e - (px*px + py*py + pz*pz)
    if m2 < 0:
        # Guard against tiny negative from floating point
        m2 = 0.0
    return math.sqrt(m2)


def mass_to_bin(m):
    """Return ROOT-like bin index in [1..NBINS], or None if out of range."""
    if m is None:
        return None
    # Match standard TH1 behavior: include [MMIN, MMAX) (x==MMAX treated as overflow)
    if not (MMIN <= m < MMAX):
        return None
    binw = (MMAX - MMIN) / NBINS
    b = int((m - MMIN) / binw) + 1
    if 1 <= b <= NBINS:
        return b
    return None


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.root output.root")
        sys.exit(1)

    inpath = sys.argv[1]
    outpath = sys.argv[2]

    ROOT.gROOT.SetBatch(True)

    fin = ROOT.TFile.Open(inpath, "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Could not open input file: {inpath}")

    tin = fin.Get(TREE_NAME)
    if not tin:
        raise RuntimeError(f"Could not find TTree '{TREE_NAME}' in {inpath}")

    # Decide whether we can actually use gTrackID
    use_trackid = USE_GTRACKID_DEDUP and hasattr(tin, "gTrackID")

    # Speed up pass1/pass2: only enable needed branches
    enable_branches_for_mass(tin, use_trackid)

    nentries = tin.GetEntries()

    # -----------------------
    # PASS 1: bin counts
    # -----------------------
    counts = [0] * (NBINS + 1)  # 1..NBINS used

    n_fail_mass = 0
    n_in_range = 0

    for i in range(nentries):
        tin.GetEntry(i)
        m = compute_mumu_mass(tin, use_trackid=use_trackid)
        b = mass_to_bin(m)
        if b is None:
            if m is None:
                n_fail_mass += 1
            continue
        counts[b] += 1
        n_in_range += 1

    nonempty_bins = [b for b in range(1, NBINS + 1) if counts[b] > 0]
    if not nonempty_bins:
        raise RuntimeError(f"No events found in mass range [{MMIN}, {MMAX})")

    min_count = min(counts[b] for b in nonempty_bins)
    nb = len(nonempty_bins)

    # Target allocation per bin: as uniform as possible, with integer counts
    target = int(N_DIMUON_OUT)

    # If user requests more than available, cap to available
    if target > n_in_range:
        print(f"WARNING: Requested N_DIMUON_OUT={target}, but only {n_in_range} events exist in range.")
        target = n_in_range

    base = target // nb  # baseline per non-empty bin
    base = min(base, min_count)  # cannot exceed least-populated bin

    desired = {b: base for b in nonempty_bins}
    total = base * nb
    remaining = target - total

    # Distribute remaining (+1 at a time) over bins that still have capacity
    rng = random.Random(SEED)
    bins_with_capacity = [b for b in nonempty_bins if counts[b] > desired[b]]

    while remaining > 0 and bins_with_capacity:
        rng.shuffle(bins_with_capacity)
        progressed = False
        for b in bins_with_capacity:
            if remaining == 0:
                break
            if desired[b] < counts[b]:
                desired[b] += 1
                remaining -= 1
                progressed = True
        bins_with_capacity = [b for b in bins_with_capacity if desired[b] < counts[b]]
        if not progressed:
            break

    if remaining > 0:
        # Could not reach target due to limited stats in some bins
        print(f"WARNING: Could not reach requested target. Short by {remaining} events.")
        print("         (This happens when many bins are near their minimum population.)")

    planned_out = sum(desired[b] for b in nonempty_bins)

    print("=== CONFIG ===")
    print(f"Tree:            {TREE_NAME}")
    print(f"Mass range:      [{MMIN}, {MMAX}) GeV")
    print(f"Flatten bins:    {NBINS}")
    print(f"Requested out:   {N_DIMUON_OUT}")
    print(f"Planned out:     {planned_out}")
    print(f"Use gTrackID:    {use_trackid}")
    print(f"Seed:            {SEED}")
    print("")
    print("=== PASS1 SUMMARY ===")
    print(f"Input entries:           {nentries}")
    print(f"Events with mass in range: {n_in_range}")
    print(f"Events failed (no mu+/mu-): {n_fail_mass}")
    print(f"Non-empty bins:          {nb} / {NBINS}")
    print(f"Min bin population:      {min_count}")
    print("")

    # -----------------------
    # PASS 2: reservoir sample per bin
    # -----------------------
    # Reservoirs store event indices for each bin
    reservoirs = {b: [] for b in nonempty_bins if desired[b] > 0}
    seen = {b: 0 for b in reservoirs.keys()}

    for i in range(nentries):
        tin.GetEntry(i)
        m = compute_mumu_mass(tin, use_trackid=use_trackid)
        b = mass_to_bin(m)
        if b is None:
            continue
        if b not in reservoirs:
            continue

        k = desired[b]
        if k <= 0:
            continue

        seen[b] += 1
        res = reservoirs[b]

        if len(res) < k:
            res.append(i)
        else:
            # Replace with decreasing probability to keep a uniform random sample from the bin
            j = rng.randrange(seen[b])
            if j < k:
                res[j] = i

    accepted = []
    for b, res in reservoirs.items():
        accepted.extend(res)
    accepted.sort()

    actual_out = len(accepted)
    print("=== PASS2 SUMMARY ===")
    print(f"Accepted events: {actual_out}")
    if actual_out != planned_out:
        print(f"NOTE: accepted != planned_out ({planned_out}). This usually means some bins had fewer events than expected.")

    # -----------------------
    # WRITE OUTPUT (full structure)
    # -----------------------
    # Re-enable all branches BEFORE cloning, so we preserve full structure
    tin.SetBranchStatus("*", 1)

    fout = ROOT.TFile.Open(outpath, "RECREATE")
    if not fout or fout.IsZombie():
        raise RuntimeError(f"Could not create output file: {outpath}")

    fout.cd()
    tout = tin.CloneTree(0)  # same branches, no entries yet

    # Optional validation histogram
    h_out = None
    if WRITE_VALIDATION_HIST:
        h_out = ROOT.TH1D("h_mumu_skim",
                          f"Dimuon mass skim;M_{{#mu#mu}} [GeV];Events",
                          NBINS, MMIN, MMAX)

    # For fast mass in validation fill, we can temporarily re-disable and re-enable,
    # but keep simple: compute mass as we fill.
    # (This loop is only ~1000 events typically.)
    for idx in accepted:
        tin.GetEntry(idx)
        tout.Fill()
        if h_out:
            m = compute_mumu_mass(tin, use_trackid=use_trackid)
            if m is not None and (MMIN <= m < MMAX):
                h_out.Fill(m)

    tout.Write(TREE_NAME)
    if h_out:
        h_out.Write()

    fout.Close()
    fin.Close()

    print("=== DONE ===")
    print(f"Wrote: {outpath}")
    print(f"Output dimuon events: {actual_out}")


if __name__ == "__main__":
    main()
