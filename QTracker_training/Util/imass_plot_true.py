import argparse
import ROOT

# Muon mass in GeV (PDG)
MUON_MASS_GEV = 0.1056583745


def pick_muon_indices(charges):
    """
    Given a vector of charges, return (idx_mu_plus, idx_mu_minus)
    using the sign of gCharge.

    This is robust to gCharge being +/-1 or +/-13, etc.
    """
    idx_plus = None
    idx_minus = None

    for i, q in enumerate(charges):
        if q > 0 and idx_plus is None:
            idx_plus = i
        elif q < 0 and idx_minus is None:
            idx_minus = i

        if idx_plus is not None and idx_minus is not None:
            break

    return idx_plus, idx_minus


def main():
    parser = argparse.ArgumentParser(
        description="Plot dimuon invariant mass from gpx/gpy/gpz using gCharge to identify mu+/mu-."
    )
    parser.add_argument("input", help="Input ROOT file")
    parser.add_argument("--tree", default="tree", help="TTree name (default: tree)")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins (default: 200)")
    parser.add_argument("--mmin", type=float, default=0.0, help="Min mass [GeV] (default: 0)")
    parser.add_argument("--mmax", type=float, default=10.0, help="Max mass [GeV] (default: 10)")
    parser.add_argument("--out", default="mumu_mass.png", help="Output image (png/pdf/etc)")
    parser.add_argument("--outroot", default="", help="Optional output ROOT file to save hist/canvas")
    parser.add_argument("--logy", action="store_true", help="Use log scale on y-axis")
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)

    f = ROOT.TFile.Open(args.input, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open input file: {args.input}")

    t = f.Get(args.tree)
    if not t:
        raise RuntimeError(f"Could not find TTree '{args.tree}' in file {args.input}")

    # Histogram
    h = ROOT.TH1D(
        "h_mumu",
        "Dimuon invariant mass;M_{#mu#mu} [GeV];Events",
        args.bins,
        args.mmin,
        args.mmax,
    )
    h.Sumw2()

    nentries = t.GetEntries()
    n_filled = 0
    n_skipped = 0

    for ievt in range(nentries):
        t.GetEntry(ievt)

        charges = t.gCharge
        pxs = t.gpx
        pys = t.gpy
        pzs = t.gpz

        # Basic consistency checks
        ntrk = len(charges)
        if ntrk < 2 or len(pxs) != ntrk or len(pys) != ntrk or len(pzs) != ntrk:
            n_skipped += 1
            continue

        idx_plus, idx_minus = pick_muon_indices(charges)
        if idx_plus is None or idx_minus is None:
            n_skipped += 1
            continue

        p4p = ROOT.TLorentzVector()
        p4m = ROOT.TLorentzVector()

        # SetXYZM computes E from p and mass: E = sqrt(p^2 + m^2)
        p4p.SetXYZM(pxs[idx_plus], pys[idx_plus], pzs[idx_plus], MUON_MASS_GEV)
        p4m.SetXYZM(pxs[idx_minus], pys[idx_minus], pzs[idx_minus], MUON_MASS_GEV)

        mumu = p4p + p4m
        mass = mumu.M()

        h.Fill(mass)
        n_filled += 1

    # Draw
    c = ROOT.TCanvas("c_mumu", "c_mumu", 900, 700)
    if args.logy:
        c.SetLogy()

    h.SetLineWidth(2)
    h.Draw("HIST")
    c.SaveAs(args.out)

    # Optional ROOT output
    if args.outroot:
        fout = ROOT.TFile.Open(args.outroot, "RECREATE")
        h.Write()
        c.Write()
        fout.Close()

    print("Done.")
    print(f"  Entries in tree: {nentries}")
    print(f"  Events filled:   {n_filled}")
    print(f"  Events skipped:  {n_skipped}")
    print(f"  Saved plot:      {args.out}")
    if args.outroot:
        print(f"  Saved ROOT:      {args.outroot}")


if __name__ == "__main__":
    main()
