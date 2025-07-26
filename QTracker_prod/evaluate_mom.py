import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def plot_residuals(residuals: np.ndarray, res_type: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=300, alpha=0.7, color='b', edgecolor='black')
    plt.title(f'Histogram of Momentum Residuals for {res_type}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    mom_plots_dir = os.path.join(os.path.dirname(__file__), 'plots', 'momentum')
    os.makedirs(mom_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(mom_plots_dir, f'mom_res_{res_type}'))
    plt.show()


def record_res_stats(residuals: np.ndarray, res_type: str, mom_res_stats: str) -> None:
    with open(mom_res_stats, 'a') as file:
        file.write(f'--- Summary of {res_type} ---\n')
        file.write(f'Mean: {np.mean(residuals):.3f}\n')
        file.write(f'Standard Deviation: {np.std(residuals):.3f}\n\n')


def momentum_residuals(root_file: str, mom_res_stats: str) -> None:
    f = ROOT.TFile.Open(root_file, 'READ')
    tree = f.Get('tree')

    gpx_muPlus, gpy_muPlus, gpz_muPlus = [], [], []
    gpx_muMinus, gpy_muMinus, gpz_muMinus = [], [], []

    qpx_muPlus, qpy_muPlus, qpz_muPlus = [], [], []
    qpx_muMinus, qpy_muMinus, qpz_muMinus = [], [], []

    if not tree:
        print("Error: Tree not found in file.")
        return 

    for event in tree:
        # Ground-truth
        gpx_muPlus.append(event.gpx[0])
        gpy_muPlus.append(event.gpy[0])
        gpz_muPlus.append(event.gpz[0])

        gpx_muMinus.append(event.gpx[1])
        gpy_muMinus.append(event.gpy[1])
        gpz_muMinus.append(event.gpz[1])

        # Prediction
        qpx_muPlus.append(event.qpx[0])
        qpy_muPlus.append(event.qpy[0])
        qpz_muPlus.append(event.qpz[0])

        qpx_muMinus.append(event.qpx[1])
        qpy_muMinus.append(event.qpy[1])
        qpz_muMinus.append(event.qpz[1])
    
    gpx_muPlus, gpy_muPlus, gpz_muPlus = np.array(gpx_muPlus), np.array(gpy_muPlus), np.array(gpz_muPlus)
    qpx_muPlus, qpy_muPlus, qpz_muPlus = np.array(qpx_muPlus), np.array(qpy_muPlus), np.array(qpz_muPlus)

    gpx_muMinus, gpy_muMinus, gpz_muMinus = np.array(gpx_muMinus), np.array(gpy_muMinus), np.array(gpz_muMinus)
    qpx_muMinus, qpy_muMinus, qpz_muMinus = np.array(qpx_muMinus), np.array(qpy_muMinus), np.array(qpz_muMinus)

    res_px_muPlus = gpx_muPlus - qpx_muPlus
    res_py_muPlus = gpy_muPlus - qpy_muPlus
    res_pz_muPlus = gpz_muPlus - qpz_muPlus

    res_px_muMinus = gpx_muMinus - qpx_muMinus
    res_py_muMinus = gpy_muMinus - qpy_muMinus
    res_pz_muMinus = gpz_muMinus - qpz_muMinus

    # Plot and record residuals per momentum type
    with open(mom_res_stats, 'w'):      # clear file contents
        pass

    plot_residuals(res_px_muPlus, res_type="px_muPlus")
    record_res_stats(res_px_muPlus, res_type="px_muPlus", mom_res_stats=mom_res_stats)
    
    plot_residuals(res_py_muPlus, res_type="py_muPlus")
    record_res_stats(res_py_muPlus, res_type="py_muPlus", mom_res_stats=mom_res_stats)

    plot_residuals(res_pz_muPlus, res_type="pz_muPlus")
    record_res_stats(res_pz_muPlus, res_type="pz_muPlus", mom_res_stats=mom_res_stats)

    plot_residuals(res_px_muMinus, res_type="px_muMinus")
    record_res_stats(res_px_muMinus, res_type="px_muMinus", mom_res_stats=mom_res_stats)

    plot_residuals(res_py_muMinus, res_type="py_muMinus")
    record_res_stats(res_py_muMinus, res_type="py_muMinus", mom_res_stats=mom_res_stats)

    plot_residuals(res_pz_muMinus, res_type="pz_muMinus")
    record_res_stats(res_pz_muMinus, res_type="pz_muMinus", mom_res_stats=mom_res_stats)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Evaluate momentum models using residuals')
    parser.add_argument('input_file', type=str, help='Input ROOT file')
    parser.add_argument('--output', type=str, default='results/mom_res_stats.txt', help='Output file name for storing residual stats')
    args = parser.parse_args()

    momentum_residuals(args.input_file, args.output)
