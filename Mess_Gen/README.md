
# Generating Messy Monte Carlo Events
Events are a recorded set of hits that we save with a high probability of having the physics process we are interested (Dimuon) embedded in
some amount of background hits.  The background hits are primarily coming from the dump region and not the target.  The event generator here
take Geant generated events for signal and adds Geant generated single muon tracks (both mu+ and mu-) hits from the dump to make events of whatever
level of background one is needing.  The script provided (messy_gen.py) assumes that you have already generated your signal and single muon tracks in the RUS formate and that you want to combined them to generate messy simulated events. See about RUS files here:  [UVA RUS files](https://github.com/uva-spin/UVA_RUS_Basic).



The messy_gen.py script injects a specified number of Monte Carlo (MC) tracks from one RUS ROOT file into another, simulating detector response through probabilistic hit retention as well as simulating fall off in hit frequency based on detector depth. It provides support for multiple propagation models and compresses the output efficiently using LZMA and writes back to the RUS file structure.

## Features

- **Track Injection**: Adds synthetic tracks into existing event data.
- **Multiple Propagation Models**: Choose between linear, Gaussian, or exponential models to simulate hit retention.
- **Probabilistic Hit Retention**: Retains hits based on a configurable mean probability and width.
- **ROOT TTree Support**: Reads from and writes to ROOT TTrees with full vector branch support.
- **High Compression Output**: Output file is compressed using LZMA with maximum compression level.

---

## How It Works

1. The script reads events from two ROOT files:
   - `file1`: The primary event file into which tracks will be injected.
   - `file2`: The MC track source file that provides additional tracks for injection.

2. Each event from `file1` is copied to the output.
3. For each event, `NUM_TRACKS` new tracks from `file2` are appended, with associated hits.
4. Hits are kept probabilistically:
   - A random per-track probability is drawn from a Gaussian centered on `PROB_MEAN` with width `PROB_WIDTH`.
   - A weighting is applied per hit using the selected propagation model.

---

## Configuration

All main configuration settings are hard-coded near the top of the script:

```python
NUM_TRACKS = 50            # Number of tracks to inject (1-100)
PROB_MEAN = 0.9            # Mean probability to retain hits
PROB_WIDTH = 0.1           # Width of the probability distribution

PROPAGATION_MODEL = "gaussian"  # Options: "linear", "gaussian", "exponential"
GAUSSIAN_SIGMA = 10.0           # For Gaussian model
EXP_DECAY_CONST = 15.0          # For Exponential model
```

You can easily modify these to suit your desired injection scenario.

---

## Input File Structure

The script expects ROOT files with a TTree named `"tree"` and the following branches:

- Scalar:
  - `eventID` (int)

- Vector branches:
  - `elementID`, `detectorID`, `hitID`, `processID`, `trackID` (int)
  - `tdcTime`, `driftDistance` (double)
  - `gCharge` (int)
  - `gvx`, `gvy`, `gvz`, `gpx`, `gpy`, `gpz` (double)

---

## Usage

```bash
python inject_tracks.py input1.root input2.root --output output.root
```

- `input1.root`: Input file with existing events.
- `input2.root`: Input file with MC tracks to inject.
- `--output`: (Optional) Output file name. Default is `mc_events.root`.

---

## Propagation Models

The per-hit retention probability is modified by one of the following models:

### Linear:
\[
w = 1 - \frac{detectorID}{100}
\]

### Gaussian:
\[
w = \exp\left(-\frac{(detectorID - 1)^2}{2 \cdot \text{GAUSSIAN\_SIGMA}^2}\right)
\]

### Exponential:
\[
w = \exp\left(-\frac{detectorID}{\text{EXP\_DECAY\_CONST}}\right)
\]

These models simulate detector efficiency decreasing with depth.

---

## 🧪 Output

The script writes a new ROOT file with the updated tree containing:
- All original hits from `file1`
- Injected tracks from `file2`
- A combined and incremented `hitID` and `trackID` to ensure uniqueness
- Maximum compression and efficient flushing using `SetAutoFlush(0)`

---

## Requirements

- Python ≥ 3.6
- [ROOT](https://root.cern/)
- NumPy

Ensure that your Python environment has access to ROOT (e.g., via PyROOT) and can import `ROOT`.

---

## Notes

- Injection stops if the number of available events in `file2` is exhausted.
- Output file uses LZMA compression (`ROOT.kLZMA`) with compression level 9.
- This script is designed for performance and compact storage.

---

