# Dimuon Track and Background Generator Suite

This repository provides a full pipeline for preparing signal and background datasets for training in dimuon track reconstruction tasks for QTracker_prod.py. 
It is specifically designed for processing Geant4-based simulated events involving dimuon pairs from a target and background muons from a dump. The final 
output includes RUS ROOT files tailored for different training tasks, including track finding and momentum regression, and other future models as needed.

---

## File Overview

### 1. `separate.py`
**Purpose:**  
Splits a signal ROOT file (`Signal_rus.root`) containing dimuon events into two separate files, one for μ⁺ and one for μ⁻ tracks. This simplifies training data handling by isolating each particle's hits and track information.

**Usage:**
```bash
python separate.py Signal_rus.root
```

**Output Files:**
- `Signal_rus_track1.root` — contains μ⁺ (muon+) events
- `Signal_rus_track2.root` — contains μ⁻ (muon−) events

---

### 2. `combine.py`
**Purpose:**  
Merges two single-muon ROOT files (e.g., `mup_rus.root` and `mum_rus.root`) by alternating events from each. The result simulates a realistic background file (`single_muons.root`) that mimics random single muon tracks from the dump.

**Usage:**
```bash
python combine.py Signal_rus_track1.root Signal_rus_track2.root --output single_muons.root
```

**Default Output:**  
- `single_muons.root` — required as input for background injection (`messy_gen.py`)

---

### 3. `gen_training.py`
**Purpose:**  
Generates all necessary training datasets by combining μ⁺ and μ⁻ signal tracks and creating hit array representations of the detector. It outputs:
- A file for training track-finding models
- Separate files for momentum regression training for each muon charge

**Usage:**
```bash
python3 gen_training.py Signal_rus_track1.root Signal_rus_track2.root
```

**Output Files:**
- `finder_training.root` — merged file with hit matrices, labels for both μ⁺ and μ⁻
- `momentum_training-1.root` — μ⁺ events with hit array
- `momentum_training-2.root` — μ⁻ events with hit array

---

### 4. `messy_gen.py`
**Purpose:**  
Injects randomly generated background muon tracks from `single_muons.root` into the signal file `finder_training.root`. This simulates realistic contamination during data taking and makes the dataset suitable for robust DNN training.

**Usage:**
```bash
python messy_gen.py finder_training.root single_muons.root
```
**Optional Arguments:**
- `--output mc_events.root` — specify the output filename (default: `mc_events.root`)

**Output:**
- `mc_events.root` — signal + injected background events

---

### 5. `noisy_gen.py`
**Purpose:**  
Injects randomly generated noise hits into the existing ROOT file containing detector hit information. It injects two types of noise (electronic noise and cluster noise) into the hit vectors, and saves the modified events to the standard compressed RUS output file.

## Noise Model

- **Electronic Noise Probability:** 1% per unused (detectorID, elementID) pair.
- **Cluster Noise Probability:** 5% per detector.
- **Cluster Length Range:** Randomly between 2 and 4 adjacent elements.
- **Injected Noise Properties:** All injected hits have `driftDistance = 0.0` and `tdcTime = 0.0` to clearly mark them as noise.

These settings can be easily modified inside the script by changing the constants:
```python
P_ELECTRONIC_NOISE = 0.01
P_CLUSTER_NOISE = 0.05
CLUSTER_LENGTH_RANGE = (2, 4)
```

**Usage:**
```bash
python noisy_gen.py mc_events.root
```

**Output:**
- `noisy_output.root` — messy event + electronic and cluster noise

---

## Training Workflow Summary

1. **Split the signal:**  
   ```bash
   python3 separate.py Signal_rus.root
   ```
2. **Prepare background hits (from dump simulation):**  
   ```bash
   python3 combine.py Signal_rus_track1.root Signal_rus_track2.root
   ```
3. **Generate training files:**  
   ```bash
   python3 gen_training.py Signal_rus_track1.root Signal_rus_track2.root
   ```
4. **Add background to training hits:**  
   ```bash
   python3 messy_gen.py finder_training.root single_muons.root
   ```
5. **Add noise to training hits:**  
   ```bash
   python3 noisy_gen.py mc_events.root
   ```
---

## Configuration Options

Each script contains hardcoded parameters at the top of the file for:
- Maximum number of events
- Number of background tracks injected
- Probability distributions for track propagation (`linear`, `gaussian`, or `exponential`)
- Background track drop-off settings (sigma, decay constant)

Modify these values directly in the script before execution to control behavior and simulation style.

---

## Dependencies

- [ROOT](https://root.cern/)
- Python 3.7+
- NumPy

Ensure that PyROOT is correctly configured in your Python environment before running any scripts.

---

## Output Structure

All RUS ROOT files produced contain TTrees named `tree` with branches that include:
- Hit-level data (`elementID`, `detectorID`, `driftDistance`, `tdcTime`, etc.)
- Track-level data (`trackID`, `gCharge`, `gpx/gpy/gpz`, `gvx/gvy/gvz`)
- Muon charge ID (`muID`)
- Optional 2D and 1D hit matrix arrays (`HitArray`, `HitArray_mup`, `HitArray_mum`)

---


- All event indexing is internally consistent (`eventID` matches across files).
- Each hit array represents one muon's trajectory through a 62-layer detector with optional positional encoding.
- Track injection in `messy_gen.py` includes tunable physics-motivated propagation models, allowing for realistic simulations of track degradation and decay.

---

