#!/usr/bin/env bash
# deploy.sh

set -euo pipefail

# Build the Apptainer image from the definition
echo "Building TfRootBuild.sif from TfRootBuild.def…"
apptainer build TfRootBuild.sif TfRootBuild.def

# Deploy to Rivanna (requires VPN or on-campus network)
echo "Copying TfRootBuild.sif to Rivanna project directory…"
scp -C TfRootBuild.sif bab6cw@login.hpc.virginia.edu:/project/ptgroup/spinquest/David/

echo "Build and deploy complete."
