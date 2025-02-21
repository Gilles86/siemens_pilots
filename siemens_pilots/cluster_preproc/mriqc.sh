#!/bin/bash
#SBATCH --job-name=mriqc_siemens
#SBATCH --output=/home/gdehol/logs/mriqc_%j_%x.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=45:00

# Load required modules
source /etc/profile.d/lmod.sh
module load singularityce/4.2.1  # Ensure latest Singularity version

# Ensure a participant label is provided
if [ -z "$1" ]; then
    echo "Error: No participant label provided."
    echo "Usage: sbatch mriqc_siemens.sh <participant_label>"
    exit 1
fi

export PARTICIPANT_LABEL=$1  # Use command-line argument
echo "Running MRIQC for participant: $PARTICIPANT_LABEL"

# Set FreeSurfer license
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt

# Run MRIQC with Singularity
singularity run --cleanenv \
    -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
    -B /shares/zne.uzh/gdehol/ds-siemenspilots:/data \
    -B /scratch/gdehol:/workflow \
    /shares/zne.uzh/containers/mriqc-24.0.0 \
    /data /data/derivatives/mriqc participant \
    --participant_label $PARTICIPANT_LABEL \
    -w /workflow

echo "MRIQC processing for participant $PARTICIPANT_LABEL completed."
