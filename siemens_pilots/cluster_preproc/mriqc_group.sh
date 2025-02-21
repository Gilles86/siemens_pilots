#!/bin/bash
#SBATCH --job-name=mriqc_siemens
#SBATCH --output=/home/gdehol/logs/mriqc_%j_%x.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00

# Load required modules
source /etc/profile.d/lmod.sh
module load singularityce/4.2.1  # Ensure latest Singularity version

export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt

# Run MRIQC with Singularity
singularity run --cleanenv \
    -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
    -B /shares/zne.uzh/gdehol/ds-siemenspilots:/data \
    -B /scratch/gdehol:/workflow \
    /shares/zne.uzh/containers/mriqc-24.0.0 \
    /data /data/derivatives/mriqc group \
    -w /workflow