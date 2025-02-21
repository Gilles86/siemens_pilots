#!/bin/bash
#SBATCH --job-name=fmriprep_siemenspilots
#SBATCH --output=/home/gdehol/logs/siemens_fmriprep_alina.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
source /etc/profile.d/lmod.sh
module load singularityce/4.2.1
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
singularity run --no-mount hostfs \
    -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
    -B /shares/zne.uzh/gdehol/ds-siemenspilots:/data \
    -B /scratch/gdehol:/workflow --cleanenv \
    /shares/zne.uzh/containers/fmriprep-23.2.1 \
    /data /data/derivatives/fmriprep participant \
    --participant_label alina \
    --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative \
    --skip_bids_validation -w /workflow --no-submm-recon