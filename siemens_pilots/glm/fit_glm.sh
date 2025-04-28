#!/bin/bash
#SBATCH --job-name=fit_glm_siemens_pilots
#SBATCH --output=/home/gdehol/logs/%x_%j.out
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

# Load environment
. $HOME/init_conda.sh

# Define variables from SLURM arguments
SUBJECT=$1
MULTIBAND=$2
BIDS_FOLDER=$3
SMOOTHED=$4

# Validate BIDS_FOLDER path
if [[ ! "$BIDS_FOLDER" =~ ^/shares/zne.uzh/gdehol/ ]]; then
    echo "Error: BIDS folder must start with '/shares/zne.uzh/gdehol/'"
    exit 1
fi

# Extract dataset name from BIDS_FOLDER path
BIDS_NAME=$(basename "$BIDS_FOLDER")

# Determine if smoothed should be set
SMOOTHED_FLAG=""
SMOOTHED_LABEL="nosmooth"
if [[ "$SMOOTHED" == "on" ]]; then
    SMOOTHED_FLAG="--smoothed"
    SMOOTHED_LABEL="smooth"
fi

# Create log filename
LOGFILE="${HOME}/logs/fit_glm_${SUBJECT}_MB${MULTIBAND}_${BIDS_NAME}_${SMOOTHED_LABEL}.log"

# Ensure logs directory exists
mkdir -p "$HOME/logs"

# Run Python script and redirect output to log file
python "$HOME/git/siemens_pilots/siemens_pilots/glm/fit_glm.py" "$SUBJECT" "$MULTIBAND" --bids_folder "$BIDS_FOLDER" $SMOOTHED_FLAG &> "$LOGFILE"
