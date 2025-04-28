#!/bin/bash
#SBATCH --job-name=nprf_fit_joint_gaussian
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --time=25:00
#SBATCH --mem=32G  # Request more memory
#SBATCH --output=/home/gdehol/logs/nprf_fit_siemenspilots_gaussian_%j.txt  # Default SLURM log

# Load environment
. $HOME/init_conda.sh
module load gpu
source activate neural_priors2

# Get required arguments: participant label, model number, multiband factor, and BIDS folder
PARTICIPANT_LABEL=${1:?Error: No participant label provided}
MODEL=${2:?Error: No model number provided}
MULTIBAND=${3:?Error: No multiband factor provided}
BIDS_FOLDER=${4:?Error: No BIDS folder provided}
SMOOTHED_FLAG=""
SMOOTHED_SUFFIX="raw"
LOG_SPACE_FLAG=""
LOG_SPACE_SUFFIX="natural"

# Validate BIDS_FOLDER path
if [[ ! "$BIDS_FOLDER" =~ ^/shares/zne.uzh/gdehol/ ]]; then
    echo "Error: BIDS folder must start with '/shares/zne.uzh/gdehol/'"
    exit 1
fi

# Extract dataset name from BIDS folder
BIDS_NAME=$(basename "$BIDS_FOLDER")

# Check additional script arguments
for arg in "$@"; do
    case "$arg" in
        --smoothed)
            SMOOTHED_FLAG="--smoothed"
            SMOOTHED_SUFFIX="smoothed"
            ;;
        --log_space)
            LOG_SPACE_FLAG="--log_space"
            LOG_SPACE_SUFFIX="log"
            ;;
    esac
done

# Define dynamic log file
LOGFILE="/home/gdehol/logs/nprf_fit_joint_gaussian_${SLURM_JOB_ID}_${PARTICIPANT_LABEL}_model-${MODEL}_MB${MULTIBAND}_${BIDS_NAME}_${SMOOTHED_SUFFIX}_${LOG_SPACE_SUFFIX}.txt"

# Run the encoding model fit and redirect output manually
python $HOME/git/siemens_pilots/siemens_pilots/encoding_model/fit_encoding_model.py \
    "$PARTICIPANT_LABEL" "$MULTIBAND" --bids_folder "$BIDS_FOLDER" \
    $SMOOTHED_FLAG --model "$MODEL" $LOG_SPACE_FLAG > "$LOGFILE" 2>&1
