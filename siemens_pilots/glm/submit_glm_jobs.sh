#!/bin/bash

# Check if subject argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <subject>"
    exit 1
fi

SUBJECT=$1

# Define multiband factors and BIDS folders
MULTIBAND_FACTORS=(-1 0 2 4)
#BIDS_FOLDERS=(
    #"/shares/zne.uzh/gdehol/ds-siemenspilotsfmap"
    #"/shares/zne.uzh/gdehol/ds-siemenspilots24"
#)

BIDS_FOLDERS=(
    "/shares/zne.uzh/gdehol/ds-siemenspilots"
)

# Submit jobs
for MB in "${MULTIBAND_FACTORS[@]}"; do
    for BIDS_FOLDER in "${BIDS_FOLDERS[@]}"; do
        sbatch fit_glm.sh "$SUBJECT" "$MB" "$BIDS_FOLDER" "on"
        sbatch fit_glm.sh "$SUBJECT" "$MB" "$BIDS_FOLDER" "off"
    done
done
