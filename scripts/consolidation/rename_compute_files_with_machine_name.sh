#!/bin/bash

# Define the directory containing the CSV files
DIRECTORY="/Volumes/Smartflat/data/dataframes/frozen-metrics-logs"

# Loop through all CSV files in the directory
for file in "$DIRECTORY"/*.csv; do
    # Get the base filename (without the extension)
    base_name=$(basename "$file" .csv)
    # Append "_text" and rename the file
    mv "$file" "$DIRECTORY/${base_name}_#2.csv"
done
