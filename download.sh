#!/bin/bash

# This script zips the eval_logs, model, and runs folders into a single archive.
# The -r flag is used to recursively include all files and subdirectories.
zip -r fyp-code-backup.zip eval_logs model runs
