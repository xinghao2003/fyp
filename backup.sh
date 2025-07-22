#!/bin/bash

# This script zips the eval_logs, model, and runs folders into a single archive.
# The -r flag is used to recursively include all files and subdirectories.
zip -r result.zip eval_logs model runs optuna_studies optuna_trials optuna_logs output.log
