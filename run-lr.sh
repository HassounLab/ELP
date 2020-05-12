#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=lr
#
## output files
#SBATCH --output=logs/output-lr-%j.log
#SBATCH --error=logs/output-lr-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-24:00:00
#SBATCH --mem=350gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p largemem
#SBATCH -n 8
stdbuf -o0 python -u run-exp.py kegg_20 -m logreg --start_from 1 --load_folds
