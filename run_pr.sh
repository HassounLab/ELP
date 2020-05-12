#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=elp
#
## output files
#SBATCH --output=logs/output-ep-pr-%j.log
#SBATCH --error=logs/output-ep-pr-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-24:00:00
#SBATCH --mem=75gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p gpu 
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --exclude=pgpu01
nvidia-smi
source activate liping
stdbuf -o0 python -u run-exp.py kegg_20 -m ep -e pr
