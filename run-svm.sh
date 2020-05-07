#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=svm
#
## output files
#SBATCH --output=logs/output-svm-%j.log
#SBATCH --error=logs/output-svm-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-6:00:00
#SBATCH --mem=300gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p largemem
#SBATCH -n 8
stdbuf -o0 python -u run-exp.py kegg_20 -m svm
