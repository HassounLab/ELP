#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=n2v
#
## output files
#SBATCH --output=logs/output-n2v-%j.log
#SBATCH --error=logs/output-n2v-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-7:00:00
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
source activate lipinggpu
stdbuf -o0 python -u run-exp.py kegg_20_n2vshort -m n2v -e lp --load_folds
