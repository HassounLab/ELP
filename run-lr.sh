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
#SBATCH --time=1-12:00:00
#SBATCH --mem=1000gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p largemem
#SBATCH -n 8
#SBATCH -N 8
source activate lipinggpu
export PYTHONPATH="/cluster/tufts/liulab/lib/anaconda3/envs/lipinggpu/lib/python3.7/site-packages/:$PYTHONPATH"
stdbuf -o0 python -u run-exp.py kegg_20_maccs -m logreg --load_folds --start_from 0
