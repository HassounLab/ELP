#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=js
#
## output files
#SBATCH --output=logs/output-js-%j.log
#SBATCH --error=logs/output-js-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-8:00:00
#SBATCH --mem=50gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -n 8
source activate lipinggpu

export PYTHONPATH="/cluster/tufts/liulab/lib/anaconda3/envs/lipinggpu/lib/python3.7/site-packages/:$PYTHONPATH"
stdbuf -o0 python -u run-exp.py kegg_20_maccs -m js -e lp
#stdbuf -o0 python -u run-exp.py kegg_20_maccs -m js -e rr 
