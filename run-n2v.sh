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
#SBATCH -n 8
source activate lipinggpu
stdbuf -o0 python -u run-exp.py kegg_20_node_only -m n2v -e lp
