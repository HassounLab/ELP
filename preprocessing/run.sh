#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=elppre
#
## output files
#SBATCH --output=output-%j.log
#SBATCH --error=output-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=1-00:00:00
#SBATCH --mem=10gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
source activate lipinggpu
export PYTHONPATH="/cluster/tufts/liulab/lib/anaconda3/envs/lipinggpu/lib/python3.7/site-packages/:$PYTHONPATH"

stdbuf -o0 python3 -u consolidate_graph.py

