#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=fgpt_classifier
#
## output files
#SBATCH --output=output-%j.log
#SBATCH --error=output-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=1-0:00:00
#SBATCH --mem=10gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p gpu 
#SBATCH -N 2
#SBATCH -n 2

python -u fgpt_classifier.py ../logs/em/kegg_smiles_epoch50




