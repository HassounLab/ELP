#!/bin/bash
#SBATCH --output=output-%j.log
#SBATCH --error=output-%j.err
#SBATCH --job-name=kegg-em
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=300
#SBATCH --mem=100gb
#SBATCH --gres=gpu:1
nvidia-smi
#python3 run-exp.py kegg -m js
python3 -u run-exp.py kegg -m em
