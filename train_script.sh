#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --partition=shas
#SBATCH --output=train.out
#SBATCH --qos=normal

module load anaconda
conda activate /projects/jasi7701/.conda/envs/venv
cd mdl
python model_cnn_1.py dataset_test_1 10

