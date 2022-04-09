#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=shas
#SBATCH --output=partition.out
#SBATCH --qos=normal

module load anaconda
conda activate /curc/sw/anaconda3/2019.03/envs/idp
cd src
python build_train_val_test.py dataset_test_1 0.3 0.2 1 2

