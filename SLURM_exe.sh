#!/bin/bash

#SBATCH --job-name=image-net
#SBATCH --account=cse585f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40g
#SBATCH --mail-user=bchao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=fine-tune.out

python image_net_eval.py