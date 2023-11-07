#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=64G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="thresh_10x_sparse" # Name the job
#SBATCH --output=clusteroutput%j.out
##SBATCH --exclude=iris[1-3],iris-hp-z8
#SBATCH --mail-user=oliviayl@stanford.edu
#SBATCH --mail-type=ALL

export MUJOCO_GL="egl"
cd /iris/u/oliviayl/repos/affordance-learning/vip/evaluation/trajopt/trajopt
source activate vip
python run_mppi.py env=kitchen_sdoor_open-v3 embedding=vip paths_per_cpu=256