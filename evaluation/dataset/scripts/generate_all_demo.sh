#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=64G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="vip_data" # Name the job
#SBATCH --output=clusteroutput%j.out
#SBATCH --exclude=iris[1-5],iris-hp-z8
##SBATCH --mail-user=oliviayl@stanford.edu
##SBATCH --mail-type=ALL

cd /iris/u/oliviayl/repos/affordance-learning/vip/evaluation
source activate vip
python trajopt/trajopt/generate_demo.py env=kitchen_micro_open-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_micro_close-v3;
# python trajopt/trajopt/generate_demo.py env=kitchen_sdoor_open-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_sdoor_close-v3; 
python trajopt/trajopt/generate_demo.py env=kitchen_rdoor_open-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_rdoor_close-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_ldoor_open-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_ldoor_close-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_knob1_off-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_knob1_on-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_light_off-v3;
python trajopt/trajopt/generate_demo.py env=kitchen_light_on-v3;

