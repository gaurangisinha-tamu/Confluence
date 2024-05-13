#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=exp-1        #Set the job name to "JobExample1"
#SBATCH --time=12:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=5120M               #Request 2560MB (2.5GB) per node
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=exp-1.%j    #Send stdout/err to "Example1Out.[jobID]"

cd /scratch/user/arunim_samudra/ISR_Project
source env/bin/activate
python3 main.py --use_mf=False --use_clip=False