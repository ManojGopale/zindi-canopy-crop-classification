#!/bin/bash

### Set the job name 
#SBATCH --job-name hyperparam_search_03262025

## Request email when job begins and ends
#SBATCH --mail-user=manojgopale@email.arizona.edu
#SBATCH --mail-type=ALL

###### Error and Output Path
#SBATCH -e ../logs/hyperparam_search_03262025_error.txt
#SBATCH -o ../logs/hyperparam_search_03262025_output.txt

## Specify the PI group found with va command
#SBATCH --account=rlysecky

### Set the queue to submit this job
#### We can use windfall or standard, use standard if job cannot be submitted
#SBATCH --partition=standard

### Set the number of cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=94
###Ocelote
########SBATCH --ntasks=28
########SBATCH --ntasks=14

#########SBATCH --gres=gpu:1

##When -mem is not specidies 5Gb/task is used
####SBATCH --mem=15gb

######SBATCH --ntasks=25
#######SBATCH --mem=100gb

######SBATCH --constraint=hi_mem

#######SBATCH --exclusive

### Specify upto a maximum of 240 hours walltime for job
#SBATCH --time 20:0:0

####Stop core dumps
ulimit -c 0

cd /xdisk/bethard/manojgopale/extra/zindi-canopy-crop-classification/
module load python/3.9
source .venv/bin/activate
cd scr

date
/usr/bin/time python hyperparameter_search.py 
date
