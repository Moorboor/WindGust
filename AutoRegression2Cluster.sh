#!/bin/bash
#SBATCH --job-name=Wind30m              # Job name
#SBATCH --mail-type=NONE              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mvando@uos.de    # Where to send mail
#SBATCH --ntasks=1                      # Run a single task		
#SBATCH --mem=8000                      # Memory per processor
#SBATCH --cpus-per-task=1              		  # number of CPUs
#SBATCH --time=999:60:60                # Time limit hrs:min:sec
#SBATCH --output=test.out        # Standard output and error log
#SBATCH --partition=All_Nodes
#SBATCH --exclude=statph[35,39]

#pwd; hostname; date

p = 5

JobID=$( expr $SLURM_ARRAY_TASK_ID-1 )
echo $JobID

Filename = "./data/2015-2017_100m.npy"
echo $Filename

python3 AutoRegression2.py $p