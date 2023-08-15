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
#!/bin/bash

# Lists of integers as space-separated strings
train_pers="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 26 27 29 31 33 35 37 39 42 44 47 50 53 56 60 64 68 72 76 81 86 92 97 103 110 117 124 132 140 149 158 168 179 190 202 214 228 242 257 273 290 308 327 348 370 393 417 443 471 500 532 565 600 638 677 720 765 812 863 917 974 1035 1100 1169 1242 1319 1402 1489 1582 1681 1786 1898 2016 2142 2276 2418 2570 2730 2901 3082 3274 3479 3696 3927 4173 4433 4710 5005 5317 5650 6002 6377 6776 7199 7649 8127 8635 9174 9747 10356"

train_pers2="11003 11691 12421 13197 14022 14898 15828 16817 17868 18984 20170 21431 22770 24192"
train_pers3="25704 27309 29016 30828 32754 34801 36975 39285 41740 44347 47118 50062"
train_pers4="53189 56513 60043 63794 67780 72015 76514 81294"
train_pers5="86373 91770 97503 103595 110067 116944 124250"
train_pers6="132012 140260 149023 158334 168226 178736 189903 201767 214373 227766 241996 257115 273179 290246 308379 327646 348116 369865 392973 417524 443610 471325 500772 532058 565299 600617 638142 678011 720370 765376 813194 864000"



JobID=$( expr $SLURM_ARRAY_TASK_ID-1 )
echo $JobID

Filename = "./data/2015-2017_100m.npy"
echo $Filename

# Run the Python script and pass the lists of integers as arguments
python3 AutoRegression2.py $p $train_pers