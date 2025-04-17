#!/bin/bash
#SBATCH --job-name=2.5Dmul
#SBATCH --nodes=4               # adjust as needed
#SBATCH --ntasks-per-node=16    # adjust to match cores/node
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=512
#SBATCH --export=ALL
module purge
module use /projects/comp422/modulefiles
module load assignment3

# Define test points: P values and chosen c so P/c is a perfect square
declare -A CVALS
CVALS[4]=1
CVALS[9]=1
CVALS[16]=4
CVALS[25]=5
CVALS[36]=9
CVALS[49]=7
CVALS[64]=16

for P in 4 9 16 25 36 49 64; do
  c=${CVALS[$P]}
  echo \"Running 2.5D multiply with P=$P, c=$c...\"
  srun -n $P ./multiply_2_5d 7500 $c
done

# Example trace collection on 16 ranks
echo \"Collecting MPE trace on 16 ranks...\"
srun -n 16 ./multiply_2_5d-trace 2000 ${CVALS[16]}
