#!/bin/bash
#SBATCH --job-name=2.5Dmul
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=512    # 512 MB is fine for b ≤ 3750
#SBATCH --export=ALL

module load assignment3 gompi/2023b

EXEC=./multiply_2_5d
N=7500

# explicit c avoids dangerously large tiles
declare -A CVAL=( [4]=1 [9]=1 [16]=4 [25]=1 [36]=9 [49]=1 [64]=16 )

echo "=== 2.5D scalability (explicit c) ==="
for P in 4 9 16 25 36 49 64; do
  echo ">> P=$P, c=${CVAL[$P]}"
  srun -n $P $EXEC $N ${CVAL[$P]}
done
