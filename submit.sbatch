#!/bin/bash
#SBATCH --job-name=2.5Dmul
#SBATCH --nodes=4                 # hard upper bound for all runs
#SBATCH --ntasks=64               # 4 nodes × 16 ranks each
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=512
#SBATCH --export=ALL
# #SBATCH --reservation=comp422   # activate if you have one

module purge
module use /projects/comp422/modulefiles
module load assignment3 gompi/2023b   # OpenMPI + GCC + OpenBLAS + MPE

EXEC=./multiply_2_5d
N=7500

# Valid P‑c table (49 dropped; everything fits 512 MB/rank)
declare -A CVAL=(
  [4]=1   # p=2
  [9]=1   # p=3
  [16]=4  # p=2
  [25]=1  # p=5
  [36]=9  # p=2
  [64]=16 # p=2
)

echo "=== 2.5D scalability ==="
for P in 4 9 16 25 36 64; do
  c=${CVAL[$P]}
  echo ">> P=$P  c=$c"
  srun -n $P $EXEC $N $c          # no -N, no warnings
done
