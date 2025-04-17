#!/bin/bash
#SBATCH --job-name=2.5Dmul
#SBATCH --nodes=4                # total nodes you want
#SBATCH --ntasks-per-node=16     # tasks per node → 64 total ranks
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=512
#SBATCH --export=ALL

module purge
module use /projects/comp422/modulefiles
module load assignment3

# Map P → c so that P/c is a perfect square
declare -A CVALS=( [4]=1 [9]=1 [16]=4 [25]=5 [36]=9 [49]=7 [64]=16 )

for P in 4 9 16 25 36 49 64; do
  c=${CVALS[$P]}
  echo "=== Running 2.5D multiply with P=$P, c=$c ==="
  mpirun -np $P ./multiply_2_5d 7500 $c
done

# And collect an MPE trace on 16 ranks if you like:
echo "=== Trace run on 16 ranks ==="
mpirun -np 16 ./multiply_2_5d-trace 2000 ${CVALS[16]}
