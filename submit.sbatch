#!/bin/bash
#SBATCH --job-name=2.5Dmul
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=512
#SBATCH --export=ALL
# #SBATCH --reservation=comp422   # uncomment when the reservation is active

module purge
module use /projects/comp422/modulefiles
module load assignment3
module load gompi/2023b        # OpenMPI + GCC
# module load OpenBLAS          # if you built with BLAS=1

EXEC=./multiply_2_5d
N=7500

echo "=== 2.5D Matrix Multiply Scalability (auto‑c) ==="
for P in 4 9 16 25 36 49 64; do
  echo ">> P=$P"
  srun -n $P ${EXEC} ${N}          # c omitted → code picks the best one
done
