#!/bin/bash
#SBATCH --job-name=2.5D-trace
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4    # 4 × 4 = 16 total
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=512
#SBATCH --export=ALL

module purge
module use /projects/comp422/modulefiles
module load assignment3 gompi/2023b

EXEC=./multiply_2_5d-trace
N=2000

echo "Collecting MPE trace on 16 ranks..."
srun -n 16 ${EXEC} ${N}
