#!/bin/bash

foo="srun castep.mpi ml_dos"

# Submit job to cluster
sbatch --ntasks=15  --cpus-per-task=1 --account=def-ravh011 --mem-per-cpu=10G --time=4:00:00 --job-name=FP_chg_pDOS --wrap="$foo"
