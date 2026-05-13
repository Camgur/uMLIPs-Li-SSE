#!/bin/bash

foo="srun castep.mpi ml_dos"

# Submit job to cluster
sbatch --ntasks=18  --cpus-per-task=1 --account=def-ravh011 --mem-per-cpu=10G --time=6:00:00 --job-name=LZP_m3g_pDOS --wrap="$foo"
