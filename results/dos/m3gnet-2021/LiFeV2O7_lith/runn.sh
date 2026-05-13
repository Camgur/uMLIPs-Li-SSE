#!/bin/bash

foo="srun castep.mpi ml_dos"

# Submit job to cluster
sbatch --ntasks=40  --cpus-per-task=1 --account=def-ravh011 --mem-per-cpu=10G --time=24:00:00 --job-name=LFVO_L_m3g_pDOS --wrap="$foo"
