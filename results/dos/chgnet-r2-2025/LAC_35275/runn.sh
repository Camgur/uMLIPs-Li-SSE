#!/bin/bash

# foo="srun castep.mpi LAC_35275"
foo="srun castep.mpi ml_dos"

# Submit job to cluster
sbatch --ntasks=20  --cpus-per-task=1 --account=def-ravh011 --mem-per-cpu=8G --time=6:00:00 --job-name=LAC_pdos --wrap="$foo"
