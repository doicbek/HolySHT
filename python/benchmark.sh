#!/bin/bash
#SBATCH -J holysht_bench
#SBATCH -o benchmark_%j.out
#SBATCH -e benchmark_%j.err
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-01:00

module load python cmake

cd /n/home01/dbeck/software/holysht/python
export PYTHONPATH=.:build
export HOLYSHT_NTHREADS=$SLURM_CPUS_PER_TASK

conda run --no-capture-output -n pyenv python benchmark.py
