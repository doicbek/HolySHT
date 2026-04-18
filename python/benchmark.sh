#!/bin/bash
#SBATCH -J holysht_bench
#SBATCH -o benchmark_%j.out
#SBATCH -e benchmark_%j.err
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-03:00

module load python cmake

cd /n/home01/dbeck/software/holysht/python

# Rebuild on compute node (correct -march=native for Xeon 8358)
CONDA_PYTHON=$(conda run --no-capture-output -n pyenv python -c "import sys; print(sys.executable)")
rm -rf build && mkdir build && cd build
cmake .. -DPython3_EXECUTABLE="$CONDA_PYTHON" -DPYTHON_EXECUTABLE="$CONDA_PYTHON" -DPython3_FIND_STRATEGY=LOCATION
cmake --build . -j$(nproc)
cd ..

export PYTHONPATH=.:build
export HOLYSHT_NTHREADS=$SLURM_CPUS_PER_TASK

conda run --no-capture-output -n pyenv python benchmark.py
