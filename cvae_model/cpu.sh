#!/bin/bash

#!/bin/bash
#SBATCH --job-name=gen_samples
#SBATCH --output=log%j.log               # Standard output and error log
#SBATCH --nodes=1                        # Run all processes on a single node
#SBATCH --ntasks=5                      # Number of processes
#SBATCH --partition=gpu8Q                # fatQ-80cores,1536GB; cpuQ-48cores,192GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpuQ
#SBATCH -A pi_niuyuanling

source activate py3.8
python cvae.py
