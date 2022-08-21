#!/bin/bash
#SBATCH --job-name=sample              
#SBATCH --output=%jlog.log               # Standard output and error log
#SBATCH --error=%jerror.err
#SBATCH --nodes=1                        # Run all processes on a single node
#SBATCH --ntasks=10                      # Number of processes
#SBATCH --partition=freecpuQ                # fatQ-80cores,1536GB; cpuQ-48cores,192GB


date;hostname;pwd
echo -e "\n"
echo "Running program on $SLURM_CPUS_ON_NODE CPU cores,${parall_nodes} parall_nodes for ${bi}th group parameters"
echo -e "\n"

#/public/home/hpc220115/.conda/envs/dlenv/bin/python  gen_samples_for_image_reconstruct.py --bin_num_total ${b} --bin_num ${bi} --parall_nodes ${parall_nodes}
# /public/home/hpc202111105/anaconda3/bin/python
/public/home/hpc202111105/.conda/envs/py3.8/bin/python  gen_samples_for_heat_conduction.py --bin_num_total ${b} --bin_num ${bi} --parall_nodes ${parall_nodes}


date
