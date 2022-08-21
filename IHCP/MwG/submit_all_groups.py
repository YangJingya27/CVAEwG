import os
bin_num_total = 2

ntasks = 10
parall_nodes = ntasks//2
for i in range(bin_num_total):
    sbatch_i = f"sbatch --export=parall_nodes={parall_nodes},b={bin_num_total},bi={i} " \
               f"-J outer " \
               f"--ntasks {ntasks} " \
               f"deblur_submit_template.sh"
    os.system(sbatch_i)
