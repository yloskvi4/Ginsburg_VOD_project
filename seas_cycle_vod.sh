#!/bin/sh
#

#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=seasCy_vod    # The job name

####SBATCH --partition=ocp_gpu       # Request ocp_gpu nodes first. If none are available, the scheduler will request non-OCP gpu nodes.
###SBATCH --gres=gpu:1              # Request 1 gpu (Up to 2 gpus per GPU node)
#####SBATCH --constraint=rtx8000      # You may specify rtx8000 or v100s or omit this line for either


#SBATCH -N 1                     # The number of nodes to request
#SBATCH -c 5                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=1-17:30            # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=150G         # The memory the job will use per cpu core
 
date
module load anaconda/3-2021.05
#conda init bash
#conda create -n tf tensorflow
#conda activate tf 
source activate tf

#Command to execute Python program
python VOD_project/seas_cycle_vod.py
date
 
# End of script
