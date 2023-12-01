#!/bin/bash
#SBATCH --job-name=ss      # create a short name for your job
#SBATCH --mem=8G       # memory
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=david.j.deng@rutgers.edu

