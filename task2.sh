#!/bin/bash
#SBATCH -N 2
#SBATCH -n 64
#SBATCH -o greed2.log
#SBATCH -J greed2
#SBATCH -p skx-normal
#SBATCH -t 2:00:00
#SBATCH --mail-user=zxshang@utexas.edu
#SBATCH --mail-type=all
source ~/anaconda3/bin/activate
conda activate base
conda activate hdrvmaf

python calculate_hdrgreed_features_2022.py --nonlinear local_exp --parameter 0.5 --wsize 31 --channel 0 --band_pass MSCN
python calculate_hdrgreed_features_2022.py --nonlinear local_exp --parameter -0.5 --wsize 31 --channel 0 --band_pass MSCN
python calculate_hdrgreed_features_2022.py --nonlinear local_exp --parameter 5 --wsize 31 --channel 0 --band_pass MSCN
                