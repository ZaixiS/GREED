spaces = ['ycbcr']
channels = [0]
nonlinear = ['local_exp']
counter = 0
parameters = [-0.5,0.5,1,-1,2,-2,5,-5]
dog_param = [[30,45],[60,90],[133,200]]
for s in spaces:
    for c in channels:
        for n in nonlinear:
            for p in parameters:
                for d1,d2 in dog_param: 
                    f = open(f"./scripts/job_{counter}.script", "w")
                        
                    string = f"""#!/bin/bash
#SBATCH -N 2
#SBATCH -n 90
#SBATCH -o vmaf{counter}.log
#SBATCH -J vmaf{counter}
#SBATCH -p skx-normal
#SBATCH -t 1:10:00
#SBATCH --mail-user=zxshang@utexas.edu
#SBATCH --mail-type=all
source ~/anaconda3/bin/activate
conda activate base
conda activate hdrvmaf
python calculate_hdrgreed_features_2022.py --nonlinear {n} --parameter {p} --band_pass dog --wsize 31 --channel  0 --dog_param1 {d1} --dog_param2 {d2}

                """
                    f.write(string)
                    f.close()
                    counter += 1

spaces = ['ycbcr']
channels = [0]
nonlinear = ['none']
parameters = [0]
dog_param = [[30,45],[60,90],[133,200]]
for s in spaces:
    for c in channels:
        for n in nonlinear:
            for p in parameters:
                for d1,d2 in dog_param: 
                    f = open(f"./scripts/job_{counter}.script", "w")
                        
                    string = f"""#!/bin/bash
#SBATCH -N 2
#SBATCH -n 90
#SBATCH -o vmaf{counter}.log
#SBATCH -J vmaf{counter}
#SBATCH -p skx-normal
#SBATCH -t 2:00:00
#SBATCH --mail-user=zxshang@utexas.edu
#SBATCH --mail-type=all
source ~/anaconda3/bin/activate
conda activate base
conda activate hdrvmaf
python calculate_hdrgreed_features_2022.py --nonlinear {n} --parameter {p} --band_pass dog --wsize 31 --channel  0 --dog_param1 {d1} --dog_param2 {d2}

                """
                    f.write(string)
                    f.close()
                    counter += 1
