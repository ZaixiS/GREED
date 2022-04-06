import os 
import glob

from joblib import Parallel, delayed


in_pth = "/scratch/06776/kmd1995/video/HDR_2022_SPRING_mp4"
out_pth = "/scratch/06776/kmd1995/video/HDR_2022_SPRING_yuv"
files = glob.glob(os.path.join(in_pth,'*.mp4'))

def generate(idx):
    inname = files[idx]
    base = os.path.basename(inname)
    base = base.replace('_AmazonProprietaryConfidential.mp4','.yuv')
    outname = os.path.join(out_pth,base)
    print(outname)
    cmd = f"ffmpeg -y -nostdin -i {inname} -an -vf scale=3840:2160 -c:v rawvideo -pix_fmt yuv420p10le {outname} "
    os.system(cmd)







Parallel(n_jobs=32)(delayed(generate)(i) for i in range(400))