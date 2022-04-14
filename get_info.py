import os
import glob
import pandas as pd


files = glob.glob("/scratch/06776/kmd1995/video/HDR_2022_SPRING_yuv/*.yuv")
names = [os.path.basename(files[i]) for i in range(len(files))]
fnos = [int(os.path.getsize(files[i])/2160/3840/3) for i in range(len(files))]
content =  [name.split('_')[0] for name in names]
ref = [content_+'_res2160_mu100000_sigma0_rate100000_aqOn.yuv' for content_ in content]

print(fnos)

info = pd.DataFrame({'yuv':names,'framenos':fnos,'content':content,'ref':ref})
print(info)
info.to_csv('spring2022_yuv_info.csv')