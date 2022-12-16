import glob, os 
import pandas as pd

csvs = glob.glob('/scratch/06776/kmd1995/video/LIVE_AQ/features/hdrgreed/greed_none_5.0_w31.0_c0_bandDoG-133-200/*.csv')

lists = []

for c in csvs:
    lists.append(pd.read_csv(c),index_col = 0)

feats = pd.concat(lists)
feats.to_csv('/scratch/06776/kmd1995/video/LIVE_AQ/features/hdrgreed/greed_none_5.0_w31.0_c0_bandDoG-133-200/feats.csv')