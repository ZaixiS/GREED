import glob, os 
import pandas as pd

csvs = glob.glob('/home/zs5397/code/GREED/temp_feat/greed_lhe_None_w31.0_c0_bandmscn/*.csv')

lists = []

for c in csvs:
    lists.append(pd.read_csv(c,index_col = 0))

feats = pd.concat(lists)
feats.to_csv('/home/zs5397/code/GREED/temp_feat/greed_lhe_None_w31.0_c0_bandmscn/feats.csv')