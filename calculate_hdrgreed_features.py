import argparse
from utils.HDR_functions import hdr_yuv_read, local_exp, global_exp
from utils.hdrgreed import hdr_greed
from os.path import join
import pandas as pd
import os
from datetime import datetime
from joblib import dump, Parallel, delayed

parser = argparse.ArgumentParser()

parser.add_argument(
    "--nonlinear", help="select the nonliearity. Support 'local_exp', 'global_exp', 'equal' or 'none'.")
parser.add_argument(
    "--parameter", help="the parameter for the nonliear. Use with --nonliear", type=float)
parser.add_argument(
    "--wsize", help="the parameter for the nonliear window size. Use with --nonliear and local transform.", type=float)
parser.add_argument(
    "--channel", help="indicate which channel to process. Please provide 0, 1, or 2", type=int)


args = parser.parse_args()
ref_path = '/media/josh-admin/nebula_josh/hdr/fall2021_hdr_upscaled_yuv/4k_ref_UrbanLandmark_upscaled.yuv'
dis_path = '/media/josh-admin/nebula_josh/hdr/fall2021_hdr_upscaled_yuv/1080p_1M_UrbanLandmark_upscaled.yuv'
vid_pth = '/mnt/fdc70e83-f7d8-42c4-91b4-6dd928077e01/zaixi/fall2021_hdr_upscaled_yuv'

# feats = hdr_greed(ref_path,dis_path,args)
info = pd.read_csv('fall2021_yuv_rw_info.csv', index_col=0)


def process_video(ind):
    video = info['yuv'][ind]
    ref = info['ref'][ind]
    fcount = info['framenos'][ind]
    bname = os.path.basename(video)
    if video != ref:
        feats = hdr_greed(join(vid_pth, video),
                          join(vid_pth, ref), fcount, args)
    df = pd.DataFrame(feats).transpose()
    df['video'] = bname
    return df


# # process_video(164)
# process_video(163)
r = Parallel(n_jobs=31, verbose=1, backend="multiprocessing")(
    delayed(process_video)(i) for i in range(len(info)))
feats = pd.concat(r)
feats.to_csv('feats.csv')
