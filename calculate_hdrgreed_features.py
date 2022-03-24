import argparse
from utils.HDR_functions import hdr_yuv_read,local_exp,global_exp
from utils.hdrgreed import hdr_greed

import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument("--nonlinear",help="select the nonliearity. Support 'local_exp', 'global_exp' or 'none'.")
parser.add_argument("--parameter",help="the parameter for the nonliear. Use with --nonliear",type=float)
parser.add_argument("--wsize",help="the parameter for the nonliear window size. Use with --nonliear and local transform.",type=float)
parser.add_argument("--channel",help="indicate which channel to process. Please provide 0, 1, or 2",type=int)

args = parser.parse_args()
ref_path = '/media/josh-admin/nebula_josh/hdr/fall2021_hdr_upscaled_yuv/4k_ref_UrbanLandmark_upscaled.yuv'
dis_path = '/media/josh-admin/nebula_josh/hdr/fall2021_hdr_upscaled_yuv/1080p_1M_UrbanLandmark_upscaled.yuv'
feats = hdr_greed(ref_path,dis_path,args)
