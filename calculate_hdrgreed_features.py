import argparse
from utils.HDR_functions import hdr_yuv_read, local_exp, global_exp
from utils.hdrgreed import hdr_greed
from os.path import join
import pandas as pd
import os
from datetime import datetime

from joblib import dump, Parallel, delayed
import socket

parser = argparse.ArgumentParser()

parser.add_argument(
    "--nonlinear", help="select the nonliearity. Support 'lnl'(for single exponential) ,'local_exp', 'global_exp', 'equal' or 'none'.")
parser.add_argument(
    "--parameter", help="the parameter for the nonliear. Use with --nonliear", type=float)
parser.add_argument(
    "--band_pass", help="select the bandpass in NSS. Support 'SPyr', 'NS', 'MSCN', 'DoG'")
parser.add_argument(
    "--wsize", help="the parameter for the nonliear window size. Use with --nonliear and local transform.", type=float)
parser.add_argument(
    "--channel", help="indicate which channel to process. Please provide 0, 1, or 2", type=int)
parser.add_argument(
    "--dog_param1", help="Dog sigma low", type=int, default=60)
parser.add_argument(
    "--dog_param2", help="Dog sigma high", type=int, default=90)
parser.add_argument(
    "--v1lhe", help="LHE after nonlinear", action='store_true')
parser.add_argument(
    "--footprint", help="footprint of LHE transform", type=int, default=15)
parser.set_defaults(v1lhe=False)


if socket.gethostname().find('tacc') > 0:
    scratch = os.environ['SCRATCH']
    vid_pth = '/scratch/06776/kmd1995/video/HDR_2021_fall_yuv_upscaled/fall2021_hdr_upscaled_yuv'
    out_root = join(scratch, 'feats/feats_hdrgreed/')


elif socket.gethostname().find('a51969') > 0:  # Odin
    vid_pth = '/mnt/7e60dcd9-907d-428e-970c-b7acf5c8636a/fall2021_hdr_upscaled_yuv/'
    out_root_vif = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrvifnew/vif'
    out_root_dlm = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrdlmnew/dlm'

elif socket.gethostname().find('895095') > 0:  # DarthVader
    vid_pth = '/media/josh/seagate/hdr_videos/fall2021_hdr_upscaled_yuv/'
    out_root_vif = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrvifnew/vif'
    out_root_dlm = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrdlmnew/dlm'


elif socket.gethostname().find('stormtrooper') >= 0:  # stormtrooper
    vid_pth = '/mnt/fdc70e83-f7d8-42c4-91b4-6dd928077e01/zaixi/fall2021_hdr_upscaled_yuv'
    out_root_vif = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrvifnew/vif'
    out_root_dlm = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrdlmnew/dlm'

elif socket.gethostname().find('a51999') >= 0:  # Diaochan
    vid_pth = '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2021_fall_yuv_upscaled/fall2021_hdr_upscaled_yuv'
    out_root = '/media/zaixi/zaixi_nas/HDRproject/feats/fr_evaluate_HDRLIVE_correct/greed'

args = parser.parse_args()
# ref_path = '/media/josh-admin/nebula_josh/hdr/fall2021_hdr_upscaled_yuv/4k_ref_UrbanLandmark_upscaled.yuv'
# dis_path = '/media/josh-admin/nebula_josh/hdr/fall2021_hdr_upscaled_yuv/1080p_1M_UrbanLandmark_upscaled.yuv'
# vid_pth = '/mnt/fdc70e83-f7d8-42c4-91b4-6dd928077e01/zaixi/fall2021_hdr_upscaled_yuv'

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
        feats.to_csv(os.path.join(outpth,bname.replace('.yuv','.csv')))
        # df = pd.DataFrame(feats).transpose()
        # df['video'] = bname
        return feats
    return

if args.band_pass.lower() != 'dog':
    if args.v1lhe:
        outpth = join(
            out_root, f'greed_{args.nonlinear}_{args.parameter}_w{args.wsize}_c{args.channel}_band{args.band_pass}_v1lhe_{args.footprint}')
    else:
        outpth = join(
            out_root, f'greed_{args.nonlinear}_{args.parameter}_w{args.wsize}_c{args.channel}_band{args.band_pass}')

else:
    if args.v1lhe:
        outpth = join(
            out_root, f'greed_{args.nonlinear}_{args.parameter}_w{args.wsize}_c{args.channel}_band{args.band_pass}-{args.dog_param1}-{args.dog_param2}_v1lhe_{args.footprint}')
    else:
        outpth = join(
            out_root, f'greed_{args.nonlinear}_{args.parameter}_w{args.wsize}_c{args.channel}_band{args.band_pass}-{args.dog_param1}-{args.dog_param2}')
os.makedirs(outpth,exist_ok=True)

r = Parallel(n_jobs=80, verbose=1, backend="multiprocessing")(
    delayed(process_video)(i) for i in range(len(info)))
feats = pd.concat(r)
outpth = join(
    out_root, f'greed_{args.nonlinear}_{args.parameter}_w{args.wsize}_c{args.channel}')
os.makedirs(outpth)
feats.to_csv(join(outpth, 'feats.csv'))
