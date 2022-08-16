import argparse
from utils.HDR_functions import hdr_yuv_read, local_exp, global_exp
from entropy.entropy_cal_lhe_spyr import entrpy_frame
from entropy.entropy_params import estimate_ggdparam, generate_ggd
import pandas as pd
import numpy as np
import pdb

from matplotlib.pyplot import imsave
from skimage.filters import rank
from skimage.morphology import disk
from datetime import datetime


def cal_difference_by_band(ref_ent, dis_ent):
    return np.array([np.abs((ref_ent[i]-dis_ent[i])).mean() for i in range(len(ref_ent))])


def hdr_greed(ref_name, dis_name, framenum, args):
    h = 2160  # hs[dis_index]
    w = 3840  # ws[dis_index]
    skip = 25
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print(dis_name)
    channel = args.channel

    ref_file_object = open(ref_name)
    dis_file_object = open(dis_name)
    framelist = list(range(0, framenum, skip))
    nonlinear = args.nonlinear
    feats = []
    for framenum in framelist:
        try:
            ref_multichannel = hdr_yuv_read(ref_file_object, framenum, h, w)
            dis_multichannel = hdr_yuv_read(dis_file_object, framenum, h, w)

        except Exception as e:
            print(e)
            break

        ref_singlechannel = ref_multichannel[channel]
        dis_singlechannel = dis_multichannel[channel]

        if(nonlinear == 'local_exp'):
            # nonlinear_ref1 = nonlinear_ref*10
            nonlinear_ref = local_exp(
                ref_singlechannel, args.parameter, args.wsize)
            nonlinear_dis = local_exp(
                dis_singlechannel, args.parameter, args.wsize)
            nonlinear_ref = local_exp(
                ref_singlechannel, -args.parameter, args.wsize)
            nonlinear_dis = local_exp(
                dis_singlechannel, -args.parameter, args.wsize)
            ref_ent_1 = entrpy_frame(nonlinear_ref, args)
            dis_ent_1 = entrpy_frame(nonlinear_dis, args)
            ent_diff_1 = cal_difference_by_band(ref_ent_1, dis_ent_1)

        elif(nonlinear == 'global_exp'):
            nonlinear_ref = global_exp(ref_singlechannel, args.parameter)
            nonlinear_dis = global_exp(dis_singlechannel, args.parameter)
            ref_ent_1 = entrpy_frame(nonlinear_ref, args)
            dis_ent_1 = entrpy_frame(nonlinear_dis, args)
            ent_diff_1 = cal_difference_by_band(ref_ent_1, dis_ent_1)

        elif(nonlinear == 'equal'):
            footprint = disk(30)
            ref_singlechannel = ref_singlechannel / \
                np.max(ref_singlechannel)*1023
            dis_singlechannel = dis_singlechannel / \
                np.max(dis_singlechannel)*1023
            ref_singlechannel = ref_singlechannel.astype(np.uint16)
            dis_singlechannel = dis_singlechannel.astype(np.uint16)

            img_eq_ref = rank.equalize(ref_singlechannel, selem=footprint)
            img_eq_dis = rank.equalize(dis_singlechannel, selem=footprint)

            ref_ent_1 = entrpy_frame(img_eq_ref, args)
            dis_ent_1 = entrpy_frame(img_eq_dis, args)
            ent_diff_1 = cal_difference_by_band(ref_ent_1, dis_ent_1)

        else:
            ref_ent_none = entrpy_frame(ref_singlechannel, args)

            dis_ent_none = entrpy_frame(dis_singlechannel, args)
            ent_diff_1 = cal_difference_by_band(ref_ent_none, dis_ent_none)

        feats.append(ent_diff_1)
    feats = np.stack(feats)
    feats = feats.mean(axis=0)
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Finish Time =", current_time)
    return feats
