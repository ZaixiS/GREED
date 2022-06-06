import argparse
from utils.HDR_functions import hdr_yuv_read, local_exp, global_exp
from entropy.entropy_cal_plot import entrpy_frame
from entropy.entropy_params import estimate_ggdparam, generate_ggd
import pandas as pd
import numpy as np
import pdb
import os
from matplotlib.pyplot import imsave
from skimage.filters import rank
from skimage.morphology import disk
from datetime import datetime
import matplotlib.pyplot as plt


def cal_difference_by_band(ref_ent, dis_ent):
    return np.array([np.abs((ref_ent[i]-dis_ent[i])).mean() for i in range(len(ref_ent))])


def hdr_greed(ref_name, dis_name, framenum, args):
    # h = 1728  # hs[dis_index]
    # w = 3072  # ws[dis_index]
    h = args.h  # hs[dis_index]
    w = args.w  # ws[dis_index]
    skip = 25
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print(dis_name)
    channel = args.channel

    ref_file_object = open(ref_name)
    dis_file_object = open(dis_name)
    framelist = list(range(0, 9999, skip))
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

        path1 = './plots/frames/'

        if not os.path.exists(path1):
            os.makedirs(path1)
        plt.imsave(os.path.join(path1, os.path.basename(ref_name)+f'frame_fn_{framenum}.jpg'),
                   ref_singlechannel, cmap='gray')
        plt.imsave(os.path.join(path1, os.path.basename(dis_name)+f'frame_fn_{framenum}.jpg'),
                   dis_singlechannel, cmap='gray')

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
            ref_ent_1 = entrpy_frame(nonlinear_ref, args, ref_name, framenum)
            dis_ent_1 = entrpy_frame(nonlinear_dis, args, ref_name, framenum)
            ent_diff_1 = cal_difference_by_band(ref_ent_1, dis_ent_1)

        elif(nonlinear == 'global_exp'):
            nonlinear_ref = global_exp(ref_singlechannel, args.parameter)
            nonlinear_dis = global_exp(dis_singlechannel, args.parameter)
            ref_ent_1 = entrpy_frame(nonlinear_ref, args, ref_name, framenum)
            dis_ent_1 = entrpy_frame(nonlinear_dis, args, ref_name, framenum)
            ent_diff_1 = cal_difference_by_band(ref_ent_1, dis_ent_1)

        elif(nonlinear == 'equal'):
            footprint = disk(30)
            ref_singlechannel = ref_singlechannel / \
                np.max(ref_singlechannel)*1023
            dis_singlechannel = dis_singlechannel / \
                np.max(dis_singlechannel)*1023
            ref_singlechannel = ref_singlechannel.astype(np.uint16)
            dis_singlechannel = dis_singlechannel.astype(np.uint16)
            path_equal = './plots/frames_equal_31/'
            if not os.path.exists(path_equal):
                os.makedirs(path_equal)

            img_eq_ref = rank.equalize(ref_singlechannel, selem=footprint)
            img_eq_dis = rank.equalize(dis_singlechannel, selem=footprint)
            plt.imsave(os.path.join(path_equal, os.path.basename(ref_name)+f'frame_equal31_fn_{framenum}.jpg'),
                       img_eq_ref, cmap='gray')
            plt.imsave(os.path.join(path_equal, os.path.basename(dis_name)+f'frame_equal31_fn_{framenum}.jpg'),
                       img_eq_dis, cmap='gray')
            ref_ent_1 = entrpy_frame(img_eq_ref, args, ref_name, framenum)
            dis_ent_1 = entrpy_frame(img_eq_dis, args, dis_name, framenum)

        else:
            ref_ent_none = entrpy_frame(
                ref_singlechannel, args, ref_name, framenum)

            dis_ent_none = entrpy_frame(
                dis_singlechannel, args, dis_name, framenum)
            ent_diff_1 = cal_difference_by_band(ref_ent_none, dis_ent_none)

        # feats.append(ent_diff_1)
    # feats = np.stack(feats)
    # feats = feats.mean(axis=0)
    # now = datetime.now()

    # current_time = now.strftime("%H:%M:%S")
    # print("Finish Time =", current_time)
    return [0, 0, 0, 0]
