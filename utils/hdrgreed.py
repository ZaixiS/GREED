import argparse
from utils.HDR_functions import hdr_yuv_read,local_exp,global_exp
from entropy.entropy_cal import entrpy_frame
import pandas as pd
import numpy as np

def cal_difference_by_band(ref_ent, dis_ent):
    return  np.array([np.abs((ref_ent[i]-dis_ent[i])).mean() for i in range(len(ref_ent))])


def hdr_greed(ref_name,dis_name,framenum,args):
    h = 2160 #hs[dis_index]
    w = 3840 #ws[dis_index]
    skip = 50
    channel = args.channel
    framenum =101
    ref_file_object = open(ref_name)
    dis_file_object = open(dis_name)
    framelist =  list(range(0,framenum,skip))
    nonlinear = args.nonlinear
    feats = []
    for framenum in framelist:
        try:
            ref_multichannel = hdr_yuv_read(ref_file_object,framenum,h,w)
            dis_multichannel = hdr_yuv_read(dis_file_object,framenum,h,w)

        except Exception as e:
            print(e)
            break
    

        ref_singlechannel = ref_multichannel[channel]
        dis_singlechannel = dis_multichannel[channel]       
        if(nonlinear == 'local_exp'):
                    
            nonlinear_ref = local_exp(ref_singlechannel,args.parameter,args.wsize)
            nonlinear_dis = local_exp(dis_singlechannel,args.parameter,args.wsize)
            ref_ent_1 = entrpy_frame(nonlinear_ref)
            dis_ent_1 = entrpy_frame(nonlinear_dis)
            ent_diff_1 = cal_difference_by_band(ref_ent_1,dis_ent_1)
            # nonlinear_ref = local_exp(ref_singlechannel,-args.parameter,args.wsize)
            # nonlinear_dis = local_exp(dis_singlechannel,-args.parameter,args.wsize)
            # ref_ent_2 = entrpy_frame(nonlinear_ref)
            # dis_ent_2 = entrpy_frame(nonlinear_dis)         
            # ent_diff_2 = cal_difference_by_band(ref_ent_2,dis_ent_2)
            
        elif(nonlinear == 'global_exp'):
            nonlinear_ref = global_exp(ref_singlechannel,args.parameter)
            nonlinear_dis = global_exp(dis_singlechannel,args.parameter)
            ref_ent_1 = entrpy_frame(nonlinear_ref)
            dis_ent_1 = entrpy_frame(nonlinear_dis)
            ent_diff_1 = cal_difference_by_band(ref_ent_1,dis_ent_1)
            # nonlinear_ref = global_exp(ref_singlechannel,-args.parameter)
            # nonlinear_dis = global_exp(dis_singlechannel,-args.parameter)
            # ref_ent_2 = entrpy_frame(nonlinear_ref)
            # dis_ent_2 = entrpy_frame(nonlinear_dis)   
            # ent_diff_2 = cal_difference_by_band(ref_ent_2,dis_ent_2)
        else:
            ref_ent_none = entrpy_frame(nonlinear_ref)
            dis_ent_none = entrpy_frame(nonlinear_dis)   
            ent_diff_1 = cal_difference_by_band(ref_ent_none,dis_ent_none)
        feats.append(ent_diff_1)
    feats = np.stack(feats)
    feats = feats.mean(axis=0)
    return feats
