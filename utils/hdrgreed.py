import argparse
from utils.HDR_functions import hdr_yuv_read,local_exp,global_exp
from entropy.entropy_cal import entrpy_frame
import pandas as pd


def hdr_greed(ref_name,dis_name,args):
    h = 2160 #hs[dis_index]
    w = 3840 #ws[dis_index]
    
    framenos = 100
    channel = args.channel
    ref_file_object = open(ref_name)
    dis_file_object = open(dis_name)
    framelist =  list(range(0,framenos,50))
    nonlinear = args.nonlinear
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
            
            nonlinear_ref = local_exp(ref_singlechannel,-args.parameter,args.wsize)
            nonlinear_dis = local_exp(dis_singlechannel,-args.parameter,args.wsize)
            ref_ent_2 = entrpy_frame(nonlinear_ref)
            dis_ent_2 = entrpy_frame(nonlinear_dis)         
            
        elif(nonlinear == 'global_exp'):
            nonlinear_ref = global_exp(ref_singlechannel,args.parameter)
            nonlinear_dis = global_exp(dis_singlechannel,args.parameter)
            ref_ent_1 = entrpy_frame(nonlinear_ref)
            dis_ent_1 = entrpy_frame(nonlinear_dis)
            nonlinear_ref = global_exp(ref_singlechannel,-args.parameter)
            nonlinear_dis = global_exp(dis_singlechannel,-args.parameter)
            ref_ent_2 = entrpy_frame(nonlinear_ref)
            dis_ent_2 = entrpy_frame(nonlinear_dis)   
        
        else:
            ref_ent_none = entrpy_frame(nonlinear_ref)
            dis_ent_none = entrpy_frame(nonlinear_dis)   
            
