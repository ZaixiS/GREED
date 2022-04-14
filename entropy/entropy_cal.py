import numpy as np
from entropy.yuvRead import yuvRead_frame
import os
from entropy.entropy_params import est_params_ggd_temporal
from entropy.entropy_params import est_params_ggd,estimate_ggdparam,generate_ggd
import scipy.signal
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage
from pywt import wavedec2
from pyrtools.pyramids import SteerablePyramidSpace as SPyr
import matplotlib.pyplot as plt
def compute_MS_transform(image, window, extend_mode='reflect'):
    h,w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image

def video_process(vid_path, width, height, bit_depth, gray, T, filt, num_levels, scales):
    
    #Load WPT filters
    
    filt_path = 'WPT_Filters/' + filt + '_wpt_' + str(num_levels) + '.mat'
    wfun = scipy.io.loadmat(filt_path)
    wfun = wfun['wfun']
        
    blk = 5
    sigma_nsq = 0.1
    win_len = 7
    
    entropy = {}
    vid_stream = open(vid_path,'r')
    
    for scale_factor in scales:
        sz = 2**(-scale_factor)
        frame_data = np.zeros((int(height*sz),int(width*sz),T))
        
        spatial_sig = []
        spatial_ent = []
        for frame_ind in range(0, T):
            frame_data[:,:,frame_ind],_,_ = \
            yuvRead_frame(vid_stream, width, height, \
                                  frame_ind, bit_depth, gray, sz)
            
            window = gen_gauss_window((win_len-1)/2,win_len/6)
            MS_frame = compute_MS_transform(frame_data[:,:,frame_ind], window)
        
            spatial_sig_frame, spatial_ent_frame = est_params_ggd(MS_frame, blk, sigma_nsq)
            spatial_sig.append(spatial_sig_frame)
            spatial_ent.append(spatial_ent_frame)
        
        #Wavelet Packet Filtering
        #valid indices for start and end points
        valid_lim = frame_data.shape[2] - wfun.shape[1] + 1
        start_ind = wfun.shape[1]//2 - 1
        dpt_filt = np.zeros((frame_data.shape[0], frame_data.shape[1],\
                             2**num_levels - 1, valid_lim))
        
        for freq in range(wfun.shape[0]):
            dpt_filt[:,:,freq,:] = scipy.ndimage.filters.convolve1d(frame_data,\
                    wfun[freq,:],axis=2,mode='constant')[:,:,start_ind:start_ind + valid_lim]
        
        temporal_sig, temporal_ent = est_params_ggd_temporal(dpt_filt, blk, \
                                                                     sigma_nsq)
        

        #convert lists to numpy arrays
        spatial_sig = np.array(spatial_sig)
        spatial_sig[np.isinf(spatial_sig)] = 0
        
        spatial_ent = np.array(spatial_ent)
        spatial_ent[np.isinf(spatial_ent)] = 0
        
        temporal_sig = np.array(temporal_sig)
        temporal_sig[np.isinf(temporal_sig)] = 0
        
        temporal_ent = np.array(temporal_ent)
        temporal_ent[np.isinf(temporal_ent)] = 0
        
        #calculate rescaled entropy
        spatial_ent_scaled = np.log(1 + spatial_sig**2) * spatial_ent
        temporal_ent_scaled = np.log(1 + temporal_sig**2) * temporal_ent
        
        # reshape spatial entropy to heightxwidthxnum_frames
        spatial_ent_scaled = spatial_ent_scaled.transpose(1,2,0)
        
        entropy['spatial_scale' + str(scale_factor)] = spatial_ent_scaled[:,:,:valid_lim]
        entropy['temporal_scale' + str(scale_factor)] = temporal_ent_scaled
        
    return entropy


def entrpy_frame(frame_data,vname = None):
    pyr = SPyr(frame_data, 4, 5, 'reflect1').pyr_coeffs
    subband_keys = []
    for key in list(pyr.keys())[1:-2:3]:
        subband_keys.append(key)
    subband_keys.reverse()

    blk = 5
    sigma_nsq = 0.1

    ents = []
    for i, subband_key in enumerate(subband_keys):
        subband_coef = pyr[subband_key]
        spatial_sig_frame, spatial_ent_frame = est_params_ggd(subband_coef, blk, sigma_nsq)
        plot = True
        if plot:
            savepth = './plots/ggd'
            try:
                os.makedirs(savepth)
            except:
                pass
            SPcoefpth = './plots/SPycoef'
            try:
                os.makedirs(SPcoefpth)
            except:
                pass
            plt.imsave(os.path.join(SPcoefpth,os.path.basename(vname)[:-4]+'_SPycoef.jpg'),subband_coef,cmap = 'gray')
            im = plt.hist(subband_coef.flatten(),bins=1900,range = [-1000,1000],density=True,label='SPyr coefficient')
            x = np.linspace(-1000,1000,1600)
            alphaparam,sigma = estimate_ggdparam(subband_coef.flatten())
            ggd= generate_ggd(x,alphaparam,sigma )
            plt.plot(x,ggd,label = f'ggd fit,alpha={alphaparam}')
            plt.legend()
            plt.title(os.path.basename(vname)[:-4] + '_coef')
            plt.ylim((0, 0.002))
            plt.savefig(os.path.join(savepth,os.path.basename(vname)[:-4]+'_ggd.pdf'))
            plt.savefig(os.path.join(savepth,os.path.basename(vname)[:-4]+'_ggd.jpg'))
            plt.cla()
            return
        spatial_sig_frame = np.array(spatial_sig_frame)
        spatial_sig_frame[np.isinf(spatial_sig_frame)] = 0
        
        spatial_ent_frame = np.array(spatial_ent_frame)
        spatial_ent_frame[np.isinf(spatial_ent_frame)] = 0
        spatial_ent_scaled = np.log(1 + spatial_sig_frame**2) * spatial_ent_frame
        ents.append(spatial_ent_scaled)
    return ents
        