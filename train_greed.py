from training import *
import glob
import pandas as pd
import numpy as np
import sklearn
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import itertools
# from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from os.path import join
from mpi4py import MPI
import sys
import re

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# The input path for the feature files
feats_pth_root = '/home/zs5397/code/GREED/temp_feat/greed_none_5.0_w31.0_c0_bandmscn_v1lhe_15'
# The path to the input score file
score_file = '/home/zs5397/code/training/Spring_2022_score.csv'
# The output path for prediction csv file. i.e. the predicted scores for each video. It's not strictly defined but rather just the average prediction of each train-test split.
pred_csvpth = './predicts/greed/'
# Output path for scatterplot.
plot_pth = './plots/fr'
os.makedirs(pred_csvpth, exist_ok=True)
os.makedirs(plot_pth, exist_ok=True)
N_times = 100
df_score = pd.read_csv(score_file, index_col=0)
counter = 0
configs = []

for method in ['mscn']:

    configs.append([method])


print('number of options ', len(configs))
methods = []
sroccs = []
plccs = []
rmses = []
allres = []
for cfig_index in range(rank, len(configs), size):
    print(configs[cfig_index])
    method = configs[cfig_index][0]

    feature = pd.read_csv(
        join(feats_pth_root, f'feats_{method}.csv'), index_col=0)
    feature1 = pd.read_csv(
        join(feats_pth_root, 'feats_mscnlhespyr.csv'), index_col=0)
    feature = feature.merge(feature1, on='video')
    print(feature.shape)
    feature['content'] = feature['video'].map(lambda x: x.split('_')[0])
    # feature['video'] = feature['video'].map(lambda x: x[:-4])
    feature = feature.merge(df_score[['video', 'sureal_DMOS']])
    feature = feature.rename({'sureal_DMOS': 'score'}, axis=1)
    r = []
    for times in range(N_times):
        r_eachtime = train_for_srocc_svr(feature)
        r.append(r_eachtime)
    plotname = os.path.join(
        plot_pth, f'par__{method}_both_scatter.jpg')

    if not os.path.exists(os.path.dirname(plotname)):
        os.makedirs(os.path.dirname(plotname))
    srocc, plcc, rmse, pred = unpack_and_plot(
        r, plotname, feature, get_pred=True)
    pred_csvfile = join(
        pred_csvpth, f'par__{method}_both_.csv')

    pred.to_csv(pred_csvfile)
    plccs.append(plcc)
    rmses.append(rmse)
    sroccs.append(srocc)
    methods.append(method)

    res = pd.DataFrame({
        'method': methods, 'srocc': sroccs, 'plcc': plccs, 'rmse': rmses})
    allres.append(res)

print("process {} send data to root...".format(rank))
recv_data = comm.gather(res, root=0)
if rank == 0:
    print("process {} gather all data ...".format(rank))
    df = pd.concat(recv_data)
    df.to_csv('eval_greed2both.csv')
