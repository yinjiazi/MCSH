import os
import random
import pickle
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def read_pickle(file_path):
    import pickle
    with open(file_path,'rb') as f:
        text = pickle.load(f)

    return text

def read_json_files(file):
    import glob
    import json
    json_files = glob.glob(file)
    gather_eposide_dict = {}

    for file in json_files:
        eposide = json.load(open(file,encoding='utf8'))
        gather_eposide_dict.update(eposide)

    return gather_eposide_dict


def ten_fold_split():
    #读取各种特征文件
    import joblib
    text = read_pickle('feature_extractor/text/bert.p')
    audio = joblib.load('feature_extractor/audio/audio_pad.p')
    vision = joblib.load('feature_extractor/video/video_pad.p')
    json_file = read_json_files('json_data_dir/*')


    #定义多层字典
    from collections import defaultdict
    combined_file = defaultdict(dict)

    #循环十集 十次交叉验证
    for eposide in range(1,11):

        eposide_1 = '1_001'
        len_v = 0
        len_t = 0
        keys = json_file.keys()
        for key in keys:
            if key.split('_')[0] == str(eposide):
                len_v += 1
            else:
                len_t += 1
        #得到valid 和 train的维度 （长度）

        train_text_list = np.empty((len_t,text[eposide_1].shape[0], text[eposide_1].shape[1]),dtype=np.float32)
        valid_text_list = np.empty((len_v,text[eposide_1].shape[0], text[eposide_1].shape[1]),dtype=np.float32)

        train_audio_list = np.empty((len_t, audio[eposide_1].shape[0], audio[eposide_1].shape[1]),dtype=np.float32)
        valid_audio_list = np.empty((len_v, audio[eposide_1].shape[0], audio[eposide_1].shape[1]),dtype=np.float32)

        train_vision_list = np.empty((len_t, vision[eposide_1].shape[0], vision[eposide_1].shape[1]),dtype=np.float32)
        valid_vision_list = np.empty((len_v, vision[eposide_1].shape[0], vision[eposide_1].shape[1]),dtype=np.float32)

        train_sarcasm = np.empty((len_t,1),np.int)
        valid_sarcasm = np.empty((len_v,1),np.int)

        train_humor = np.empty((len_t,1),np.int)
        valid_humor = np.empty((len_v,1),np.int)

        count_v = count_t = 0
        for  key in tqdm(keys):
            text_ele = text[key]
            audio_ele = audio[key]
            vision_ele = vision[key]

            if key.split('_')[0] == str(eposide):
                valid_text_list[count_v] = text_ele
                valid_audio_list[count_v] = audio_ele
                valid_vision_list[count_v]  = vision_ele
                valid_sarcasm[count_v] = int(json_file[key]['sarcasm'])
                valid_humor[count_v] = int(json_file[key]['humor'])
                count_v += 1

            else:
                train_text_list[count_t] = text_ele
                train_audio_list[count_t] = audio_ele
                train_vision_list[count_t]  = vision_ele
                train_sarcasm[count_t] = int(json_file[key]['sarcasm'])
                train_humor[count_t] = int(json_file[key]['humor'])
                count_t += 1


        combined_file['valid']['text'] = valid_text_list
        combined_file['valid']['vision'] = valid_vision_list
        combined_file['valid']['audio'] = valid_audio_list
        combined_file['valid']['sarcasm'] = valid_sarcasm
        combined_file['valid']['humor'] = valid_humor

        combined_file['train']['text'] = train_text_list
        combined_file['train']['vision'] = train_vision_list
        combined_file['train']['audio'] = train_audio_list
        combined_file['train']['sarcasm'] = train_sarcasm
        combined_file['train']['humor'] = train_humor

        joblib.dump(combined_file,'data/ten_fold/valid_'+ str(eposide),compress= 3)

        # MyLoader(args,combined_file)

    return combined_file




if __name__ == '__main__':
    ten_fold_split()
