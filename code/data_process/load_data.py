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
from utils.functions import read_pickle, read_json_files
from tqdm import tqdm
import joblib

__all__ = ['MMDataLoader']


class MMDataset(Dataset):
    def __init__(self, args, index, mode='train', data=None):
        self.mode = mode
        self.index = index
        self.data = data
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'sims': self.__init_msaZH,
            'mcsh': self.__init_mcsh
        }
        DATA_MAP[args.datasetName](args)

    def __init_mosi(self, args):
        data = self.data
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.label = {
            args.tasks: {
                'sarcasm': data[self.mode]['sarcasm'].astype(np.float32),
                'humor': data[self.mode]['humor'].astype(np.float32)
            }
        }
        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['train']['vision']), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max == 0] = 1
            self.__normalize()

    def __init_msaZH(self, args):
        data = np.load(args.datapath)
        self.vision = data['feature_V'][self.index[self.mode]]
        self.audio = data['feature_A'][self.index[self.mode]]
        self.text = data['feature_T'][self.index[self.mode]]
        self.label = {
            'M': data['label_M'][self.index[self.mode]],
            'T': data['label_T'][self.index[self.mode]],
            'A': data['label_A'][self.index[self.mode]],
            'V': data['label_V'][self.index[self.mode]]
        }
        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['feature_V'][self.index['train']]), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max == 0] = 1
            self.__normalize()

    def __init_mcsh(self, args):
        data = self.data
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.label = {
            args.tasks: {
                'sarcasm': data[self.mode]['sarcasm'].astype(np.float32),
                'humor': data[self.mode]['humor'].astype(np.float32)
            }
        }
        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['train']['vision']), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max == 0] = 1
            self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))


    def __len__(self):
        # return len(self.labels)
        return len(self.index[self.mode])

    def get_seq_len(self):
        return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):

        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': {
                'sarcasm': torch.Tensor(self.label[self.args.tasks]['sarcasm'][index].reshape(-1)),
                'humor': torch.Tensor(self.label[self.args.tasks]['humor'][index].reshape(-1)),
            }
        }

        return sample


def MMDataLoader(args):
    test_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'test_index.csv'))).reshape(-1)
    train_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'train_index.csv'))).reshape(-1)
    val_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'val_index.csv'))).reshape(-1)

    print('Train Samples Num: {0}'.format(len(train_index)))
    print('Valid Samples Num: {0}'.format(len(val_index)))
    print('Test Samples Num: {0}'.format(len(test_index)))

    index = {
        'train': train_index,
        'valid': val_index,
        'test': test_index
    }

    datasets = {
        'train': MMDataset(args, index=index, mode='train'),
        'valid': MMDataset(args, index=index, mode='valid'),
        'test': MMDataset(args, index=index, mode='test')
    }

    if 'input_lens' in args.keys():
        args.input_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }

    return dataLoader


def get_splited_ten_fold_files(args):
    # path = '/home/xjnu1/mmsa/data/ten_fold/120valid_9_test_10'
    path = '/root/MMSA/data/emb/120valid_9_test_10'
    # for path in paths:
    file = joblib.load(path)
    yield MyLoader(args, file)


def ten_fold_split(args):
    # 读取各种特征文件
    import joblib
    text = read_pickle('feature_extractor/text/bert.p')
    audio = joblib.load('feature_extractor/audio/audio_pad.p')
    # vision = joblib.load('feature_extractor/video/video_pad.p')
    json_file = read_json_files('json_data_dir/*')

    # 定义多层字典
    from collections import defaultdict
    combined_file = defaultdict(dict)

    # 循环十集 十次交叉验证
    for eposide in range(1, 11):

        eposide_1 = '1_001'
        train_text_list = valid_text_list = np.empty((0, text[eposide_1].shape[0], text[eposide_1].shape[1]))
        train_audio_list = valid_audio_list = np.empty((0, audio[eposide_1].shape[0], audio[eposide_1].shape[1]))
        # train_vision_list = valid_vision_list = np.empty((0, vision[eposide_1].shape[0], vision[eposide_1].shape[1]))
        train_sarcasm = valid_sarcasm = np.empty((0, 1))
        train_humor = valid_humor = np.empty((0, 1))

        keys = json_file.keys()
        for key in tqdm(keys):
            if key.split('_')[0] == str(eposide):
                text_ele = text[key][None, ...]
                audio_ele = audio[key][None, ...]
                # vision_ele = vision[key][None,...]
                sarcasm_ele = [[int(json_file[key]['sarcasm'])]]
                humor_ele = [[int(json_file[key]['humor'])]]
                valid_text_list = np.concatenate([valid_text_list, text_ele], axis=0)
                valid_audio_list = np.concatenate([valid_audio_list, audio_ele], axis=0)
                # valid_vision_list = np.concatenate([valid_vision_list, vision_ele],axis = 0)
                valid_sarcasm = np.concatenate([valid_sarcasm, sarcasm_ele], axis=0)
                valid_humor = np.concatenate([valid_humor, humor_ele], axis=0)
            else:
                text_ele = text[key][None, ...]
                audio_ele = audio[key][None, ...]
                # vision_ele = vision[key][None, ...]
                sarcasm_ele = [[int(json_file[key]['sarcasm'])]]
                humor_ele = [[int(json_file[key]['humor'])]]
                train_text_list = np.concatenate([train_text_list, text_ele], axis=0)
                train_audio_list = np.concatenate([train_audio_list, audio_ele], axis=0)
                # valid_vision_list = np.concatenate([valid_vision_list, vision_ele],axis = 0)
                train_sarcasm = np.concatenate([train_sarcasm, sarcasm_ele], axis=0)
                train_humor = np.concatenate([train_humor, humor_ele], axis=0)

        combined_file['valid']['text'] = np.array(valid_text_list)
        # combined_file['valid']['vision'] = np.array(valid_vision_list)
        combined_file['valid']['audio'] = np.array(valid_audio_list)
        combined_file['valid']['sarcasm'] = np.array(valid_sarcasm)
        combined_file['valid']['humor'] = np.array(valid_humor)

        combined_file['train']['text'] = np.array(train_text_list)
        # combined_file['train']['vision'] = np.array(train_vision_list)
        combined_file['train']['audio'] = np.array(train_audio_list)
        combined_file['train']['sarcasm'] = np.array(train_sarcasm)
        combined_file['train']['humor'] = np.array(train_humor)

        joblib.dump(combined_file, 'data/ten_fold/valid_' + str(eposide), compress=3)

        # MyLoader(args,combined_file)

    return combined_file


def ten_fold_split_v2():
    # 读取各种特征文件

    text = read_pickle('feature_extractor/text/bert.p')
    audio = joblib.load('feature_extractor/audio/audio_pad.p')
    vision = pickle.load(open('feature_extractor/video/openface_pad.p', 'rb'))
    json_file = read_json_files('json_data_dir/*')

    # 定义多层字典
    from collections import defaultdict
    combined_file = defaultdict(dict)

    # 循环十集 十次交叉验证
    for eposide in range(1, 11):

        eposide_1 = '1_001'
        len_v = 0
        len_t = 0
        keys = json_file.keys()
        for key in keys:
            if key.split('_')[0] == str(eposide):
                len_v += 1
            else:
                len_t += 1
        # 得到valid 和 train的维度 （长度）

        train_text_list = np.empty((len_t, text[eposide_1].shape[0], text[eposide_1].shape[1]), dtype=np.float32)
        valid_text_list = np.empty((len_v, text[eposide_1].shape[0], text[eposide_1].shape[1]), dtype=np.float32)

        train_audio_list = np.empty((len_t, audio[eposide_1].shape[0], audio[eposide_1].shape[1]), dtype=np.float32)
        valid_audio_list = np.empty((len_v, audio[eposide_1].shape[0], audio[eposide_1].shape[1]), dtype=np.float32)

        train_vision_list = np.empty((len_t, vision[eposide_1].shape[0], vision[eposide_1].shape[1]))
        valid_vision_list = np.empty((len_v, vision[eposide_1].shape[0], vision[eposide_1].shape[1]))

        train_sarcasm = np.empty((len_t, 1), np.int)
        valid_sarcasm = np.empty((len_v, 1), np.int)

        train_humor = np.empty((len_t, 1), np.int)
        valid_humor = np.empty((len_v, 1), np.int)

        count_v = count_t = 0
        for key in tqdm(keys):
            if vision.has_key(key):
                vision_ele = vision[key]
                text_ele = text[key]
                audio_ele = audio[key]

                if key.split('_')[0] == str(eposide):
                    valid_text_list[count_v] = text_ele
                    valid_audio_list[count_v] = audio_ele
                    valid_vision_list[count_v] = vision_ele
                    valid_sarcasm[count_v] = int(json_file[key]['sarcasm'])
                    valid_humor[count_v] = int(json_file[key]['humor'])
                    count_v += 1

                else:
                    train_text_list[count_t] = text_ele
                    train_audio_list[count_t] = audio_ele
                    train_vision_list[count_t] = vision_ele
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

        joblib.dump(combined_file, 'data/ten_fold/valid_' + str(eposide), compress=3)

        # MyLoader(args,combined_file)

    return combined_file


def split_9asValid_10asTest():
    # 读取各种特征文件
    # /home/xjnu1/mmsa
    text = read_pickle('/home/xjnu1/mmsa/feature/text/bert.p')
    audio = joblib.load('/home/xjnu1/mmsa/feature/audio/audio_pad.p')
    vision = pickle.load(open('/home/xjnu1/mmsa/feature/video/video_pad.p', 'rb'))
    json_file = read_json_files('/home/xjnu1/mmsa/json_data_dir/*')

    # 定义多层字典
    from collections import defaultdict
    combined_file = defaultdict(dict)

    # 循环十集 十次交叉验证
    eposide_1 = '1_001'

    len_t = 0
    len_v = 0
    len_test = 0

    keys = json_file.keys()
    for key in keys:
        episode_prefix = key.split('_')[0]
        if episode_prefix == str(9):
            len_v += 1
        elif episode_prefix == str(10):
            len_test += 1
        else:
            len_t += 1

    # 得到valid 和 train的维度 （长度）

    train_text_list = np.empty((len_t, text[eposide_1].shape[0], text[eposide_1].shape[1]), dtype=np.float32)
    valid_text_list = np.empty((len_v, text[eposide_1].shape[0], text[eposide_1].shape[1]), dtype=np.float32)
    test_text_list = np.empty((len_test, text[eposide_1].shape[0], text[eposide_1].shape[1]), dtype=np.float32)

    train_audio_list = np.empty((len_t, audio[eposide_1].shape[0], audio[eposide_1].shape[1]), dtype=np.float32)
    valid_audio_list = np.empty((len_v, audio[eposide_1].shape[0], audio[eposide_1].shape[1]), dtype=np.float32)
    test_audio_list = np.empty((len_test, audio[eposide_1].shape[0], audio[eposide_1].shape[1]), dtype=np.float32)

    train_vision_list = np.empty((len_t, vision[eposide_1].shape[0], vision[eposide_1].shape[1]))
    valid_vision_list = np.empty((len_v, vision[eposide_1].shape[0], vision[eposide_1].shape[1]))
    test_vision_list = np.empty((len_test, vision[eposide_1].shape[0], vision[eposide_1].shape[1]))

    train_sarcasm = np.empty((len_t, 1), np.int32)
    valid_sarcasm = np.empty((len_v, 1), np.int32)
    test_sarcasm = np.empty((len_test, 1), np.int32)

    train_humor = np.empty((len_t, 1), np.int32)
    valid_humor = np.empty((len_v, 1), np.int32)
    test_humor = np.empty((len_test, 1), np.int32)

    count_v = count_t = count_test = 0
    for key in tqdm(keys):
        if key in vision and key in text and key in audio:
            vision_ele = vision[key]
            text_ele = text[key]
            audio_ele = audio[key]

            if key.split('_')[0] == str(9):
                valid_text_list[count_v] = text_ele
                valid_audio_list[count_v] = audio_ele
                valid_vision_list[count_v] = vision_ele
                valid_sarcasm[count_v] = int(json_file[key]['sarcasm'])
                valid_humor[count_v] = int(json_file[key]['humor'])
                count_v += 1
            elif key.split('_')[0] == str(10):
                test_text_list[count_test] = text_ele
                test_audio_list[count_test] = audio_ele
                test_vision_list[count_test] = vision_ele
                test_sarcasm[count_test] = int(json_file[key]['sarcasm'])
                test_humor[count_test] = int(json_file[key]['humor'])
                count_test += 1
            else:
                train_text_list[count_t] = text_ele
                train_audio_list[count_t] = audio_ele
                train_vision_list[count_t] = vision_ele
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

    combined_file['test']['text'] = test_text_list
    combined_file['test']['vision'] = test_vision_list
    combined_file['test']['audio'] = test_audio_list
    combined_file['test']['sarcasm'] = test_sarcasm
    combined_file['test']['humor'] = test_humor

    # /home/xjnu1/mmsa
    joblib.dump(combined_file, '/home/xjnu1/mmsa/data/ten_fold/120valid_9_test_10', compress=3)

    return combined_file


def MyLoader(args, combined_file):
    train_index = [i for i in range(len(combined_file['train']['sarcasm']))]
    val_index = [i for i in range(len(combined_file['valid']['sarcasm']))]
    test_index = [i for i in range(len(combined_file['test']['sarcasm']))]

    index = {
        'train': train_index,
        'valid': val_index,
        'test': test_index
    }

    datasets = {
        'train': MMDataset(args, index=index, mode='train', data=combined_file),
        'valid': MMDataset(args, index=index, mode='valid', data=combined_file),
        'test': MMDataset(args, index=index, mode='test', data=combined_file)
    }

    if 'input_lens' in args.keys():
        args.input_lens = datasets['train'].get_seq_len()

    dataLoader = {
        'train': DataLoader(datasets['train'],
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            drop_last=True),
        'valid': DataLoader(datasets['valid'],
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            drop_last=True),
        'test': DataLoader(datasets['test'],
                           batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=False,
                           drop_last=True)
    }

    return dataLoader


if __name__ == '__main__':
    split_9asValid_10asTest()
