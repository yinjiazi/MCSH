from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
import os
# import librosa
import numpy as np
import pickle


def getTextEmbedding(text):
    tokenizer_class = BertTokenizer
    model_class = BertModel
    # directory is fine
    pretrained_weights = '/root/MMSA/bert/bert-base-chinese/'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    # add_special_tokens will add start and end token
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
    return last_hidden_states.squeeze().numpy()


def getVideoEmbedding(csv_path, pool_size=5):
    df = pd.read_csv(csv_path)

    features, local_features = [], []
    for i in range(len(df)):
        local_features.append(np.array(df.loc[i][df.columns[5:]]))
        if (i + 1) % pool_size == 0:
            features.append(np.array(local_features).mean(axis=0))
            local_features = []
    if len(local_features) != 0:
        features.append(np.array(local_features).mean(axis=0))
    return np.array(features)


# def getAudioEmbedding(audio_path):
#     y, sr = librosa.load(audio_path)
#     # using librosa to get audio features (f0, mfcc, cqt)
#     hop_length = 512  # hop_length smaller, seq_len larger
#     f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T  # (seq_len, 1)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T  # (seq_len, 20)
#     cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T  # (seq_len, 12)
#
#     return np.concatenate([f0, mfcc, cqt], axis=-1)  # (seq_len, 33)


# padding
padding_mode = 'zeros'
padding_location = 'back'


def padding(feature, MAX_LEN):
    """
    mode:
        zero: padding with 0
        normal: padding with normal distribution
    location: front / back
    """
    assert padding_mode in ['zeros', 'normal']
    assert padding_location in ['front', 'back']

    length = feature.shape[0]
    if length >= MAX_LEN:
        return feature[:MAX_LEN, :]

    if padding_mode == "zeros":
        pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
    elif padding_mode == "normal":
        mean, std = feature.mean(), feature.std()
        pad = np.random.normal(mean, std, (MAX_LEN - length, feature.shape[1]))

    feature = np.concatenate([pad, feature], axis=0) if (padding_location == "front") else \
        np.concatenate((feature, pad), axis=0)
    return feature


def paddingSequence(sequences):
    feature_dim = sequences[0].shape[-1]
    lens = [s.shape[0] for s in sequences]
    # confirm length using (mean + std)
    final_length = int(np.mean(lens) + 3 * np.std(lens))
    # padding sequences to final_length
    final_sequence = np.zeros([len(sequences), final_length, feature_dim], dtype=np.float32)
    for i, s in enumerate(sequences):
        final_sequence[i] = padding(s, final_length)

    return final_sequence


def paddingSequenceV2(sequences):
    # feature_dim = sequences['1_002'].shape[-1]
    lens = [sequences[s].shape[0] for s in sequences]
    # confirm length using (mean + std)
    final_length = int(np.mean(lens) + 3 * np.std(lens))
    # padding sequences to final_length
    final_sequence = {}
    for key in sequences:
        final_sequence[key] = padding(sequences[key], final_length)

    return final_sequence


label_path = "/data1/yinjiazi/mmsa/data"
data_dir = "/data1/yinjiazi/mmsa/Processed"
df_label_T = pd.read_csv(os.path.join(label_path, 'label.csv'))
features_T, features_A, features_V = [], [], []
label = []
dict_T, dict_A, dict_V = {}, {}, {}
for i in tqdm(range(len(df_label_T))):
# for i in tqdm(range(1, 11)):
    video_id = df_label_T.loc[i, 'video_id']
    # text
    # embedding_T = getTextEmbedding(df_label_T.loc[i, 'sentence'])
    # dict_T[df_label_T.loc[i, 'video_id']] = embedding_T
    # features_T.append(embedding_T)

    # audio
    # audio_path = os.path.join(data_dir, 'audio', video_id + '.wav')
    # embedding_A = getAudioEmbedding(audio_path)
    # print(type(embedding_A))
    # print(embedding_A)
    # dict_A[df_label_T.loc[i, 'video_id']] = embedding_A
    # features_A.append(embedding_A)

    # video
    csv_path = os.path.join(data_dir, 'video/OpenFace2/Raw', video_id, video_id + '.csv')
    embedding_V = getVideoEmbedding(csv_path, pool_size=5)
    dict_V[df_label_T.loc[i, 'video_id']] = embedding_V
    # features_V.append(embedding_V)

    # label_sarcasm
    # label.append(df_label_T.loc[i, 'sarcasm'])

save_path = "/home/xjnu1/mmsa/feature/video"
with open('/home/xjnu1/mmsa/feature/video/video_op.p', 'wb') as f1:
    pickle.dump(dict_V, f1)
print('Features are saved in %s!' % save_path)
# padding
# feature_T = paddingSequence(features_T)
# dict_T_pad =  paddingSequenceV2(dict_T)
# dict_A_pad = paddingSequenceV2(dict_A)
dict_V_pad = paddingSequenceV2(dict_V)
with open('/home/xjnu1/mmsa/feature/video/video_op_pad.p', 'wb') as f:
    pickle.dump(dict_V_pad, f)
print('Features are saved in %s!' % save_path)
# save
# text_save_path = os.path.join(save_path, "text")
# audio_save_path = os.path.join(save_path, "audio")
# video_save_path = os.path.join(save_path, "video")
# with open('bert.p', 'wb') as f:
#     pickle.dump(dict_T_pad, f)
# print('Features are saved in %s!' % text_save_path)
# with open('audio_pad.p', 'wb') as f:
#     pickle.dump(dict_A_pad, f)
# print('Features are saved in %s!' % audio_save_path)

