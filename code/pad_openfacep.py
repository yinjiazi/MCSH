import pickle
import copy
import numpy as np

with open('/home/xjnu1/mmsa/feature/video/video_op_pad.p', 'rb') as f:
    data = pickle.load(f)

longest_seq = 0
seqlen = []
for key, value in data.items():
    len_dim = value[0]
    seqlen.append(len(value))
    if len(value) > longest_seq:
        longest_seq = len(value)

for L in [70, 120]:
    c120 = 0
    for i in seqlen:
        if i <= L:
            c120 += 1

    print(f'{str(L)} : {c120 / len(seqlen)}')
# 92 % seqlen shorter than 120. so choose 120 to pad and truncate


pad_data = np.zeros((120, len(len_dim)), dtype=np.float32)

video_paded = {}

for key, value in data.items():

    pad_v = copy.deepcopy(pad_data)

    L = len(value)
    if L <= 120:
        pad_v[:len(value), :] = value
    elif L > 120:
        pad_v = value[:120, :]
    video_paded[key] = pad_v

with open('/home/xjnu1/mmsa/feature/video/video_pad.p', 'wb') as f:
    pickle.dump(video_paded, f)
