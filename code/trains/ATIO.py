"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from trains.singleTask import *
from trains.multiTask import *

__all__ = ['ATIO']


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'mult': MulT,
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'mef_lstm': MEF_LSTM,
            'matt': MATT,
            'misa': MISA,
            'mmim': MMIM,
            'self_mm': SELF_MM,
            'cubemlp': CubeMLP,
            'miat': MIAT,
            'attmi': ATTMI,
            'unimodal': Unimodal,
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.lower()](args)
