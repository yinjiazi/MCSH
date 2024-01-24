from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers.fc import MLP
from .layers.layer_norm import LayerNorm


class doubel_pool(nn.Module):

    def __int__(self):
        super(doubel_pool, self).__int__()

        self.max_pool_t = nn.AdaptiveMaxPool1d(1)
        self.max_pool_a = nn.AdaptiveMaxPool1d(1)
        self.max_pool_v = nn.AdaptiveMaxPool1d(1)

        self.avg_pool_t = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_a = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_v = nn.AdaptiveAvgPool1d(1)

    def forward(self, t, a, v):
        max_t = self.max_pool_t(t).squeeze(-1)
        avg_t = self.avg_pool_t(t).squeeze(-1)
        merge_a = torch.cat([max_t, avg_t], dim=-1)


class subModel(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, pool_dim):
        super(subModel, self).__init__()

        # self.attention_ta = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        # self.attention_tv = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        # self.attention_at = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        # self.attention_av = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        # self.attention_vt = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        # self.attention_va = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)

        self.attention_ta = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        self.attention_tv = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        self.attention_at = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        self.attention_av = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        self.attention_vt = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        self.attention_va = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)

        self.max_pool_t = nn.AdaptiveMaxPool1d(pool_dim)
        self.max_pool_a = nn.AdaptiveMaxPool1d(pool_dim)
        self.max_pool_v = nn.AdaptiveMaxPool1d(pool_dim)

        self.avg_pool_t = nn.AdaptiveAvgPool1d(pool_dim)
        self.avg_pool_a = nn.AdaptiveAvgPool1d(pool_dim)
        self.avg_pool_v = nn.AdaptiveAvgPool1d(pool_dim)

        # self.max_pool_t = nn.AdaptiveMaxPool1d(1)
        # self.max_pool_a = nn.AdaptiveMaxPool1d(1)
        # self.max_pool_v = nn.AdaptiveMaxPool1d(1)
        #
        # self.avg_pool_t = nn.AdaptiveAvgPool1d(1)
        # self.avg_pool_a = nn.AdaptiveAvgPool1d(1)
        # self.avg_pool_v = nn.AdaptiveAvgPool1d(1)

        self.post_pool_t = nn.Linear(pool_dim * 2, pool_dim * 2)
        self.post_pool_a = nn.Linear(pool_dim * 2, pool_dim * 2)
        self.post_pool_v = nn.Linear(pool_dim * 2, pool_dim * 2)

    def forward(self, t, a, v):
        ta = self.attention_ta(t, a, a)
        tv = self.attention_tv(t, v, v)
        merge_t = torch.cat([ta, tv], dim=-1)
        merge_t_max = self.max_pool_t(merge_t)
        # .view(merge_t.size(0), -1)
        # print(merge_t_max.shape)
        merge_t_avg = self.avg_pool_t(merge_t)
        # print(merge_t_avg.shape)
        merge_t = torch.cat([merge_t_max, merge_t_avg], dim=-1)
        # merge_t = merge_t * g_t
        # print(merge_t.shape)
        at = self.attention_at(a, t, t)
        av = self.attention_av(a, v, v)

        merge_a = torch.cat([at, av], dim=-1)
        merge_a_max = self.max_pool_t(merge_a)
        merge_a_avg = self.avg_pool_t(merge_a)
        merge_a = torch.cat([merge_a_max, merge_a_avg], dim=-1)
        # merge_a = merge_a * g_a

        vt = self.attention_vt(v, t, t)
        va = self.attention_va(v, a, a)

        merge_v = torch.cat([vt, va], dim=-1)
        merge_v_max = self.max_pool_t(merge_v)
        merge_v_avg = self.avg_pool_t(merge_v)
        merge_v = torch.cat([merge_v_max, merge_v_avg], dim=-1)

        merge_t = F.dropout(F.relu(self.post_pool_t(merge_t), inplace=True), p=0.2)
        merge_a = F.dropout(F.relu(self.post_pool_a(merge_a), inplace=True), p=0.2)
        merge_v = F.dropout(F.relu(self.post_pool_v(merge_v), inplace=True), p=0.2)

        merge = torch.cat([merge_v, merge_a, merge_t], dim=-1)

        return merge


class subModel_global_pool(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v):
        super(subModel_global_pool, self).__init__()

        self.attention_ta = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_tv = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_at = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_av = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_vt = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_va = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)

        # self.attention_ta = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_tv = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_at = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_av = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_vt = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_va = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)

        self.max_pool_t = nn.AdaptiveMaxPool1d(1)
        self.max_pool_a = nn.AdaptiveMaxPool1d(1)
        self.max_pool_v = nn.AdaptiveMaxPool1d(1)

        self.avg_pool_t = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_a = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_v = nn.AdaptiveAvgPool1d(1)

        seq_len = 2 * 32
        self.post_pool_t = nn.Linear(seq_len, 64)
        self.post_pool_a = nn.Linear(seq_len, 64)
        self.post_pool_v = nn.Linear(seq_len, 64)

    def forward(self, t, a, v):
        ta = self.attention_ta(t, a, a)
        tv = self.attention_tv(t, v, v)
        merge_t = torch.cat([ta, tv], dim=-1)
        merge_t_max = self.max_pool_t(merge_t).view(merge_t.size(0), -1).contiguous()
        merge_t_avg = self.avg_pool_t(merge_t).view(merge_t.size(0), -1).contiguous()
        merge_t = torch.cat([merge_t_max, merge_t_avg], dim=-1)

        at = self.attention_at(a, t, t)
        av = self.attention_av(a, v, v)
        merge_a = torch.cat([at, av], dim=-1)
        merge_a_max = self.max_pool_t(merge_a).view(merge_t.size(0), -1).contiguous()
        merge_a_avg = self.avg_pool_t(merge_a).view(merge_t.size(0), -1).contiguous()
        merge_a = torch.cat([merge_a_max, merge_a_avg], dim=-1)

        vt = self.attention_vt(v, t, t)
        va = self.attention_va(v, a, a)
        merge_v = torch.cat([vt, va], dim=-1)
        merge_v_max = self.max_pool_t(merge_v).view(merge_t.size(0), -1).contiguous()
        merge_v_avg = self.avg_pool_t(merge_v).view(merge_t.size(0), -1).contiguous()
        merge_v = torch.cat([merge_v_max, merge_v_avg], dim=-1)

        merge_t = F.dropout(F.relu(self.post_pool_t(merge_t), inplace=True), p=0.2)
        merge_a = F.dropout(F.relu(self.post_pool_a(merge_a), inplace=True), p=0.2)
        merge_v = F.dropout(F.relu(self.post_pool_v(merge_v), inplace=True), p=0.2)

        merge = torch.cat([merge_v, merge_a, merge_t], dim=-1)

        return merge


class subModel_global_pool_v2(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v):
        super(subModel_global_pool_v2, self).__init__()
        #
        self.attention_ta = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_tv = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_at = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_av = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_vt = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)
        self.attention_va = GuidedAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, hidden_size=d_model)

        # self.attention_ta = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_tv = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_at = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_av = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_vt = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        # self.attention_va = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        pool_dim = 32
        self.max_pool_t = nn.AdaptiveMaxPool1d(pool_dim)
        self.max_pool_a = nn.AdaptiveMaxPool1d(pool_dim)
        self.max_pool_v = nn.AdaptiveMaxPool1d(pool_dim)

        self.avg_pool_t = nn.AdaptiveAvgPool1d(pool_dim)
        self.avg_pool_a = nn.AdaptiveAvgPool1d(pool_dim)
        self.avg_pool_v = nn.AdaptiveAvgPool1d(pool_dim)

        seq_len = 32 * 2
        self.post_pool_t = nn.Linear(seq_len, seq_len)
        self.post_pool_a = nn.Linear(seq_len, seq_len)
        self.post_pool_v = nn.Linear(seq_len, seq_len)

    def forward(self, t, a, v):
        ta = self.attention_ta(t, a, a)
        tv = self.attention_tv(t, v, v)
        merge_t = torch.cat([ta, tv], dim=-1)
        merge_t_max = self.max_pool_t(merge_t)
        # .view(merge_t.size(0), -1)
        merge_t_avg = self.avg_pool_t(merge_t)
        merge_t = torch.cat([merge_t_max, merge_t_avg], dim=-1)
        # print(merge_t.shape)
        # merge_t = merge_t * g_t

        # print(merge_t.shape)
        at = self.attention_at(a, t, t)
        av = self.attention_av(a, v, v)

        merge_a = torch.cat([at, av], dim=-1)
        merge_a_max = self.max_pool_t(merge_a)
        merge_a_avg = self.avg_pool_t(merge_a)
        merge_a = torch.cat([merge_a_max, merge_a_avg], dim=-1)
        # merge_a = merge_a * g_a

        vt = self.attention_vt(v, t, t)
        va = self.attention_va(v, a, a)

        merge_v = torch.cat([vt, va], dim=-1)
        merge_v_max = self.max_pool_t(merge_v)
        merge_v_avg = self.avg_pool_t(merge_v)
        merge_v = torch.cat([merge_v_max, merge_v_avg], dim=-1)

        merge_t = F.dropout(F.relu(self.post_pool_t(merge_t), inplace=True), p=0.2)
        merge_a = F.dropout(F.relu(self.post_pool_a(merge_a), inplace=True), p=0.2)
        merge_v = F.dropout(F.relu(self.post_pool_v(merge_v), inplace=True), p=0.2)

        return merge_t, merge_a, merge_v


class Model_v2(nn.Module):
    conv = False

    post_lstm = False

    def __init__(self, args=None):
        super(Model_v2, self).__init__()
        self.args = args
        # text_hidden, audio_hidden, video_hidden = args.hidden_dims
        # 'hidden_dims': (128, 16, 256)
        unified_seq_len = 32
        self.con_a = nn.Conv1d(1590, unified_seq_len, kernel_size=1, padding=0, bias=False, stride=1)
        self.con_v = nn.Conv1d(120, unified_seq_len, kernel_size=1, padding=0, bias=False, stride=1)
        # self.con_t = nn.Conv1d(text_in, text_out, kernel_size= 3, padding=0, bias=False, stride=1)

        unifined_dim = 256
        self.pool_dim = 32
        self.proj_a = nn.Linear(33, unifined_dim)
        self.proj_v = nn.Linear(709, unifined_dim)
        self.proj_t = nn.Linear(768, unifined_dim)

        self.proj_g_a = nn.Linear(33, 2 * self.pool_dim)
        self.proj_g_v = nn.Linear(709, 2 * self.pool_dim)
        self.proj_g_t = nn.Linear(768, 2 * self.pool_dim)

        n_head = 4
        d_model = unifined_dim
        d_inner = 512
        d_k = d_v = d_model // n_head
        r_drop = 0.2
        self.drop0 = 0.3
        n_layers = 4
        self.n_sub_model = 3
        self.subModel1 = subModel_global_pool(n_head, d_model, d_k, d_v)

        self.gate_t = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(32, 1, self.pool_dim)))
        self.gate_a = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(32, 1, self.pool_dim)))
        self.gate_v = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(32, 1, self.pool_dim)))

        self.proj_x_t = nn.Linear(unifined_dim * 3, unified_seq_len)
        self.proj_x_a = nn.Linear(unifined_dim * 3, unified_seq_len)
        self.proj_x_v = nn.Linear(unifined_dim * 3, unified_seq_len)
        # last concat output of all expert layer , and paded
        flatten_output = self.pool_dim * 2 * self.n_sub_model * unified_seq_len

        self.finalS = nn.Linear(64 * 3, 2)
        self.finalH = nn.Linear(64 * 3, 2)

    def forward(self, t, a, v):
        a = self.con_a(a)  # 1588 -> avg 28(1507, 3)
        v = self.con_v(v)  # 118

        a = self.proj_a(a)
        v = self.proj_v(v)
        t = self.proj_t(t)

        t = F.dropout(F.relu(t), p=self.drop0)
        a = F.dropout(F.relu(a), p=self.drop0)
        v = F.dropout(F.relu(v), p=self.drop0)

        # concat t a v to x for gating
        # last_dim_of_x = t.size(1) * 256 * 3 / 3
        x = torch.cat((t, a, v), dim=-1)

        # x_t = F.dropout(F.relu(self.proj_x_t(x)), p=self.drop0)
        # x_a = F.dropout(F.relu(self.proj_x_a(x)), p=self.drop0)
        # x_v = F.dropout(F.relu(self.proj_x_v(x)), p=self.drop0)

        x_t = self.proj_x_t(x)
        x_a = self.proj_x_a(x)
        x_v = self.proj_x_v(x)
        # print(x.shape)

        merge = self.subModel1(t, a, v)
        # print(self.gate_t.shape, x_t.shape)

        # g_t = torch.matmul(self.gate_t, x_t)
        # g_t = F.softmax(g_t, dim=-1)
        #
        # g_a = torch.matmul(self.gate_a, x_a)
        # g_a = F.softmax(g_a, dim=-1)
        #
        # g_v = torch.matmul(self.gate_v, x_v)
        # g_v = F.softmax(g_v, dim=-1)
        # # print(g_t.shape, merge_t.shape)
        #
        # merged_t = torch.matmul(g_t, merge_t).squeeze(1)
        # merged_a = torch.matmul(g_a, merge_a).squeeze(1)
        # merged_v = torch.matmul(g_v, merge_v).squeeze(1)

        # merged = torch.cat((merge_t, merge_a, merge_v), dim = -1).view(merge_t.size(0), -1).contiguous()
        # print(, merged.shape)
        output_fusion = {'sarcasm': self.finalS(merged), 'humor': self.finalH(merged)}
        res = {
            self.args.tasks: output_fusion,
        }
        return res


class Model(nn.Module):
    conv = False

    post_lstm = False

    def __init__(self, args=None):
        super(Model, self).__init__()
        self.args = args
        # text_hidden, audio_hidden, video_hidden = args.hidden_dims
        # 'hidden_dims': (128, 16, 256)
        unified_seq_len = 32
        self.con_a = nn.Conv1d(795, unified_seq_len, kernel_size=1, padding=0, bias=False, stride=1)
        self.con_v = nn.Conv1d(120, unified_seq_len, kernel_size=1, padding=0, bias=False, stride=1)
        self.con_t = nn.Conv1d(72, unified_seq_len, kernel_size=1, padding=0, bias=False, stride=1)

        unifined_dim = 256
        self.pool_dim = 32
        self.proj_a = nn.Linear(33, unifined_dim)
        self.proj_v = nn.Linear(709, unifined_dim)
        self.proj_t = nn.Linear(768, unifined_dim)

        self.proj_g_a = nn.Linear(33, 2 * self.pool_dim)
        self.proj_g_v = nn.Linear(709, 2 * self.pool_dim)
        self.proj_g_t = nn.Linear(768, 2 * self.pool_dim)

        n_head = 4
        d_model = unifined_dim
        d_inner = 512
        d_k = d_v = d_model // n_head
        r_drop = 0.2
        self.drop0 = 0.3
        n_layers = 4
        self.n_sub_model = 3
        self.subModel1 = subModel_global_pool(n_head, d_model, d_k, d_v)
        self.subModel2 = subModel_global_pool(n_head, d_model, d_k, d_v)
        self.subModel3 = subModel_global_pool(n_head, d_model, d_k, d_v)
        # self.subModel_gate1 = subModel(n_head, d_model, d_k, d_v, self.pool_dim)
        # self.subModel_gate2 = subModel(n_head, d_model, d_k, d_v, self.pool_dim)
        # gate_dim = self.pool_dim * 2
        # d_k = d_v = gate_dim // n_head
        # self.attention_tt = MultiHeadAttention(n_head=n_head, d_model=gate_dim, d_k=d_k, d_v=d_v)
        # self.attention_aa = MultiHeadAttention(n_head=n_head, d_model=gate_dim, d_k=d_k, d_v=d_v)
        # self.attention_vv = MultiHeadAttention(n_head=n_head, d_model=gate_dim, d_k=d_k, d_v=d_v)

        # n_layers, n_head, d_k, d_v,
        # d_model, d_inner,
        # self.attention_ta = Encoder(n_layers = n_layers , n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_inner = d_inner )
        # self.attention_tv = Encoder(n_layers = n_layers , n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_inner = d_inner )
        # self.attention_at = Encoder(n_layers = n_layers , n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_inner = d_inner )
        # self.attention_av = Encoder(n_layers = n_layers , n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_inner = d_inner )
        # self.attention_vt = Encoder(n_layers = n_layers , n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_inner = d_inner )
        # self.attention_va = Encoder(n_layers = n_layers , n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_inner = d_inner )

        # ta+ tv Dense  -> avg_pool max_pool - dense
        # self.lstm_t = nn.LSTM(768, hidden_size=unifined_dim, num_layers=1,  bidirectional=False,
        #                       batch_first=True)
        # self.lstm_a = nn.LSTM(33, hidden_size=unifined_dim, num_layers=1,  bidirectional=False,
        #                       batch_first=True)
        # self.lstm_v = nn.LSTM(709, hidden_size=unifined_dim, num_layers=1,  bidirectional=False,
        #                       batch_first=True)
        # self.gate_s = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(3, 32, unifined_dim , self.pool_dim * 6)))
        # self.gate_h = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(3, 32, unifined_dim , self.pool_dim * 6)))

        self.gate_s = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(32, 1, self.n_sub_model)))
        self.gate_h = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(32, 1, self.n_sub_model)))

        # last concat output of all expert layer , and paded
        flatten_output = self.pool_dim * 2 * self.n_sub_model * 3  # 576
        # flatten_output = unified_seq_len * self.pool_dim * 2 * self.n_sub_model # 6144

        self.finalS = nn.Linear(flatten_output, 2)
        self.finalH = nn.Linear(flatten_output, 2)

        self.proj_x_s = nn.Linear(256 * 32, self.n_sub_model)
        self.proj_x_h = nn.Linear(256 * 32, self.n_sub_model)

    def forward(self, t, a, v):
        # a = a.permute(0, 2, 1).contiguous()
        # v = v.permute(0, 2, 1).contiguous()
        # t = t.permute(0, 2, 1).contiguous()
        # 'input_lens': (32, 1590, 120),
        # 'feature_dims': (768, 33, 709),  # (text, audio, video)
        # all length - 2

        t = self.con_t(t)
        a = self.con_a(a)  # 1588 -> avg 28(1507, 3)
        v = self.con_v(v)  # 118

        # require_s = True
        # require_h = True
        # if label == 'sa':
        #     require_s = True
        #     require_h = False
        # elif label =='hu':
        #     require_s = False
        #     require_h = True
        # elif label == 'test':
        #     require_s = False
        #     require_h = False
        #
        # for p in self.proj_x_s.parameters():
        #     p.requires_grad = require_s
        #
        # for p in self.proj_x_h.parameters():
        #     p.requires_grad = require_h
        #
        # self.gate_s.requires_grad = require_s
        # self.gate_h.requires_grad = require_h
        #
        # for p in self.finalS.parameters():
        #     p.requires_grad = require_s
        # for p in self.finalH.parameters():
        #     p.requires_grad = require_h

        a = self.proj_a(a)
        v = self.proj_v(v)
        t = self.proj_t(t)

        t = F.dropout(F.relu(t), p=self.drop0)
        a = F.dropout(F.relu(a), p=self.drop0)
        v = F.dropout(F.relu(v), p=self.drop0)

        # concat t a v to x for gating
        # last_dim_of_x = t.size(1) * 256 * 3 / 3
        x = torch.stack((t, a, v), dim=1).view(t.size(0), self.n_sub_model, -1).contiguous()
        x_s = F.dropout(F.relu(self.proj_x_s(x)), p=self.drop0)

        x_h = F.dropout(F.relu(self.proj_x_h(x)), p=self.drop0)
        # x = torch.cat((t,a,v), dim=-1).view(t.size(0), self.n_sub_model, int(last_dim_of_x)).contiguous()

        # print(x.shape)
        merge1 = self.subModel1(t, a, v)
        merge2 = self.subModel2(t, a, v)
        merge3 = self.subModel3(t, a, v)

        # bs, 3 , d -> bs
        merged = torch.stack((merge1, merge2, merge3), dim=1).view(merge3.size(0), -1).contiguous()
        # print(x.shape, merged.shape)
        #
        # g_s = torch.matmul(self.gate_s, x_s)
        # g_s = F.softmax(g_s, dim=-1)
        # #
        # g_h = torch.matmul(self.gate_h, x_h)
        # g_h = F.softmax(g_h, dim=-1)
        # #
        # merged_s = torch.matmul(g_s, merged).squeeze(1)
        # merged_h = torch.matmul(g_h, merged).squeeze(1)

        # print(output_h.shape)

        # merged_s = torch.cat([merge1 * g_s,merge2 * g_s, merge3 * g_s], dim = -1)
        # merged_h = torch.cat([merge1 * g_h,merge2 * g_h, merge3 * g_h], dim = -1)
        # merged_s = merged * g_s
        # merged_h = merged * g_h

        # merged_s = merged_s.sum(0)
        # merged_h = merged_h.sum(0)

        # merged_s = merged_s.view(merged_s.size(0), -1).contiguous()
        # merged_h = merged_h.view(merged_h.size(0), -1).contiguous()

        output_fusion = {'sarcasm': self.finalS(merged), 'humor': self.finalH(merged)}
        #
        res = {
            self.args.tasks: output_fusion,
            'feature': [merge1, merge2, merge3]
        }
        return res


class GuidedAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, hidden_size, dropout=0.1, dropout_r=0.2, ):
        super(GuidedAttention, self).__init__()

        self.mhatt1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=0.2)
        self.mhatt2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=0.2)
        # self.ffn = FFN(args_MLP)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, q, k, v):
        x = self.norm1(q + self.dropout1(
            self.mhatt1(q, q, q)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(q, k, v)
        ))

        # x = self.norm3(x + self.dropout3(
        #     self.ffn(x)
        # ))

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.slf_attn = GuidedAttention(n_head, d_model, d_k, d_v, hidden_size=d_model)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.2, n_position=200):
        super().__init__()

        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        # self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask=None, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        enc_output = src_seq
        # enc_output = self.dropout(self.position_enc(src_seq))
        # enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output


class AttFlat(nn.Module):
    def __init__(self, args, flat_glimpse, merge=False):
        super(AttFlat, self).__init__()
        self.args = args
        self.merge = merge
        self.flat_glimpse = flat_glimpse
        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        if self.merge:
            self.linear_merge = nn.Linear(
                args.hidden_size * flat_glimpse,
                args.hidden_size * 2
            )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)

            return x_atted

        return torch.stack(att_list).transpose_(0, 1)


class Block(nn.Module):
    def __init__(self, args, i):
        super(Block, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa2 = SGA(args)
        self.sa3 = SGA(args)

        self.last = (i == args.layer - 1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.hidden_size)
            self.norm_a = LayerNorm(args.hidden_size)
            self.norm_v = LayerNorm(args.hidden_size)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask, z, z_mask):

        ax = self.sa1(x, x_mask)
        ay = self.sa2(y, x, y_mask, x_mask)
        az = self.sa3(z, x, z_mask, x_mask)

        x = ax + x
        y = ay + y
        z = az + z

        if self.last:
            return x, y, z

        ax = self.att_lang(x, x_mask)
        ay = self.att_audio(y, y_mask)
        az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y + self.dropout(ay)), \
               self.norm_v(z + self.dropout(az))


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, args):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)
