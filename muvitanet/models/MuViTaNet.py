import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from .SupConLoss import SupConLoss


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class VisitAttention(nn.Module):
    def __init__(self, hidden_size, device):
        super(VisitAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_value_ori_func = nn.Linear(self.hidden_size, 1)
        self.device = device

    def forward(self, input_data):
        # # shape of input_data: <n_batch, n_seq, hidden_size>
        n_batch, n_seq, hidden_size = input_data.shape
        # # shape of attention_value_format: <n_batch, 1, n_seq>
        attention_value_format = torch.exp(self.attention_value_ori_func(input_data)).permute(0, 2, 1)
        # shape of ensemble flag format: <1, n_seq, n_seq>
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal=0).permute(1, 0).unsqueeze(0).to(self.device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-9
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value / accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output, attention_weight_format


class VisitTaskAttention(nn.Module):
    def __init__(self, hidden_size, n_task, device):
        super(VisitTaskAttention, self).__init__()
        self.attn = clones(VisitAttention(hidden_size, device), n_task)
        self.encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

    def forward(self, input_data, length_visit, task_idx):
        attn_data, attn_weight = self.attn[task_idx](input_data)
        input_encoded = self.encoder(input_data[torch.arange(len(length_visit)), length_visit - 1])
        last_attn = attn_data[torch.arange(len(length_visit)), length_visit - 1]
        output = torch.cat([input_encoded, last_attn], dim=-1)
        # attn_data = torch.cat([input_data, attn_data], dim=-1)
        # last_visit = attn_data[torch.arange(len(length_visit)), length_visit - 1]
        return output, attn_weight[:, length_visit-1, :]


class CodeAttention(nn.Module):
    def __init__(self, hidden_size, device):
        super(CodeAttention, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.attention_value_function = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
                                                      nn.Linear(hidden_size // 2, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask):
        # input = [(batch * num_visit) * num_code_per_visit * code_embed_dim]
        # mask = [(batch * num_visit) * num_code_per_visit * 1]
        w = self.attention_value_function(input)
        # w = [(batch * num_visit) * num_code_per_visit * 1]
        w = self.softmax(w.masked_fill(mask == 0, -1e9)).permute(0, 2, 1)
        # w = [(batch * num_visit) * 1 * num_code_per_visit]
        output = torch.matmul(w, input).squeeze(1)
        # output = [(batch * num_visit) * code_embed_dim]
        return output


class CodeEncoder(nn.Module):
    def __init__(self, num_code, code_embed_dim, device):
        super(CodeEncoder, self).__init__()
        self.code_embed = nn.Embedding(num_code, code_embed_dim, padding_idx=0)
        self.code_attn = CodeAttention(code_embed_dim, device)
        self.device = device

    def forward(self, input_code, length_code):
        # input_code = [batch * num_visit * num_code_per_visit]
        # length_code = [batch * num_visit]
        num_batch, num_visit, num_code_per_visit = input_code.shape
        length_code = length_code.reshape(num_batch * num_visit)
        # length_code = [(batch * num_visit)]
        mask = (torch.arange(num_code_per_visit)[None, :].to(self.device) < length_code[:, None]).unsqueeze(-1)
        # mask = [(batch * num_visit) * num_code_per_visit * 1]
        code_embed = self.code_embed(input_code).reshape(num_batch * num_visit, num_code_per_visit, -1)
        # code_embed = [(batch * num_visit) * num_code_per_visit * code_embed_dim]
        output = self.code_attn(code_embed, mask).reshape(num_batch, num_visit, -1)
        # output = [batch * num_visit * code_embed_dim]
        return output


class FeatureCNN(nn.Module):
    def __init__(self, input_dim, cnn_hidden_dim):
        super(FeatureCNN, self).__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_hidden_dim, 3)

    def forward(self, input, mask):
        # input = [batch * seq_len * 1]
        cnn_output = self.cnn(input.permute(0, 2, 1))
        # cnn_output = [batch * hidden_dim * (seq_len - 2)]
        output, _ = torch.max(cnn_output.masked_fill(mask == 0, -1e9), dim=-1)
        # output, _ = torch.max(cnn_output, dim=-1)
        # output = [batch * hidden_dim]
        return output


class VisitViewEncoder(nn.Module):
    def __init__(self, num_feature, demo_dim, encode_dim, rnn_hidden_dim, dropout, device):
        super(VisitViewEncoder, self).__init__()
        self.code_encoder = CodeEncoder(num_feature + 1, encode_dim, device)
        self.encoder = nn.Linear(encode_dim + demo_dim, encode_dim)
        self.rnn_model = nn.GRU(input_size=encode_dim, hidden_size=rnn_hidden_dim, dropout=dropout, bidirectional=True,
                                batch_first=True)
        self.time_embed = PositionalEmbedding(encode_dim, 2200, device)

    def forward(self, input_demo, input_code, length_visit, length_code, time_interval):
        # demos = [batch * num_visit * demo_dim]
        # length_visit = [batch]
        # input_code = [batch * num_visit * num_code_per_visit]
        # length_code = [batch * num_visit]
        input_embed = self.code_encoder(input_code, length_code)
        # [batch * num_visit * encode_dim]
        input_embed = self.encoder(torch.cat([input_embed, input_demo], dim=-1))
        # [batch * num_visit * encode_dim]

        time_embed = self.time_embed(time_interval)
        # input_embed = input_embed + time_embed

        # rnn_output, _ = self.rnn_model(cnn_output)
        packed_input = nn.utils.rnn.pack_padded_sequence(input_embed, length_visit.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        rnn_output, _ = self.rnn_model(packed_input)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        # [batch * num_step * (2 * rnn_hidden_dim)]
        return rnn_output


class FeatureViewEncoder(nn.Module):
    def __init__(self, num_feature, feature_hidden_dim):
        super(FeatureViewEncoder, self).__init__()
        self.feature_cnn = clones(FeatureCNN(input_dim=1, cnn_hidden_dim=feature_hidden_dim), num_feature)

    def forward(self, input_visit, visit_mask):
        num_batch, num_visit, num_code = input_visit.shape
        feature_attn_output = []
        for i in range(num_code):
            feature_attn_output.append(self.feature_cnn[i](input_visit[:, :, i].unsqueeze(-1), visit_mask).unsqueeze(1))
            # feature_rnn_output = list of [batch * 1 * feature_hidden_dim]
        feature_attn_output = torch.cat(feature_attn_output, dim=1)
        # feature_rnn_output = [batch * feature_dim * feature_hidden_dim]
        return feature_attn_output


class FeatureTaskAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_task):
        super(FeatureTaskAttention, self).__init__()
        self.task_attention = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False), nn.Tanh(),
                                            nn.Linear(hidden_dim, n_task, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask, task_idx):
        attn_weight = self.softmax(self.task_attention(input).masked_fill(mask == 0, -1e9))
        # attn_weight = [batch * num_feature * n_task]
        attn_output = torch.matmul(attn_weight.permute(0, 2, 1), input)
        # attn_output = [batch * n_task * (input_dim)]
        attn_output = torch.split(attn_output, 1, dim=1)[task_idx]
        # feature_task_attn = [batch * 1 * (4 * rnn_hidden_dim)]
        return attn_output.squeeze(1), attn_weight[:, :, task_idx]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float().to(device)

        position = torch.arange(0, max_len).float().unsqueeze(1).to(device)
        div_term = (torch.arange(0, d_model, 2).float().to(device) * -(math.log(100000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = nn.Embedding.from_pretrained(pe, freeze=True)

    def forward(self, time_interval):
        return self.pe(time_interval)


class MVMTEncoder(nn.Module):
    def __init__(self, num_feature, demo_dim, encode_dim, rnn_hidden_dim, feature_hidden_dim, dropout, n_task, device=None):
        super(MVMTEncoder, self).__init__()
        self.visit_view_encoder = VisitViewEncoder(num_feature, demo_dim, encode_dim, rnn_hidden_dim, dropout, device)
        self.feature_view_encoder = FeatureViewEncoder(num_feature, feature_hidden_dim)

        self.visit_task_attention = VisitTaskAttention(2 * rnn_hidden_dim, n_task, device)
        self.feature_task_attention = FeatureTaskAttention(4 * rnn_hidden_dim, rnn_hidden_dim, n_task)
        self.device = device

    def forward(self, input_visit, input_demo, length_visit, input_code, length_code, time_interval, task_idx):
        # input_visit = [batch * num_visit * num_code]
        # demos = [batch * num_visit * demo_dim]
        # length_visit = [batch]
        # input_code = [batch * num_visit * num_code_per_visit]
        # length_code = [batch * num_visit]
        num_batch, num_visit, num_code = input_visit.shape
        visit_mask = torch.arange(num_visit - 2)[None, :].to(self.device) < (length_visit[:, None] - 2).unsqueeze(1)
        feature_mask = (torch.sum(input_visit, dim=1) > 0).unsqueeze(-1)
        # feature_mask = [batch * num_code * 1]
        visit_view_input = self.visit_view_encoder(input_demo, input_code, length_visit, length_code, time_interval)
        # visit_view_input = [batch * num_visit * (2 * rnn_hidden_dim)]
        visit_task_attn, visit_attn_weight = self.visit_task_attention(visit_view_input, length_visit, task_idx)
        # visit_task_attn = [batch  * (4 * rnn_hidden_dim)]
        feature_view_input = self.feature_view_encoder(input_visit, visit_mask)
        # feature_view_input = [batch * num_code * rnn_hidden_dim]
        feature_task_attn, feature_attn_weight = self.feature_task_attention(feature_view_input, feature_mask, task_idx)
        # feature_task_attn = [batch  * (4 * rnn_hidden_dim)]
        output = [visit_task_attn, feature_task_attn, visit_attn_weight, feature_attn_weight]
        # 2 * [batch * (4 * rnn_hidden_dim)
        return output


class MuViTaNet(nn.Module):
    def __init__(self, num_feature, demo_dim, encode_dim, rnn_hidden_dim, feature_hidden_dim, decode_dim, dropout,
                 n_task, device=None):
        super(MuViTaNet, self).__init__()
        self.mvmtencoder = MVMTEncoder(num_feature, demo_dim, encode_dim, rnn_hidden_dim, feature_hidden_dim, dropout,
                                       n_task, device)
        self.decoder = nn.ModuleList(
            [nn.Sequential(nn.Linear(8 * rnn_hidden_dim, decode_dim), nn.Tanh(), nn.Linear(decode_dim, 1),
                           nn.Sigmoid()) for _ in range(n_task)])
        self.device = device

    def forward(self, input_visit, input_demo, length_visit, input_code, length_code, time_interval, task_idx):
        # input_visit = [batch * num_visit * num_code]
        # input_demo = [batch * num_visit * demo_dim]
        # length_visit = [batch]
        # input_code = [batch * num_visit * num_code_per_visit]
        # length_code = [batch * num_visit]
        # time_interval = [batch * num_visit]
        visit_task_attn, feature_task_attn, visit_attn_weight, feature_attn_weight = \
            self.mvmtencoder(input_visit, input_demo, length_visit, input_code, length_code, time_interval, task_idx)

        output = self.decoder[task_idx](torch.cat([visit_task_attn, feature_task_attn], dim=-1)).squeeze(1)
        # [batch]
        return output, visit_attn_weight, feature_attn_weight

    def loss(self, predict, label):
        loss = nn.BCELoss()
        output = loss(predict, label.float())
        return output


class MuViTaNetSS(nn.Module):
    def __init__(self, num_feature, demo_dim, encode_dim, rnn_hidden_dim, feature_hidden_dim, decode_dim, dropout,
                 n_task, device=None):
        super(MuViTaNetSS, self).__init__()
        self.mvmtencoder = MVMTEncoder(num_feature, demo_dim, encode_dim, rnn_hidden_dim, feature_hidden_dim, dropout,
                                       n_task, device)
        self.decoder = nn.ModuleList(
            [nn.Sequential(nn.Linear(8 * rnn_hidden_dim, decode_dim), nn.Tanh(), nn.Linear(decode_dim, 1),
                           nn.Sigmoid()) for _ in range(n_task)])
        self.head = nn.Linear(4 * rnn_hidden_dim, 4 * rnn_hidden_dim)
        self.supconloss = SupConLoss(temperature=0.07)
        self.n_task = n_task
        self.device = device

    def forward(self, input_visit, input_demo, length_visit, input_code, length_code, time_interval, task_idx):
        # input_visit = [batch * num_visit * num_code]
        # input_demo = [batch * num_visit * demo_dim]
        # length_visit = [batch]
        # input_code = [batch * num_visit * num_code_per_visit]
        # length_code = [batch * num_visit]
        # time_interval = [batch * num_visit]
        visit_task_attn, feature_task_attn, visit_attn_weight, feature_attn_weight = \
            self.mvmtencoder(input_visit, input_demo, length_visit, input_code, length_code, time_interval, task_idx)
        if task_idx == self.n_task - 1:
            output = torch.cat([F.normalize(self.head(visit_task_attn), dim=1).unsqueeze(1),
                                F.normalize(self.head(feature_task_attn), dim=1).unsqueeze(1)], dim=1)
        else:
            output = self.decoder[task_idx](torch.cat([visit_task_attn, feature_task_attn], dim=-1)).squeeze(1)
            # [batch]
        return output, visit_attn_weight, feature_attn_weight

    def loss(self, predict, label, task_idx):
        if task_idx == self.n_task - 1:
            output = self.supconloss(predict)
        else:
            loss = nn.BCELoss()
            output = loss(predict, label.float())
        return output
