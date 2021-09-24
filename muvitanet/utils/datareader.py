from .data_utils import *
from torch.utils.data import Dataset
import torch


class PadSequence:
    def __call__(self, batch):
        sequences = [x['feature'] for x in batch]
        sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        demos = [x['demos'] for x in batch]
        demos = [torch.tensor(s, dtype=torch.float32) for s in demos]
        demos_padded = torch.nn.utils.rnn.pad_sequence(demos, batch_first=True)

        time_gap = [x['time_gap'] for x in batch]
        time_gap = [torch.tensor(s, dtype=torch.float32) for s in time_gap]
        time_gap_padded = torch.nn.utils.rnn.pad_sequence(time_gap, batch_first=True)

        time_interval = [x['time_interval'] for x in batch]
        time_interval = [torch.tensor(s, dtype=torch.long) for s in time_interval]
        time_interval_padded = torch.nn.utils.rnn.pad_sequence(time_interval, batch_first=True)

        lengths = torch.LongTensor([len(x) for x in sequences])

        max_code_length = max([len(i) for x in batch for i in x['feature_code']])
        code_length = []
        code_sequences_padded = []
        for x in batch:
            clength = []
            sq = x['feature_code']
            seq_len = len(sq)
            a = np.zeros([seq_len, max_code_length])
            for i, s in enumerate(sq):
                clength.append(len(s))
                a[i, :len(s)] = s
            code_sequences_padded.append(torch.tensor(a, dtype=torch.int64))
            code_length.append(torch.LongTensor(clength).unsqueeze(-1))
        code_sequences_padded = torch.nn.utils.rnn.pad_sequence(code_sequences_padded, batch_first=True)
        code_length_padded = torch.nn.utils.rnn.pad_sequence(code_length, batch_first=True).squeeze(-1)

        labels = torch.tensor([x['label'] for x in batch], dtype=torch.float32)
        return {'feature': sequences_padded, 'demos': demos_padded, 'time_gap': time_gap_padded,
                'time_interval': time_interval_padded, 'length': lengths, 'label': labels,
                'feature_code': code_sequences_padded, 'code_length': code_length_padded}


class EHRDataset(Dataset):
    def __init__(self, data, code_set):
        self.feature, self.demos, self.time_gap, self.time_interval, self.label, self.feature_code = \
            create_sequence_data_demos(data, code_set)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = {'feature': self.feature[idx], 'demos': self.demos[idx], 'time_gap':  self.time_gap[idx],
                  'time_interval':  self.time_interval[idx], 'label': self.label[idx],
                  'feature_code': self.feature_code[idx]}
        return sample
