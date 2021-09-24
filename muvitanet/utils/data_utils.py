import json
import numpy as np
from datetime import datetime


def read_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data


def read_data_split(data_path, data_file, index_file, fold):
    index_dict = read_data(data_path + index_file)
    data_idx = index_dict[data_file][fold]
    train_idx = data_idx['train_idx']
    dev_idx = data_idx['dev_idx']
    test_idx = data_idx['test_idx']
    data = read_data(data_path + data_file)
    train_data = []
    dev_data = []
    test_data = []
    for idx in train_idx:
        train_data.append(data[idx])
    for idx in dev_idx:
        dev_data.append(data[idx])
    for idx in test_idx:
        test_data.append(data[idx])
    return train_data, dev_data, test_data, test_idx


def read_data_cl_split(data_path, data_file, dev=True):
    data = read_data(data_path + data_file)
    data_idx = sorted(data.keys())
    train_idx = data_idx[:int(0.8*len(data_idx))]
    dev_idx = data_idx[int(0.8*len(data_idx)):]
    train_data = []
    dev_data = []
    for idx in train_idx:
        train_data.append(data[idx])
    for idx in dev_idx:
        dev_data.append(data[idx])
    if dev:
        return train_data, dev_data
    else:
        return train_data + dev_data



def get_code_set(input_file):
    with open(input_file, 'r') as f:
        code_set = f.readline().strip().split(',')
    return code_set


def create_non_sequence_data(data, code_set):
    code_feature = np.zeros((len(data), len(code_set)))
    age_feature = np.zeros((len(data), 3))
    region_feature = np.zeros((len(data), 5))
    label = np.zeros(len(data))
    for i, k in enumerate(sorted(data.keys())):
        for e in data[k]['ENC']:
            for c in e['dx']:
                code_feature[i][code_set.index(c)] += 1
        dob = int(data[k]['DOB'])
        region = int(data[k]['REGION'])
        if dob < 1959:
            age_feature[i][0] = 1
        elif dob < 1969:
            age_feature[i][1] = 1
        else:
            age_feature[i][2] = 1
        region_feature[i][region - 1] = 1
        label[i] = data[k]['LB']
    feature = np.concatenate((code_feature, age_feature, region_feature), axis=1)
    return feature, label


def convert_str_2_time(input_str):
    output = datetime.strptime(input_str, '%m/%d/%Y')
    return output


def convert_enc_2_feature(patient, code_set):
    num_encounter = len(patient)
    num_feature = len(code_set)
    time_gap = []
    time_interval = []
    prev_time = None
    end_time = convert_str_2_time(patient[-1]['date'])
    data = np.zeros((num_encounter, num_feature), dtype=np.float32)
    data_code = []
    for i, e in enumerate(patient):
        dc = []
        for c in e['dx']:
            data[i][code_set[c]] = 1
            dc.append(code_set[c] + 1)
        data_code.append(dc)
        curr_time = convert_str_2_time(e['date'])
        if prev_time is not None:
            time_gap.append((curr_time - prev_time).days)
        else:
            time_gap.append(0)
        time_interval.append((end_time - curr_time).days)
        prev_time = curr_time
    return data, time_gap, time_interval, data_code


def padding_batch_data(data_batch):
    max_num_encounter = max([len(i) for i in data_batch])
    data_batch_new = np.zeros((len(data_batch), max_num_encounter, data_batch[0].shape[1]), dtype=np.float32)
    for i in range(len(data_batch)):
        data_batch_new[i][:len(data_batch[i]), :] = data_batch[i]
    return data_batch_new


def create_sequence_data_demos(data, code_set):
    code_set = dict(zip(code_set, list(range(len(code_set)))))
    label = np.zeros(len(data))
    feature = []
    time_gap = []
    time_interval = []
    demos = []
    feature_code = []
    for i, dt in enumerate(data):
        label[i] = dt['LB']
        dob = int(dt['DOB'])
        region = int(dt['REGION'])
        age_feature = np.zeros(3)
        region_feature = np.zeros(5)
        if dob < 1959:
            age_feature[0] = 1
        elif dob < 1969:
            age_feature[1] = 1
        else:
            age_feature[2] = 1
        region_feature[region - 1] = 1
        label[i] = dt['LB']
        ft, tg, ti, ftc = convert_enc_2_feature(dt['ENC'], code_set)
        d = np.tile(np.concatenate((age_feature, region_feature), axis=-1), (len(dt['ENC']), 1))
        feature.append(ft)
        time_gap.append(tg)
        time_interval.append(ti)
        demos.append(d)
        feature_code.append(ftc)
    return np.array(feature), np.array(demos), time_gap, time_interval, label, np.array(feature_code)


def generate_prediction_file(idx, length, predict, label, output_file):
    with open(output_file, 'w') as f:
        for i, l, p, lb in zip(idx, length, predict, label):
            f.write('%s,%d,%.4f,%d\n' % (i, l,  p, lb))
