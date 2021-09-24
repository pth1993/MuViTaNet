import os
from torch.utils.data import DataLoader
import torch
from utils import EHRDataset, get_code_set, PadSequence, read_data_split, generate_prediction_file
from models import MuViTaNet
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm

SEED = 97

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

start_time = datetime.now()

parser = argparse.ArgumentParser(description='MuViTaNet Training')
parser.add_argument('--fold')
parser.add_argument('--batch_size')
parser.add_argument('--max_epoch')
parser.add_argument('--model_name')
parser.add_argument('--gpu')
parser.add_argument('--warm_start')
parser.add_argument('--inference')

args = parser.parse_args()

fold = args.fold
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
model_name = args.model_name
gpu = args.gpu
warm_start = True if args.warm_start == 'True' else False
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
inference = True if args.inference == 'True' else False

data_file_list = ['atrial_fibrillation.json', 'coronary_artery_disease.json', 'heart_failure.json', 'hypertension.json',
                  'peripheral_arterial_disease.json', 'stroke.json']

data_path = 'data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Use cuda: %s' % torch.cuda.is_available())
code_set = get_code_set(data_path + 'code_set.txt')

multi_data_loader = dict()
num_batch_list = []

for i, data_file in enumerate(data_file_list):
    print('Load %s' % data_file)
    train_data, dev_data, test_data, test_idx = read_data_split(data_path, data_file, 'data_split_index.txt', fold)
    train_data = EHRDataset(train_data, code_set)
    dev_data = EHRDataset(dev_data, code_set)
    test_data = EHRDataset(test_data, code_set)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence())
    dev_data_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=PadSequence())
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=PadSequence())
    multi_data_loader[i] = {'train_data_loader': train_data_loader, 'dev_data_loader': dev_data_loader,
                            'test_data_loader': test_data_loader, 'test_idx': test_idx}
    num_batch_list.append(len(train_data_loader))

batch_idx = []
for i, nb in enumerate(num_batch_list):
    batch_idx += [i] * nb
model = MuViTaNet(num_feature=len(code_set), demo_dim=8, encode_dim=32, rnn_hidden_dim=32, feature_hidden_dim=128,
                  decode_dim=32, dropout=0.1, n_task=len(data_file_list), device=device)
if inference:
    checkpoint = torch.load('saved_model/mt/%s_%s.ckpt' % (model_name, fold), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        auc_score_list = []
        prc_score_list = []
        for i, dt in enumerate(data_file_list):
            epoch_loss = 0
            predict_list = np.empty(0)
            label_list = np.empty(0)
            length_list = np.empty(0)
            for j, batch in enumerate(tqdm(multi_data_loader[i]['test_data_loader'])):
                feature = batch['feature'].to(device)
                demos = batch['demos'].to(device)
                feature_code = batch['feature_code'].to(device)
                length = batch['length'].to(device)
                code_length = batch['code_length'].to(device)
                time_interval = batch['time_interval'].to(device)
                label = batch['label'].to(device)
                predict, visit_attn_weight, feature_attn_weight = \
                    model(feature, demos, length, feature_code, code_length, time_interval, i)
                predict_list = np.concatenate((predict_list, predict.cpu().numpy()), axis=0)
                label_list = np.concatenate((label_list, label.cpu().numpy()), axis=0)
            auc_score = roc_auc_score(label_list, predict_list)
            prc_score = average_precision_score(label_list, predict_list)
            auc_score_list.append(auc_score)
            prc_score_list.append(prc_score)
        for i, dt in enumerate(data_file_list):
            print("AUC score (%s) on test set: %.4f" % (
                dt, auc_score_list[i]))
        for i, dt in enumerate(data_file_list):
            print("PRC score (%s) on test set: %.4f" % (
                dt, prc_score_list[i]))
else:
    if warm_start:
        checkpoint = torch.load('saved_model/mt/%s_%s.ckpt' % (model_name, fold), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    best_dev_auc = float("-inf")
    dev_auc_list = []
    test_auc_list = []
    dev_prc_list = []
    test_prc_list = []
    for epoch in range(max_epoch):
        model.train()
        epoch_loss = [0 for i in range(len(data_file_list))]
        predict_list = [np.empty(0) for i in range(len(data_file_list))]
        label_list = [np.empty(0) for i in range(len(data_file_list))]
        shuffle_batch_idx = np.random.permutation(batch_idx)
        multi_train_data_loader = dict()
        for k, v in multi_data_loader.items():
            multi_train_data_loader[k] = iter(multi_data_loader[k]['train_data_loader'])
        for idx in tqdm(shuffle_batch_idx):
            batch = next(multi_train_data_loader[idx])
            feature = batch['feature'].to(device)
            demos = batch['demos'].to(device)
            feature_code = batch['feature_code'].to(device)
            length = batch['length'].to(device)
            code_length = batch['code_length'].to(device)
            time_interval = batch['time_interval'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            predict, _, _ = model(feature, demos, length, feature_code, code_length, time_interval, idx)
            predict_list[idx] = np.concatenate((predict_list[idx], predict.detach().cpu().numpy()), axis=0)
            label_list[idx] = np.concatenate((label_list[idx], label.cpu().numpy()), axis=0)
            loss = model.loss(predict, label)
            loss.backward()
            optimizer.step()
            epoch_loss[idx] += loss.item()

        for i, dt in enumerate(data_file_list):
            auc_score = roc_auc_score(label_list[i], predict_list[i])
            prc_score = average_precision_score(label_list[i], predict_list[i])
            print('Train loss (%s): %.4f - Train AUC: % .4f - Train PRC: % .4f'
                  % (dt, epoch_loss[i] / num_batch_list[i], auc_score, prc_score))

        model.eval()
        with torch.no_grad():
            auc_score_list = []
            prc_score_list = []
            for i, dt in enumerate(data_file_list):
                epoch_loss = 0
                predict_list = np.empty(0)
                label_list = np.empty(0)
                for j, batch in enumerate(tqdm(multi_data_loader[i]['dev_data_loader'])):
                    feature = batch['feature'].to(device)
                    demos = batch['demos'].to(device)
                    feature_code = batch['feature_code'].to(device)
                    length = batch['length'].to(device)
                    code_length = batch['code_length'].to(device)
                    time_interval = batch['time_interval'].to(device)
                    label = batch['label'].to(device)
                    predict, _, _ = model(feature, demos, length, feature_code, code_length, time_interval, i)
                    predict_list = np.concatenate((predict_list, predict.cpu().numpy()), axis=0)
                    label_list = np.concatenate((label_list, label.cpu().numpy()), axis=0)
                    loss = model.loss(predict, label)
                    epoch_loss += loss.item()
                auc_score = roc_auc_score(label_list, predict_list)
                prc_score = average_precision_score(label_list, predict_list)
                print('Dev loss (%s): %.4f - Dev AUC: % .4f - Dev PRC: % .4f'
                      % (dt, epoch_loss / (j + 1), auc_score, prc_score))
                auc_score_list.append(auc_score)
                prc_score_list.append(prc_score)
            dev_auc_list.append(auc_score_list)
            dev_prc_list.append(prc_score_list)

            if best_dev_auc < np.mean(auc_score_list):
                best_dev_auc = np.mean(auc_score_list)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           'saved_model/mt/%s_%s.ckpt' % (model_name, fold))

        with torch.no_grad():
            auc_score_list = []
            prc_score_list = []
            for i, dt in enumerate(data_file_list):
                epoch_loss = 0
                predict_list = np.empty(0)
                label_list = np.empty(0)
                for j, batch in enumerate(tqdm(multi_data_loader[i]['test_data_loader'])):
                    feature = batch['feature'].to(device)
                    demos = batch['demos'].to(device)
                    feature_code = batch['feature_code'].to(device)
                    length = batch['length'].to(device)
                    code_length = batch['code_length'].to(device)
                    time_interval = batch['time_interval'].to(device)
                    label = batch['label'].to(device)
                    predict, _, _ = model(feature, demos, length, feature_code, code_length, time_interval, i)
                    predict_list = np.concatenate((predict_list, predict.cpu().numpy()), axis=0)
                    label_list = np.concatenate((label_list, label.cpu().numpy()), axis=0)
                    loss = model.loss(predict, label)
                    epoch_loss += loss.item()
                auc_score = roc_auc_score(label_list, predict_list)
                prc_score = average_precision_score(label_list, predict_list)
                print('Test loss (%s): %.4f - Test AUC: % .4f - Test PRC: % .4f'
                      % (dt, epoch_loss / (j + 1), auc_score, prc_score))
                auc_score_list.append(auc_score)
                prc_score_list.append(prc_score)
            test_auc_list.append(auc_score_list)
            test_prc_list.append(prc_score_list)

    best_dev_epoch = np.argmax(np.mean(dev_auc_list, axis=1))
    best_test_epoch = np.argmax(np.mean(test_auc_list, axis=1))

    end_time = datetime.now()

    print('Running time: %s' % (end_time - start_time))

    for i, dt in enumerate(data_file_list):
        print("AUC score (%s) on dev set by epoch %d (best dev epoch): %.4f" % (
            dt, best_dev_epoch + 1, dev_auc_list[best_dev_epoch][i]))
        print("AUC score (%s) on test set by epoch %d (best dev epoch): %.4f" % (
            dt, best_dev_epoch + 1, test_auc_list[best_dev_epoch][i]))
        print("AUC score (%s) on test set by epoch %d (best test epoch): %.4f" % (
            dt, best_test_epoch + 1, test_auc_list[best_test_epoch][i]))
        print("PRC score (%s) on dev set by epoch %d (best dev epoch): %.4f" % (
            dt, best_dev_epoch + 1, dev_prc_list[best_dev_epoch][i]))
        print("PRC score (%s) on test set by epoch %d (best dev epoch): %.4f" % (
            dt, best_dev_epoch + 1, test_prc_list[best_dev_epoch][i]))
        print("PRC score (%s) on test set by epoch %d (best test epoch): %.4f" % (
            dt, best_test_epoch + 1, test_prc_list[best_test_epoch][i]))
    print("Average AUC score on test set by epoch %d (best dev epoch): %.4f" % (
        best_dev_epoch + 1, np.mean(test_auc_list[best_dev_epoch])))
    print("Average AUC score on test set by epoch %d (best test epoch): %.4f" % (
        best_test_epoch + 1, np.mean(test_auc_list[best_test_epoch])))
    print("Average PRC score on test set by epoch %d (best dev epoch): %.4f" % (
        best_dev_epoch + 1, np.mean(test_prc_list[best_dev_epoch])))
    print("Average PRC score on test set by epoch %d (best test epoch): %.4f" % (
        best_test_epoch + 1, np.mean(test_prc_list[best_test_epoch])))
