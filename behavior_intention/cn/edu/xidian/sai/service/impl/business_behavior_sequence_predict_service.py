'''
Created on 2024年4月9日

@author: 13507
'''

import torch
import torch.nn as nn
import time
import argparse
from collections import OrderedDict
import pandas as pd

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import csv_to_simple, get_first_file, get_param_values_from_train_dao
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import get_max_behavior_encoding

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    #hdfs = set()
    hdfs, lis= [], []
    behavior_sequences, block_ids, log_ids=csv_to_simple(f'{get_parent_path()}/service/impl/tmpdata/SeqData/' + name)
    for line_bs, line_li in zip(behavior_sequences, log_ids):
        line_bs = list(map(lambda n: n - 1, map(int, line_bs.strip("[]").split(", "))))
        line_bs = line_bs + [-1] * (window_size + 1 - len(line_bs))
        line_li=line_li.strip("[]").split(", ")
        #hdfs.add(tuple(line_bs))
        hdfs.append(tuple(line_bs))
        lis.append(line_li)
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    # for line_bs, line_bi, line_li in zip(hdfs, block_ids, log_ids):
    #     print(f"这里---{line_bi}:{line_li}:{line_bs}")
    return hdfs, block_ids, lis

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def business_behavior_sequence_predict():
    # Hyperparameters
    num_classes = int(get_max_behavior_encoding())
    input_size = 1
    model_name = get_first_file(f'{get_parent_path()}/service/impl/tmpdata/BusinessModel/')
    model_path = f'{get_parent_path()}/service/impl/tmpdata/BusinessModel/{model_name}'
    window_size_saved = int(get_param_values_from_train_dao()['window_size'])
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=100, type=int)
    parser.add_argument('-window_size', default=window_size_saved, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    global window_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader, block_ids, log_ids = generate('log_entry_to_behavior_block_predict.csv')
    #test_abnormal_loader = generate('hdfs_test_abnormal')
    TP=0
    FP=0
    # Test the model 
    start_time = time.time()
    with torch.no_grad():
        abnormal_block_ids, abnormal_log_ids, abnormal_behavior_sub_sequences=[], [], []
        for  line_nl, line_bi, line_li in zip(test_normal_loader, block_ids, log_ids):
            for i in range(len(line_nl) - window_size):
                seq = line_nl[i:i + window_size]
                detect_position=i + window_size# 检测位置
                label = line_nl[detect_position]# 检测对象
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                #正例
                if label in predicted:
                    TP+=1
                else:# 假正例(负例)
                    abnormal_behavior_sub_sequence=[x for x in (int(x+1) for x in (seq.flatten().tolist()+label.flatten().tolist())) if x != 0]
                    logid=line_li[detect_position].strip("'") if detect_position<len(line_li) else -1
                    #print(f"{line_bi}: {logid}: {abnormal_behavior_sub_sequence}")
                    abnormal_block_ids.append(line_bi)
                    abnormal_log_ids.append(logid)
                    abnormal_behavior_sub_sequences.append(abnormal_behavior_sub_sequence)
                    FP+=1
        #根据日志块id进行合并
        abnormal_block_ids2=list(OrderedDict.fromkeys(abnormal_block_ids).keys())
        abnormal_log_ids3, abnormal_behavior_sub_sequences3=[], []
        abnormal_block_ids3=abnormal_block_ids2
        for abnormal_block_id in abnormal_block_ids2:
            abnormal_log_ids2, abnormal_behavior_sub_sequences2=[], []
            position=[i for i, x in enumerate(abnormal_block_ids) if x == abnormal_block_id]
            for element_position in position:
                abnormal_log_ids2.append(abnormal_log_ids[element_position])
                abnormal_behavior_sub_sequences2.append(abnormal_behavior_sub_sequences[element_position])
            abnormal_log_ids3.append(abnormal_log_ids2)
            abnormal_behavior_sub_sequences3.append(abnormal_behavior_sub_sequences2)
            #print(f"{abnormal_block_id}: {abnormal_log_ids2}: {abnormal_behavior_sub_sequences2}")
        # 存储异常数据
        df=pd.DataFrame({"abnormal_block_ids": abnormal_block_ids3, "abnormal_log_ids": abnormal_log_ids3, "abnormal_behavior_sub_sequences": abnormal_behavior_sub_sequences3})
        df.to_csv(f"{get_parent_path()}/service/impl/tmpdata/Anormaly/business_behavior_sequence_anormal.csv", index=False, encoding='utf-8')
        #print(df)
    #print(f"{len(abnormal_block_ids3)}, {len(abnormal_log_ids3)}, {len(abnormal_behavior_sub_sequences3)}")
    # with torch.no_grad():
    #     for line in test_abnormal_loader:
    #         for i in range(len(line) - window_size):
    #             seq = line[i:i + window_size]
    #             label = line[i + window_size]
    #             seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
    #             label = torch.tensor(label).view(-1).to(device)
    #             output = model(seq)
    #             predicted = torch.argsort(output, 1)[0][-num_candidates:]
    #             # 预测正确的个数
    #             if label not in predicted:
    #                 TP += 1
    #                 break
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    # FN = len(test_abnormal_loader) - TP# 剩余的负例个数
    P = 100 * TP / (TP + FP)# 准确率
    # R = 100 * TP / (TP + FN)# 召回率
    # F1 = 2 * P * R / (P + R)# F1衡量
    print("Precision: {:.3f}%".format(P))
    # print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
