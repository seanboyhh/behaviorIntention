'''
Created on 2024年9月19日

@author: 13507
'''

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from collections import OrderedDict
import pandas as pd

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import csv_to_simple, get_first_file, get_param_values_from_train_dao
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import get_max_behavior_encoding

# cpu和gpu配置
GPU = ['cuda', 'cuda:0', 'cuda:1', 'cuda:2']
device = torch.device(GPU[1])#if torch.cuda.is_available() else "cpu"

# 使用滑动窗口分割数据，训练使用
def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    behavior_sequences, _, _ = csv_to_simple(f'{get_parent_path()}/service/impl/tmpdata/' + name)
    for line in  behavior_sequences:
        num_sessions += 1
        line = tuple(map(lambda n: n - 1, map(int, line.strip("[]").split(", "))))
        for i in range(len(line) - window_size):
            inputs.append(line[i:i + window_size])
            outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset

# 预测使用
def generate1(name):
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

'''
预测下一个行为，并形成新的行为序列
'''
def new_behavior_sequence():
    # Hyperparameters
    num_classes = int(get_max_behavior_encoding())
    input_size = 1
    model_name = get_first_file(f'{get_parent_path()}/service/impl/tmpdata/SeqModel/')
    model_path = f'{get_parent_path()}/service/impl/tmpdata/SeqModel/{model_name}'
    window_size_saved = int(get_param_values_from_train_dao()['window_size'])
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=100, type=int)
    parser.add_argument('-window_size', default=window_size_saved, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)# 预测值的候选个数
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
    test_normal_loader, block_ids, _ = generate1('log_entry_to_behavior_block_predict.csv')
    #test_abnormal_loader = generate('hdfs_test_abnormal')
    TP=0
    FP=0
    # Test the model 
    start_time = time.time()
    with torch.no_grad():
        new_behavior_sequences, new_block_ids=[], []
        for line_nl, line_bi in zip(test_normal_loader, block_ids):
            sequence_len = len(line_nl)
            seq1 = line_nl[sequence_len - window_size:sequence_len]# 截取輸入x值
            seq2 = torch.tensor(seq1, dtype=torch.float).view(-1, window_size, input_size).to(device)
            output = model(seq2)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            predicted = predicted.tolist()
            new_seq = list(seq1)
            new_seq = [num + 1 for num in new_seq]
            new_seq.append(predicted[0])
            new_behavior_sequences.append(new_seq)
            new_block_ids.append(line_bi)   
        
        # 存储新行为数据
        df=pd.DataFrame({"block_id": new_block_ids, "new_behavior_sequence": new_behavior_sequences})
        df.to_csv(f"{get_parent_path()}/service/impl/tmpdata/NextBehaviorBasedData/new_behavior_sequence.csv", index=False, encoding='utf-8')

    del df

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time)) 
    print('Finished Predicting')

def next_based_behavior_sequence_train():
    # Hyperparameters
    num_classes = int(get_max_behavior_encoding()) #类别个数
    
    num_epochs =1000
    batch_size = 2048
    input_size = 1
    model_dir = get_parent_path()+'/service/impl/tmpdata/SeqModel'
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))
    window_size_saved = int(get_param_values_from_train_dao()['window_size'])
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=100, type=int) #神经网络隐藏层神经元个数
    parser.add_argument('-window_size', default=window_size_saved, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    global window_size# 全局变量
    window_size = args.window_size
    
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate('log_entry_to_behavior_block.csv')
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(log_dir=get_parent_path()+'/service/impl/tmpdata/SeqLog/' + log)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for _, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')


