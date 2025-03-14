'''
Created on Nov 10, 2023

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

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import csv_to_simple, get_param_values_from_train_dao
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import get_max_behavior_encoding

# cpu和gpu配置
GPU = ['cuda', 'cuda:0', 'cuda:1', 'cuda:2']
device = torch.device(GPU[1]) #if torch.cuda.is_available() else "cpu"

##使用滑动窗口分割数据
def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    behavior_sequences, _, _ = csv_to_simple(f'{get_parent_path()}/service/impl/tmpdata/SeqData/' + name)
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

def behavior_sequence_train():
    # Hyperparameters
    num_classes = int(get_max_behavior_encoding()) #类别个数
    num_epochs =200
    batch_size = 2048
    input_size = 1
    window_size_saved = int(get_param_values_from_train_dao()['window_size'])
    model_dir = get_parent_path()+'/service/impl/tmpdata/SeqModel'
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))
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
    seq_dataset = generate('log_entry_to_behavior_block_train.csv')
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
        try:
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
        except Exception as e:
            print(f'异常：{e}')

        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    del model
    del seq_dataset
    print('Finished Training')
