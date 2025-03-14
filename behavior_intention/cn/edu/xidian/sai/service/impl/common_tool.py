'''
Created on Jan 29, 2024

@author: 13507
'''

import numpy as np
from numpy import subtract
import sys
import traceback
#from scipy.stats import norm
#import pickle
#import dill
#import joblib
from  tensorflow.keras.models import save_model, load_model
from datetime import datetime
import re
import os
import pandas as pd

from behavior_intention.cn.edu.xidian.sai.dao.impl import spell_dao

# 将对象保存
def save(filename, cls):
    save_model(cls, filename)

# 加载对象
def load(filename):
    cls = load_model(filename)
    return cls


def time_elapsed(time_front, time_back, format="%H:%M:%S"):
    try:
        time_front_array = datetime.strptime(time_front, format)
        # print(time_front_array)
        time_back_array = datetime.strptime(time_back, format)
        #print(time_back_array)
        # time_elapsed = time_back_stamp - time_front_stamp
        time_elapsed = (time_back_array - time_front_array).seconds
        # if time_elapsed > 100:
        #     print(f"{time_front_array}  {time_back_array}  {time_elapsed}")
        return str(time_elapsed)
    except:
        return "0"

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(r"\\ +", r'\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    # regex = "(?P<Month>.*?) (?P<Date>.*?) (?P<Time>.*?) (?P<Type>.*?) (?P<Component>.*?): (?P<Content>.*?)"
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, regex, headers, logformat):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]

                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf

def get_content(log_file, logformat):
    # 从日志文件中分理出单纯的日志
    headers, regex = generate_logformat_regex(logformat)
    # print(headers) # ['Month', 'Date', 'Time', 'Type', 'Component', 'Content']
    # print(regex) # re.compile('^(?P<Month>.*?) (?P<Date>.*?) (?P<Time>.*?) (?P<Type>.*?) (?P<Component>.*?): (?P<Content>.*?)$')
    df_log = log_to_dataframe(log_file, regex, headers, logformat)
    return df_log

def spell_log(df_log, df_type="train"):
    spell_result_path = "tmpdata/SpellResult/spell.pkl"
    if os.path.isfile(spell_result_path):
        slm = load(spell_result_path)
        # 加载保存好的结果
    else:
        # 首先需要训练一遍，找出所有的日志健，保存在spell_result_path中
        # 要选取可以涵盖所有日志类型的数据用来训练
        slm = spell_dao.lcsmap('[\\s]+')
        for i in range(len(df_log)):
            log_message = df_log["Content"][i]
            # print(log_message)
            sub = log_message.strip('\n')
            slm.insert(sub)
        # 将spell的训练结果保存在这里
        save(spell_result_path, slm)

    # 对每条日志进行训练一遍，然后保存在spell_result.txt中
    templates = [0] * df_log.shape[0]
    ids = [0] * df_log.shape[0]
    ParameterList = [0] * df_log.shape[0]
    time_interval = [0] * df_log.shape[0]
    for i in range(len(df_log)):
        log_message = df_log["Content"][i].strip()
        obj = slm.insert(log_message)
        # seq = re.split('[\\s]+', log_message)
        # ParameterList[i] = obj.param(seq) # 取出log中的参数
        # if param != []:
        #     param = reduce(operator.add, param)  # 多维数组变一维数组
        obj_json = obj.tojson(log_message)
        templates[i] = obj_json["lcsseq"]  # 获取该日志条目的日志键
        ids[i] = obj_json["lcsseq_id"]  # 获取日志键id 也就是事件编号
        ParameterList[i] = obj_json["param"]  # 取出log中的参数

    # 生成两个日志时间差，加入param参数中
    # print(df_log.shape)
    # print(len(df_log))
    for id in range(len(df_log)):
        if id == 0:
            time_interval[id] = "0"
        else:
            time_last = df_log["Time"][id-1]
            time_now = df_log["Time"][id]
            elapsed = time_elapsed(time_last, time_now)
            time_interval[id] = elapsed
        ParameterList[id].append(time_interval[id])

    # 将结果保存在df_log里面
    df_log['EventId'] = ids # 事件向量
    df_log['EventTemplate'] = templates  # 日志模板 日志键
    df_log["ParameterList"] = ParameterList

    df_log.to_csv(f"tmpdata/struct/{df_type}_structured.csv", index=False)
    return df_log

def training_data_generate(params, n_steps):
    '''
    :param params: 一个日志键的所有日志条目参数组成的矩阵
    :param n_steps: lstm的历史窗口大下
    :return: X, Y
    '''
    # print("------------")
    # print(params)
    matrix = np.array(params)
    #print(matrix)
    X, Y = list(), list()
    for i in range(matrix.shape[0]):
        # 找到这个滑动窗口的最后一个下标
        end_ix = i + n_steps
        # check whether beyond the dataset
        if end_ix > matrix.shape[0]-1:
            break
        try:
            seq_x, seq_y = matrix[i:end_ix, :], matrix[end_ix,:]
            X.append(seq_x)
            Y.append(seq_y)
        except:
            traceback.print_exc()
            sys.exit()
        #     print(111111111111111111)
        #     print(i, end_ix)
        #     print(params)
        #     print(type(matrix))
        #     print(matrix.shape)
        #     print(222222222222222)
        #     continue
    # sys.exit()
    X, Y = np.array(X), np.array(Y)
    return X, Y

def mean_squared_error_modified(y_true, y_pred):
    ''' 修改后的mse以计算平方误差
    :param y_true: the test y --- array
    :param y_pred: the predict y --- array
    :return: the mean of errors, the errors list
    '''
    # 计算两个值之间的差
    d_matrix = subtract(y_true, y_pred)
    # print("the d_matrix is:", d_matrix)
    mses = []
    # define the sum of minus
    #sum_minus = 0
    # compute mse for every row
    for i in range(d_matrix.shape[0]):
        # 计算每一行的MSE
        mse = np.mean(d_matrix[i]**2)
        mses.append(mse)
    return mses

'''
判定一个列表是否完全包含在另一个列表之中
'''
def is_list_contained(list_a, list_b):
    return set(list_a).issubset(list_b)

'''
判定一个列表是否包含在一个二维列表的某个列表中
'''
def is_two_dimensional_list_contained(a_list, two_dimensional_list):
    two_dimensional_list = sorted(two_dimensional_list, key=len, reverse=True)
    for list_ele in two_dimensional_list:
        if is_list_contained(a_list, list_ele):
            return True
    return False
    
'''
正态分布
'''
def normal_distribution(sample_data):
    pass











