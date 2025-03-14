'''
Created on Jan 29, 2024

@author: 13507
'''
import gc
import os
import pandas as pd
import re
from itertools import chain
from behavior_intention.cn.edu.xidian.sai.service.impl.param_lstm_model_service import LSTM_model
from pandas import Series
from keras.preprocessing.text import Tokenizer
import numpy as np

from behavior_intention.cn.edu.xidian.sai.service.impl.common_tool import save, load
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.service.impl.common_tool import training_data_generate
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import clear_folder

def tokens_generate(key_para_dict):
    # 从所有的param中提取出单词表
    '''
    :param: key_para_dict: the format is {Exx:[textual parameter 1],[texual parameter 2],...}
    :return: tokens: all the word tokens in the parameter value vector column
    '''
    text = []
    new_key_para_dict = {}
    for key, value in key_para_dict.items():
        para1 = []
        # extract the time part from values
        # value是对于一个日志键的所有日志的参数向量组成的列表
        for param in value:  # ['rhost=218.188.2.4', '0'], ['3', '8', '9', '3', '7']
            # print(f"value{value}")
            # print(f"param: {param}") [['rhost=220-135-151-1.hinet-ip.hinet.net', 'user=root'], '38937']
            # sys.exit()
            para2 = []
            # param是对于每个日志的参数向量列表, i是该日志的参数1，2，3，4
            for i in param:
                i = re.sub('=|\/|#|:|\[|\]|\'|\s+|\.|\-|\(|\)|rhost=|,', '', str(i))
                text.append(i)
                para2.append(i)
            para1.append(para2)
        new_key_para_dict[key] = para1

    return new_key_para_dict

def token_dict(new_key_para_dict):
    # 将token单词转化为数字
    token_encode_dict = {}
    '''
    :param new_key_para_dict: 键为日志键，值为参数列表
    :return: token_encode_dict: the format is ['fawjeiajet';[32,45,65,..],...]
    '''
    # doc : https://keras.io/zh/preprocessing/text/
    # build the dict about different value
    for key, value in new_key_para_dict.items():
        tokenizer = Tokenizer()
        tokens = list(chain(*value))
        tokenizer.fit_on_texts(tokens)
        encoded_texts = tokenizer.texts_to_sequences(tokens)
        # build the dict with tokens --> encoded_texts
        token_encode_dict_tmp = {}
        for token, encoded_text in zip(tokens, encoded_texts):
            token_encode_dict_tmp[token] = encoded_text
        # 去除其中为[]的，将其转化为[0]
        for k in token_encode_dict_tmp.keys():
            if token_encode_dict_tmp[k] == []:
                token_encode_dict_tmp[k] = [0]
        token_encode_dict[key] = token_encode_dict_tmp

    return token_encode_dict

def map_vectors(token_encode_dict, logkey_dict):
    logkey_dict_num = {}
    for key, param in logkey_dict.items(): # param 为一个日志键的所有日志条目的参数组成的列表
        p_tmp = []
        for p in param:  # p为一个日志条目中的参数向量，p为数组
            p = [token_encode_dict[key][i] for i in p]
            p_tmp.append(p)
        logkey_dict_num[key] = p_tmp
    return logkey_dict_num

# define the module to transform str into matrix
# the string is like: '10635,[21, 85, 16, 18],[21, 85, 16, 18, 307, 308, 1],[356],[424],[207]'
def save_log_para_array(dict, df_type):
    new_dict = {}
    for eventID, lists_raw in dict.items():
        new_dict[eventID] = []
        # print(f"eventID:{eventID},list_raw: {lists_raw}")
        numy = len(lists_raw[0])
        list_array = np.empty(shape=[0, numy])
        for param in lists_raw:  # param : [[4],[5]] 一条日志的参数向量
            lists = list(chain.from_iterable(param)) # 将多维列表合并为一位列表 [4, 5]
            new_dict[eventID].append(lists)
            try:
                list_array = np.append(list_array, [lists], axis=0)
            except Exception as e:
                # print("there is an error like:", e)
                pass
        # print(f"eventID:{eventID},list_raw: {new_dict[eventID]}")
        filename = f"./tmpdata/EventNpy/{df_type}_{eventID}.npy"
        np.save(filename, list_array)
    return new_dict

def param_value_train(train_data):
    # 加载train数据和test数据
    #key_para_dict_train, logkey_lineid_dict_train, key_para_dict_test, logkey_lineid_dict_test = get_para_dict(df_train_log, df_test_log)

    # 训练之前删除已存在的模型文件
    clear_folder(f"{get_parent_path()}/service/impl/tmpdata/ParamModel")

    # 对train数据进行训练
    model_generate(train_data)

def get_para_dict(df_train_log, df_test_log):

    # 首先遍历所有的logkey，找出logkey的种类
    logkey_list = list(set([EventId for EventId in df_train_log["EventId"]]).union(set([EventId for EventId in df_test_log["EventId"]])))

    # 初始化一些字典，实际上是从df_log中提取出来的
    logkey_param_dict_train = {}  # 键为log_key, 值为参数数组
    logkey_content_dict_train = {}  # 键为log_key，值为log Content组成的的数组
    logkey_lineid_dict_train = {}  # 键为log_key，值为df_type中的ids，用来异常溯源
    logkey_param_dict_test = {}  # 键为log_key, 值为参数数组
    logkey_content_dict_test = {}  # 键为log_key，值为log Content组成的的数组
    logkey_lineid_dict_test = {}  # 键为log_key，值为df_type中的ids，用来异常溯源

    logkey_param_dict = {}  # 将train和test里面的参数合并，键为log key，值为参数列表
    # 对字典进行初始化
    for key in logkey_list:
        logkey_param_dict_train[key] = []
        logkey_content_dict_train[key] = []
        logkey_lineid_dict_train[key] = []
        logkey_param_dict_test[key] = []
        logkey_content_dict_test[key] = []
        logkey_lineid_dict_test[key] = []
        logkey_param_dict[key] = []


    # 遍历df_train_log，将需要的数据依次添加到上述字典中
    for id in range(len(df_train_log)):
        log_key_tmp = df_train_log["EventId"][id]
        logkey_param_dict[log_key_tmp].append(df_train_log["ParameterList"][id])
        logkey_param_dict_train[log_key_tmp].append(df_train_log["ParameterList"][id])
        logkey_content_dict_train[log_key_tmp].append(df_train_log["Content"][id])
        logkey_lineid_dict_train[log_key_tmp].append(df_train_log["LineId"][id])

    # 遍历df_test_log，将需要的数据依次添加到上述字典中
    for id in range(len(df_test_log)):
        log_key_tmp = df_test_log["EventId"][id]
        logkey_param_dict[log_key_tmp].append(df_test_log["ParameterList"][id])
        logkey_param_dict_test[log_key_tmp].append(df_test_log["ParameterList"][id])
        logkey_content_dict_test[log_key_tmp].append(df_test_log["Content"][id])
        logkey_lineid_dict_test[log_key_tmp].append(df_test_log["LineId"][id])

    # 对参数字符进行一定处理，去除一些符号
    new_key_para_dict = tokens_generate(logkey_param_dict)
    new_key_para_dict_train = tokens_generate(logkey_param_dict_train)
    new_key_para_dict_test = tokens_generate(logkey_param_dict_test)

    # 建立一个字典，字典的键为日志键，值为一个字典（键为字符串，值为数字）
    token_encode_dict = token_dict(new_key_para_dict)

    # logkey_param_dict_train和logkey_param_dict_test
    # num_key_para_dict_train 为字典，键为logkey，值为该logkey对应日志条目的数字形式参数列表
    num_key_para_dict_train = map_vectors(token_encode_dict, new_key_para_dict_train)

    num_key_para_dict_test = map_vectors(token_encode_dict, new_key_para_dict_test)

    #  将一个日志条目中的多维列表合并为一维，并以矩阵形式保存在npy文件中
    num_key_para_dict_train = save_log_para_array(num_key_para_dict_train, df_type='train')
    num_key_para_dict_test = save_log_para_array(num_key_para_dict_test, df_type='train')



    # 以下没什么作用，只是用来保存上述字典
    # logkey_dict 键为eventID，值为log param组成的数组
    df_dict_para = pd.DataFrame(dict([(k, Series(v)) for k, v in logkey_param_dict_train.items()]))
    df_dict_para.to_csv(f"./tmpdata/ParamData/train_param.csv", index=False)
    df_dict_para = pd.DataFrame(dict([(k, Series(v)) for k, v in logkey_param_dict_test.items()]))
    df_dict_para.to_csv(f"./tmpdata/ParamData/test_param.csv", index=False)

    # logkey_content_dict 键为eventID，值为log Content条目组成的数组
    df_dict_para = pd.DataFrame(dict([(k, Series(v)) for k, v in logkey_content_dict_train.items()]))
    df_dict_para.to_csv(f"./tmpdata/ParamData/train_Content.csv", index=False)
    df_dict_para = pd.DataFrame(dict([(k, Series(v)) for k, v in logkey_content_dict_test.items()]))
    df_dict_para.to_csv(f"./tmpdata/ParamData/test_Content.csv", index=False)


    # num_key_para_dict 键为eventID，值为log param数字形式
    df_dict_para = pd.DataFrame(dict([(k, Series(v)) for k, v in num_key_para_dict_train.items()]))
    df_dict_para.to_csv(f"./tmpdata/ParamData/train_param_num.csv", index=False)
    df_dict_para = pd.DataFrame(dict([(k, Series(v)) for k, v in num_key_para_dict_test.items()]))
    df_dict_para.to_csv(f"./tmpdata/ParamData/test_param_num.csv", index=False)
    # print(logkey_lineid_dict_train)
    return num_key_para_dict_train, logkey_lineid_dict_train, num_key_para_dict_test, logkey_lineid_dict_test

# 对每个日志键组成的日志参数向量来lstm训练，将结果保存在tmpdata/ParamModel文件夹中
def model_generate(key_para_dict):

    # 对每个日志键进行训练，模型放在model_dict中，eventID为键，model为值
    model_dict = {}
    # 依次加载每个日志键的日志参数组成的矩阵，依次检测
    for behavior_id, params in key_para_dict.items():
        model_file = f"{get_parent_path()}/service/impl/tmpdata/ParamModel/{behavior_id}.h5"
        # 跳跃式生成
        if os.path.exists(model_file):
            model_dict[behavior_id] = load(model_file)
            continue

        # 如果这个日志键对应的日志条目少于8个，则跳过对该日志键的模型生成
        if len(params) <= 8:
            continue
        else:
            # 设置滑动窗口大小为3
            n_steps = 5
            X, Y = training_data_generate(params, n_steps)
            model = LSTM_model(X, Y)
            model_dict[behavior_id] = model
            save(model_file, model)
            # yhat = model.predict(test_x)
            # print("the predicted y shapeis:", yhat.shape)  # (4, 2)
            # print("the test y shape is:", test_y.shape)  # (4, 2)
            # # 测量实际值和预测值的均方误差
            # mses = mean_squared_error_modified(test_y, yhat)
            # print(f"mses: {mses}")
            # sys.exit()
    del model_dict
    gc.collect()
    # return model_dict
