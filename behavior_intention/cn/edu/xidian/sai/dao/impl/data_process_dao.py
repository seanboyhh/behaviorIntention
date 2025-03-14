'''
Created on Jan 5, 2024
Modified on Dec 24, 2024

@author: 13507
'''
import math
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
import chardet
import codecs
import csv
import shutil

from pandas.core.interchange.dataframe_protocol import DataFrame

from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import csv_or_json_data
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path

# 对指定字段进行onehot向量化
def feature_vectorize_onehot(require_vectorize_field, input_file_path, out_file_path):
    pass


# 把时间转换成时间戳
def to_timestamp(original_time):
    formated_time = datetime.strptime(original_time, '%Y-%m-%d %H:%M:%S')
    return int(formated_time.timestamp())

'''
对日志进行划分, 可以依据一个字段进行一次划分，也可以依据两个字段进行两次划分。若进行两次划分，那先按时间划分再按其他字段划分。
field_by: 依据划分的字段, 可以是两个字段。若为两个字段，第一个字段为时间字段。此参数与参数mark一一对应着
mark: 取值time，按照时间进行划分。取值none_time,按照非时间字段划分。若field_by取值两个字段，此参数取值time.
id_name: 日志中的日志唯一id
behavior_field: 日志中的行为字段
'''


def log_partition(field_by, logid_field, behavior_field, time_delta ,train_or_predict_mark, mark = "time"):
    if train_or_predict_mark == 't':
        input_file_path = f'{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_train.csv'
        out_file_path = f"{get_parent_path()}/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_train.csv"
    elif train_or_predict_mark == 'p':
        input_file_path = f'{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_predict.csv'
        out_file_path = f"{get_parent_path()}/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_predict.csv"
    df = pd.read_csv(input_file_path)

    # 第一次划分
    block_ids = []  # 日志块id
    log_id_sequences = []  # 日志id序列
    behavior_sequences = []  # 行为序列
    field_by_sequences = []  # 第二分割字段值组成的序列
    bid = 1
    block_id = "block" + str(bid)
    log_id_sequence = []
    behavior_sequence = []
    field_by_sequence = []

    if mark == "time":
        refer_time = pd.to_datetime(df.iloc[0][field_by[0]])  # 参照时间，初始值为第一行的值
        for _, row in df.iterrows():
            current_datetime = pd.to_datetime(row[field_by[0]])
            if ((current_datetime - pd.Timedelta(minutes=time_delta)) <= refer_time) and (current_datetime >= (refer_time - pd.Timedelta(minutes=time_delta))):
                field_by_sequence.append(row[field_by[1]])
                log_id_sequence.append(row[logid_field])
                behavior_sequence.append(row[behavior_field])
                # print(f"{row[id_name]}: {row[behavior_field]}")
            else:
                block_ids.append(block_id)
                field_by_sequences.append(field_by_sequence)
                log_id_sequences.append(log_id_sequence)
                behavior_sequences.append(behavior_sequence)
                bid += 1
                block_id = "block" + str(bid)
                field_by_sequence = []
                log_id_sequence = []
                behavior_sequence = []
                refer_time = current_datetime
    elif mark == "none_time":
        log_id_sequence = df[logid_field]
        behavior_sequence = df[behavior_field]
        field_by_sequence = df[field_by[0]]
        block_ids.append(block_id)
        log_id_sequences.append(log_id_sequence)
        behavior_sequences.append(behavior_sequence)
        field_by_sequences.append(field_by_sequence)

    # print(field_by_sequences)

    if mark == "time" and len(field_by) == 1:
        data = {"block_id": block_ids,
            "log_id_sequence": log_id_sequences,
            "behavior_sequence": behavior_sequences}
    else:
        if len(field_by) == 1:
            sub = 0
        else:
            sub = 1
        data = {"block_id": block_ids,
            field_by[sub]: field_by_sequences,
            "log_id_sequence": log_id_sequences,
            "behavior_sequence": behavior_sequences}
    df2 = pd.DataFrame(data)
    # print(df2.iloc[0][field_by[0]])

    # 在第一次按照时间划分的基础上进行二次划分
    if len(field_by) == 2 or mark == "none_time":
        block_ids = []  # 日志块id
        log_id_sequences = []  # 日志id序列
        behavior_sequences = []  # 行为序列
        bid = 1
        block_id = "block" + str(bid)
        log_id_sequence = []
        behavior_sequence = []
        field_by_sequence = []
        if len(field_by) == 1:
            sub = 0
        else:
            sub = 1
        for _, row in df2.iterrows():
            field_by_sequence_df2 = remove_duplicates(row[field_by[sub]])
            # print(field_by_sequence_df2)
            log_id_sequence_df2 = list(row["log_id_sequence"])
            behavior_sequence_df2 = list(row["behavior_sequence"])
            #print(f'{log_id_sequence_df2}:{behavior_sequence_df2}')
            for field_by_value in field_by_sequence_df2:
                index_del = 0
                for log_id, behavior in zip(log_id_sequence_df2, behavior_sequence_df2):
                    if df[df[logid_field] == log_id].iloc[0][field_by[sub]] == field_by_value:
                        log_id_sequence.append(log_id)
                        behavior_sequence.append(behavior)
                        del log_id_sequence_df2[index_del], behavior_sequence_df2[index_del]
                    index_del +=1

                print(f'{block_id}:{field_by_value}:{len(log_id_sequence)}:{len(behavior_sequence)}')
                field_by_sequence.append(field_by_value)
                block_ids.append(block_id)
                log_id_sequences.append(log_id_sequence)
                behavior_sequences.append(behavior_sequence)
                bid += 1
                block_id = "block" + str(bid)
                log_id_sequence = []
                behavior_sequence = []


        print(f'{len(block_ids)}:{len(field_by_sequence)}:{len(log_id_sequences)}:{len(behavior_sequences)}')
        data = {"block_id": block_ids,
                field_by[sub]: field_by_sequence,
                "log_id_sequence": log_id_sequences,
                "behavior_sequence": behavior_sequences}
    df2 = pd.DataFrame(data)
    #print(df2["block_id"])
    df2.to_csv(out_file_path, encoding='utf-8', index=False)

    del df, df2
            # print(pd.to_datetime(row[field_by])-pd.Timedelta(minutes=time_delta))
    # for index, row in df.iterrows():
    #     print(row[field_by])


'''
根据施动者字段（非时间字段）进行一次划分
'''
def log_partition2(field_by, logid_field, behavior_field, train_or_predict_mark):
    if train_or_predict_mark == 't':
        input_file_path = f'{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_train.csv'
        out_file_path = f"{get_parent_path()}/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_train.csv"
    elif train_or_predict_mark == 'p':
        input_file_path = f'{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_predict.csv'
        out_file_path = f"{get_parent_path()}/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_predict.csv"
    df = pd.read_csv(input_file_path)
    if len(field_by) == 1:
        block_ids = []  # 日志块id
        log_id_sequences = []  # 日志id序列
        behavior_sequences = []  # 行为序列
        field_by_values = []  # 施动者列表(依据分割字段值组成的列表)
        bid = 1
        block_id = "block" + str(bid)
        base_num = 100 #基准数目，即划分参照的数目
        base_num2 = 50 #每个施动者的划分的最后一个划分如果小于等于该值，就划分为上一组，否则划分为新的一组。
        agent_field = field_by[0] # 划分使用的施动者字段名
        agents = remove_duplicates(df[agent_field].tolist()) #获取不重复的施动者名组成列表

        for agent_ele in agents:
            agent_blocks = df[df[agent_field] == agent_ele]
            agent_blocks_num = agent_blocks.shape[0]
            quotient, remainder = divmod(agent_blocks_num, base_num)
            if quotient <= 1:
                block_ids.append(block_id)
                bid += 1
                block_id = "block" + str(bid)
                field_by_values.append(agent_ele)
                log_id_sequences.append(agent_blocks[logid_field].tolist())
                behavior_sequences.append(agent_blocks[behavior_field].tolist())
            else:
                for quotient_i in range(quotient):
                    if quotient_i != quotient-1:
                        agent_blocks_start = quotient_i * base_num
                        agent_blocks_end = agent_blocks_start + base_num - 1
                        block_ids.append(block_id)
                        bid += 1
                        block_id = "block" + str(bid)
                        field_by_values.append(agent_ele)
                        agent_block = agent_blocks.iloc[agent_blocks_start:agent_blocks_end]
                        log_id_sequences.append(agent_block[logid_field].tolist())
                        behavior_sequences.append(agent_block[behavior_field].tolist())
                    else:
                        if remainder <= base_num2:
                            agent_blocks_start = quotient_i * base_num
                            agent_blocks_end = agent_blocks_start + base_num + base_num2 - 1
                            block_ids.append(block_id)
                            bid += 1
                            block_id = "block" + str(bid)
                            field_by_values.append(agent_ele)
                            agent_block = agent_blocks.iloc[agent_blocks_start:agent_blocks_end]
                            log_id_sequences.append(agent_block[logid_field].tolist())
                            behavior_sequences.append(agent_block[behavior_field].tolist())
                        else:
                            continue
                    if remainder > base_num2:
                        agent_blocks_start = quotient * base_num
                        agent_blocks_end = agent_blocks_start + remainder - 1
                        block_ids.append(block_id)
                        bid += 1
                        block_id = "block" + str(bid)
                        field_by_values.append(agent_ele)
                        agent_block = agent_blocks.iloc[agent_blocks_start:agent_blocks_end]
                        log_id_sequences.append(agent_block[logid_field].tolist())
                        behavior_sequences.append(agent_block[behavior_field].tolist())
                    else:
                        continue

        # 写入数据
        data = {"block_id": block_ids,
                field_by[0]: field_by_values,
                "log_id_sequence": log_id_sequences,
                "behavior_sequence": behavior_sequences}
        df = pd.DataFrame(data)
        df.to_csv(out_file_path, encoding='utf-8', index=False)

        del data, df

'''
把csv文件或者其他格式中的行为序列转换成简单的格式
'''
def csv_to_simple(input_file_path): 
    list_data, list_block, list_logid = [], [], []   
    if os.path.splitext(input_file_path)[-1].lower() == ".csv":
        df = pd.read_csv(input_file_path, encoding="utf-8")
        selected_rows_bs = df['behavior_sequence'].values   
        selected_rows_bd = df['block_id'].values   
        selected_rows_lds = df['log_id_sequence'].values   
        for row_bs, row_bd, row_lds in  zip(selected_rows_bs, selected_rows_bd, selected_rows_lds):
            if (row_bs != "[]") and (row_bs != "nan"):
                list_data.append(row_bs)
                list_block.append(row_bd)
                list_logid.append(row_lds)
        # for row_bs, row_bd, row_lds in zip(list_data, list_block, list_logid):
        #     print(f"{row_bd}---{row_lds}---{row_bs}")
        return list_data, list_block, list_logid
    if os.path.splitext(input_file_path)[-1].lower() == "":
        with open(input_file_path, 'r') as f:
            list_data = f.readlines()
            return list_data, list_block, list_logid

'''
把csv文件或者其他格式中的业务行为序列转换成简单的格式
'''
def csv_to_simple2(input_file_path): 
    list_data, list_id = [], []   
    if os.path.splitext(input_file_path)[-1].lower() == ".csv":
        df = pd.read_csv(input_file_path)
        selected_rows_seqs = df['business_pattern_sequence'].values   
        selected_rows_ids = df['business_pattern_id'].values  
        for row_seqs, row_lds in  zip(selected_rows_seqs, selected_rows_ids):
            if row_seqs != "[]":
                list_data.append(row_seqs)
                list_id.append(row_lds)
        # for row_bs, row_bd, row_lds in zip(list_data, list_block, list_logid):
        #     print(f"{row_bd}---{row_lds}---{row_bs}")
        return list_data, list_id
    if os.path.splitext(input_file_path)[-1].lower() == "":
        with open(input_file_path, 'r') as f:
            list_data = f.readlines()
            return list_data, list_id

def csv_to_simple3(input_file_path):
    list_data, list_block = [], []
    if os.path.splitext(input_file_path)[-1].lower() == ".csv":
        df = pd.read_csv(input_file_path)
        selected_rows_bs = df['new_behavior_sequence'].values
        selected_rows_bd = df['block_id'].values
        for row_bs, row_bd in  zip(selected_rows_bs, selected_rows_bd):
            if row_bs != "[]":
                list_data.append(row_bs)
                list_block.append(row_bd)
        # for row_bs, row_bd, row_lds in zip(list_data, list_block, list_logid):
        #     print(f"{row_bd}---{row_lds}---{row_bs}")
        return list_data, list_block
    if os.path.splitext(input_file_path)[-1].lower() == "":
        with open(input_file_path, 'r') as f:
            list_data = f.readlines()
            return list_data, list_block

'''
对每条日志计算时间间隔
input_file_path: 日志文件路径
time_field: 时间字段
interval_standard: 时间间隔标准，若时间间隔超过interval_standard，那么此时间间隔就设置为0
'''


def log_time_interval(input_file_path, time_field, interval_standard):
    df = csv_or_json_data(input_file_path)
    timestamp = df[time_field]
    length = len(timestamp)
    time_intervals = [0] * length
    for index, row in enumerate(timestamp):
        current_datetime = pd.to_datetime(row)
        if index < length - 1:
            next_datetime = pd.to_datetime(timestamp[index + 1])
            time_interval = (pd.to_datetime(next_datetime) - pd.to_datetime(current_datetime)).total_seconds()
            if time_interval > 0 and time_interval < interval_standard:
                time_intervals[index] = time_interval
    df["time_interval"] = time_intervals
    df.to_csv(input_file_path, index=False)

    del df
    # print(time_intervals)

'''
针对于series类型的对象使用值value去查询检索
series中的检索可以是自定义的
'''


def find_value_for_series(data_series, refer_value):
    for index, value in data_series.items():
        if refer_value == value:
            return index

'''
返回对应值不缺省的检索
data_series: pandas series类型
'''


def get_not_NaN_indexes(data_series):
    indexes = []
    for index, value in data_series.items():
        if not (pd.isna(value)):
            indexes.append(index)
    return indexes

'''
根据编码后的日志文件针对每种行为形成对应的参数值矩阵
input_file_path: 编码后的日志文件路径以及编码映射文件路径
behavior_field: 行为字段名
para_field: 参数值字段名
logid_field: 日志id字段名
train_or_predict_mark: 指明生成的数据是训练数据还是预测数据，'t'标识训练数据，'p'标识预测数据。
'''


def generate_param_value(behavior_field, logid_field, train_or_predict_mark, para_field = []):
    input_file_path = ''
    output_file_path = ''
    para_field.append('time_interval') # time_interval为模型自动生成的字段，一般需要参与运算
    if train_or_predict_mark == 't':
        input_file_path = f"{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_train.csv,{get_parent_path()}/service/impl/tmpdata/encode_field_mapping.json"
        output_file_path = f'{get_parent_path()}/service/impl/tmpdata/ParamData/train_param.csv'
    elif train_or_predict_mark == 'p':
        input_file_path = f"{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_predict2.csv,{get_parent_path()}/service/impl/tmpdata/encode_field_mapping.json"
        output_file_path = f'{get_parent_path()}/service/impl/tmpdata/ParamData/predict_param.csv'
    input_file = input_file_path.split(",")
    input_file0 = input_file[0]; input_file1 = input_file[1]
    df_log = csv_or_json_data(input_file0); df_map = csv_or_json_data(input_file1)
    behavior_label = get_not_NaN_indexes(df_map[behavior_field])
    data_dict = {}
    behavior_category_logid = {}
    for element in behavior_label:
        selected_rows = df_log[df_log[behavior_field] == element]  # 选择指定值对应的行
        para_list = []
        logid_list = []
        for _, value_row in selected_rows.iterrows():
            para = []
            logid_list.append(value_row[logid_field])
            for field_row in para_field:
                para.append(value_row[field_row]) 
            para_list.append(para)
        behavior_category_logid[element] = logid_list
        data_dict[element] = para_list
    
    # 找到最长键的长度
    max_length = max(len(v) for v in data_dict.values())
    df_data = {k: v + [np.nan] * (max_length - len(v)) for k, v in data_dict.items()}
    para_frame = pd.DataFrame(df_data)
    para_frame.to_csv(output_file_path, index=False)

    del df_log

    return data_dict, behavior_category_logid

'''
当参数值训练和识别所需要的数据已经生成后，若需要读取生成后的数据。
'''
def get_generated_param_value(train_or_predict_mark):
    generated_param_value_path = ""
    if train_or_predict_mark == 't':
        generated_param_value_path = f'{get_parent_path()}/service/impl/tmpdata/ParamData/train_param.csv'
    elif train_or_predict_mark == 'p':
        generated_param_value_path = f'{get_parent_path()}/service/impl/tmpdata/ParamData/predict_param.csv'
    df = csv_or_json_data(generated_param_value_path)
    dict_df = df.to_dict(orient='list')

    del df

    return dict_df

'''
对序列行为异常检测预测输入的编码数据中的相关字段值进行解码
'''
def decode_map_file(predict_file, encode_map_file, behavior_field, agent_field):
    for index, row in predict_file.iterrows():
        replace_behavior_sequence = []
        replace_agent_sequence = []
        for element1 in eval(row["behavior_sequence"]):
            replace_element1 = encode_map_file[behavior_field][element1]
            if replace_element1:
                replace_behavior_sequence.append(replace_element1)
        replace_element2 = encode_map_file[agent_field][row[agent_field]]
        if replace_element2:
            replace_agent_sequence.append(replace_element2)
        predict_file["behavior_sequence"][index] = replace_behavior_sequence
        predict_file[agent_field][index] = replace_agent_sequence
    return predict_file

'''
对序列行为异常检测预测输入的编码数据中的相关字段值进行解码，并保持成新的文件。
'''
def decode_from_log_entry_to_behavior_block(agent_field, train_or_predict_mark):
    pass

'''
对行为序列识别结果生成的异常文件中的相关字段进行解码
'''
def decode_map_file2(sequence_abnormal_file, encode_map_file, behavior_field):
    for index, row in sequence_abnormal_file.iterrows():
        abnormal_behavior_sub_sequences = []
        predict_values = []
        for element1_list, element2_list in zip(eval(row["abnormal_behavior_sub_sequences"]), eval(row["predict_values"])):
            abnormal_behavior_sub_sequence_list = []
            predict_value_list = []
            for  element1 in element1_list:
                abnormal_behavior_sub_sequence_list.append(encode_map_file[behavior_field][element1])
            for element2 in element2_list:
                predict_value_list.append(encode_map_file[behavior_field][element2])
            abnormal_behavior_sub_sequences.append(abnormal_behavior_sub_sequence_list)
            predict_values.append(predict_value_list)
        sequence_abnormal_file["abnormal_behavior_sub_sequences"][index] = abnormal_behavior_sub_sequences
        sequence_abnormal_file["predict_values"][index] = predict_values
    return sequence_abnormal_file

'''
对业务意图识别结果生成的异常文件中的相关字段进行解码
'''
def decode_map_file3(business_intent_abnormal_file, encode_map_file, behavior_field):
    for index, row in business_intent_abnormal_file.iterrows():
        replace_behavior_sequences = []
        for elements in eval(row["abnormal_behavior_sub_sequences"]):
            replace_behavior_sequence = []
            for element in elements:
                replace_element = encode_map_file[behavior_field][element]
                if replace_element or replace_element !="":
                    replace_behavior_sequence.append(replace_element)
            replace_behavior_sequences.append(replace_behavior_sequence)
        business_intent_abnormal_file["abnormal_behavior_sub_sequences"][index] = replace_behavior_sequences
    return business_intent_abnormal_file

'''
获取某一文件夹下所有文件的文件名
'''
def get_filenames(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

'''
存放训练阶段设置的函数参数值，以备检测阶段使用。
'''
def save_param_values_from_train_dao(field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n):
    # 生成需要编码的字段
    require_encode_field = []
    agent_field = ""
    require_encode_field.append(behavior_field)
    if behavior_field not in para_field:
        require_encode_field += para_field
    if len(field_by) == 1:
        agent_field = field_by[0]
    elif len(field_by) ==2:
        agent_field = field_by[1]
    if agent_field not in require_encode_field:
        require_encode_field.append(agent_field)

    param_values_dict = {}
    param_values_dict['require_encode_field'] = require_encode_field
    param_values_dict['field_by'] = field_by
    param_values_dict['logid_field'] = logid_field
    param_values_dict['behavior_field'] = behavior_field
    param_values_dict['time_delta'] = time_delta
    param_values_dict['time_field'] = time_field
    param_values_dict['interval_standard'] = interval_standard
    param_values_dict['para_field'] = para_field
    param_values_dict['agent_field'] = agent_field
    param_values_dict['window_size'] = window_size
    param_values_dict['zoom_value'] = zoom_value
    param_values_dict['top_n'] = top_n

    with open(f"{get_parent_path()}/service/impl/tmpdata/param_values_from_train.txt", 'w') as f:
        # 写入参数值
        json.dump(param_values_dict, f)
    del param_values_dict

'''
检测（预测）阶段获取训练阶段存放的参数值
'''
def get_param_values_from_train_dao():
    with open(f"{get_parent_path()}/service/impl/tmpdata/param_values_from_train.txt") as f:
        param_values = json.load(f)
    return param_values

'''
把非utf-8编码格式的csv文件转换成utf-8格式的编码并替换掉原始文件
'''
def convert_to_utf8(csv_file):
    csv_filename = os.path.basename(csv_file) #获取带后缀的文件名
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    output_file_path = f'{get_parent_path()}/service/impl/tmpdata/TmpLog/{csv_filename}'
    with open(csv_file, 'r', encoding = encoding, errors='ignore') as input_file, codecs.open(output_file_path, 'w', 'utf-8') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        for row in reader:
            writer.writerow(row)

    os.remove(csv_file)
    shutil.copy(output_file_path, csv_file)
    os.remove(output_file_path)

'''
删除文件夹中的所有文件
'''
def clear_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

'''
保存Json格式的文件
'''
def save_Json_file(file_path, result_mapping):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(result_mapping, file, ensure_ascii=False, indent=4)

'''
保存行为类别的最大编码
'''
def save_max_behavior_encoding_dao(behavior_field):
    read_file_path = f'{get_parent_path()}/service/impl/tmpdata/encode_field_mapping.json'
    out_file_path = f'{get_parent_path()}/service/impl/tmpdata/max_behavior_coding.txt'
    with open(read_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    max_behavior_coding = list(data[behavior_field].keys())[-1] #最大编码

    # 写入数据
    with open(out_file_path, 'w',encoding='utf-8') as file:
        file.write(max_behavior_coding)

    del data

'''
获得行为编码的最大编码
'''
def get_max_behavior_encoding_dao():
    with open(get_parent_path() + '/service/impl/tmpdata/max_behavior_coding.txt', 'r') as f:
        return f.readline()

'''
删除向量化后csv文件中的没有对指定字段向量化的数据条目，并在原始csv文件中把对应的数据条目提取出来存放到对应的异常文件中。
'''
def delete_no_vectorize_and_save(vectorized_csv, original_csv, para_field):
    df_vectorized_data = csv_or_json_data(vectorized_csv)
    df_vectorized_data2 = df_vectorized_data
    df_original_data = csv_or_json_data(original_csv)
    df_abnormal_data = pd.DataFrame(columns=df_original_data.columns)
    df_abnormal_data['abnormal_type'] = None

    for index, row in df_vectorized_data.iterrows():
        row2 = df_original_data.iloc[index]
        for para_field_item in para_field:
            if row[para_field_item] == row2[para_field_item]:
                df_vectorized_data2 = df_vectorized_data2.drop(index)
                row2['abnormal_type'] = 'new'
                df_abnormal_data = df_abnormal_data._append(row2)

                break

    out_file_path = f'{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_predict2.csv'
    # 存入异常
    out_file_path2 = f'{get_parent_path()}/service/impl/tmpdata/Anormaly/para_anormal.csv'
    df_vectorized_data2.to_csv(out_file_path, index=False)
    df_abnormal_data.to_csv(out_file_path2, index=False)

# 获取指定文件夹下第一个文件的文件名
def get_first_file(directory):
    # 列出文件夹下的所有文件，并获取第一个文件名
    files = os.listdir(directory)
    if files:
        return files[0]
    return None

'''
删除列表中重复的元素并保持原有顺序
'''
def remove_duplicates(lst):
    seen = set()
    return list([x for x in lst if not (x in seen or seen.add(x))])

'''
对原始的csv文件按照时间字段值的先后顺序进行排序，并覆盖掉原始csv文件
'''
def csv_sort_values_by_time(csv_file_path, time_field):
    df = csv_or_json_data(csv_file_path)
    df_sorted_ascending = df.sort_values(by=time_field)
    csv_filename = os.path.basename(csv_file_path)
    output_file_path = f'{get_parent_path()}/service/impl/tmpdata/TmpLog/{csv_filename}'
    df_sorted_ascending.to_csv(output_file_path, index=False, encoding='utf-8')
    os.remove(csv_file_path)
    shutil.copy(output_file_path, csv_file_path)
    os.remove(output_file_path)


# 测试用例代码块
# if __name__ == '__main__':
#     directory = 'D:/workspace/java/test4/behaviorIntention/behavior_intention/cn/edu/xidian/sai/service/impl/tmpdata/SeqModel'
#     file_name = get_first_file(directory)
#     print(file_name)
































