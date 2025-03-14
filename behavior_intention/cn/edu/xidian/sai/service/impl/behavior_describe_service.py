'''
Created on Dec 10, 2023

@author: 13507
'''
import pandas as pd

from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import csv_or_json_data
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import get_param_values_from_train_dao
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import decode_map_file, decode_map_file2, decode_map_file3
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path


def single_behavior_description(file_path):
    file_path = file_path.split(',')
    file_path0 = file_path[0]
    file_path1 = file_path[1]
    # 读取带行为分类标签的行为数据
    df = csv_or_json_data(file_path0)
    file = open(file_path1, "w")
    for index, row in df.iterrows():
        node_description = ""
        label = ""
        data_tag = str(row['data_tag'])
        person_type = str(row['src_person_type'])
        person_status = str(row['src_person_status'])
        if row['authority_label'] == 0:
            label = "经授权的"
        else:
            label = "未经授权的"
        if data_tag == "nan":
            data_tag = "未知"
        if person_type == "企业员工":
            person_type = "企业"
        elif person_type == "编外人员":
            person_type = "编外"
        if person_status == "离职":
            person_status = "离职后"
        else:
            person_status += "内"
        node_description += "单行为" + str(index) + ": " + row['log_create_time'] + ", " + label + person_status
        node_description += "的" + person_type + row['src_person_ctpositionname'] + row[
            'src_person_name'] + ", " + "在" + str(row['src_device_dept']) + "的"
        node_description += "设备" + str(row['src_device_name']) + "上" + "进行了" + data_tag + "的操作"
        file.write(node_description + "\n")
        print(node_description)
    file.close()


'''
行为序列异常检测和参数值异常检测后，进行综合的异常判定.
'''


def abnormal_judgemet_by_sequence_and_param(original_file, predict_file, sequence_abnormal_file, param_abnormal_file,
                                            business_intent_abnormal_file, logid_field, time_field):
    block_ids, abnormal_marks, abnormal_log_ids, start_and_end_times = [], [], [], []
    for _, row in predict_file.iterrows():
        behavior_sequence_abnormal_mark, parameter_abnormal_mark, business_intent_abnormal_mark = 0, 0, 0
        abnormal_log_ids1 = []
        start_and_end_time = []
        select_block_id = row["block_id"]

        # 行为序列和对应的参数值异常判定
        if select_block_id in sequence_abnormal_file["abnormal_block_ids"].values:
            behavior_sequence_abnormal_mark = 1
            select_list = eval(sequence_abnormal_file[sequence_abnormal_file["abnormal_block_ids"] == select_block_id][
                                   "abnormal_log_ids"].values[0])
            abnormal_log_ids1 += select_list
            for log_id in eval(row["log_id_sequence"]):
                if log_id in param_abnormal_file[logid_field].values:
                    parameter_abnormal_mark = 1
                    if log_id not in sequence_abnormal_file["abnormal_log_ids"].values:
                        abnormal_log_ids1.append(log_id)
        else:
            for log_id in eval(row["log_id_sequence"]):
                if log_id in param_abnormal_file[logid_field].values:
                    parameter_abnormal_mark = 1
                    abnormal_log_ids1.append(log_id)

        # 业务意图(异常判定)
        if select_block_id in business_intent_abnormal_file['abnormal_block_ids'].values:
            business_intent_abnormal_mark = 1
            select_list = eval(
                business_intent_abnormal_file[business_intent_abnormal_file["abnormal_block_ids"] == select_block_id][
                    "abnormal_log_ids"].values[0])
            for log_id in select_list:
                if log_id not in abnormal_log_ids1:
                    abnormal_log_ids1.append(log_id)

        block_ids.append(select_block_id)
        abnormal_marks.append([behavior_sequence_abnormal_mark, parameter_abnormal_mark, business_intent_abnormal_mark])
        abnormal_log_ids.append(abnormal_log_ids1)

        # 获得开始和结束时间
        log_id_sequence = eval(row["log_id_sequence"])
        if log_id_sequence:
            start_log = log_id_sequence[0]
            end_log = log_id_sequence[-1]
            start_time = original_file[original_file[logid_field] == start_log][time_field].values[0]
            end_time = original_file[original_file[logid_field] == end_log][time_field].values[0]
            start_and_end_time = [start_time, end_time]
        start_and_end_times.append(start_and_end_time)
    # print(f"{block_ids[78]}:{abnormal_marks[78]}:{abnormal_log_ids[78]}")
    # for row1, row2, row3 in zip(block_ids, abnormal_marks, abnormal_log_ids):
    #     print(f"{row1}:{row2}:{row3}")
    # print(f"{len(block_ids)}:{len(abnormal_marks)}:{len(abnormal_log_ids)}:{len(start_and_end_times)}")
    return block_ids, abnormal_marks, abnormal_log_ids, start_and_end_times


'''
生成行为序列描述.
'''


def behavior_sequence_description(original_file_path, predict_file_path, sequence_abnormal_file_path,
                                  param_abnormal_file_path, business_intent_abnormal_file_path, encode_map_file_path,
                                  behavior_field, agent_field, logid_field, time_field):
    original_file = csv_or_json_data(original_file_path)
    predict_file = csv_or_json_data(predict_file_path)
    sequence_abnormal_file = csv_or_json_data(sequence_abnormal_file_path)
    param_abnormal_file = csv_or_json_data(param_abnormal_file_path)
    business_intent_abnormal_file = csv_or_json_data(business_intent_abnormal_file_path)

    # 解码成明文
    encode_map_file = csv_or_json_data(encode_map_file_path)
    decode_predict_file = decode_map_file(predict_file, encode_map_file, behavior_field, agent_field)
    decode_sequence_abnormal_file = decode_map_file2(sequence_abnormal_file,encode_map_file, behavior_field)
    decode_business_intent_abnormal_file = decode_map_file3(business_intent_abnormal_file, encode_map_file, behavior_field)

    block_ids, abnormal_marks, abnormal_log_ids, start_and_end_times = abnormal_judgemet_by_sequence_and_param(
        original_file, predict_file, sequence_abnormal_file, param_abnormal_file, business_intent_abnormal_file,
        logid_field, time_field)
    behavior_sequence_descriptions = []
    abnormal_block_ids = []
    description_ids = []
    did = 1
    for block_id, abnormal_mark, abnormal_log_id, start_and_end_time in zip(block_ids, abnormal_marks, abnormal_log_ids,
                                                                            start_and_end_times):
        # 生成异常自然语言描述，仅仅生成异常的，正常的不生成。
        if any(element == 1 for element in abnormal_mark):
            window_size = int(get_param_values_from_train_dao()['window_size'])
            predict_behavior_sequence = decode_predict_file[decode_predict_file['block_id'] == block_id]['behavior_sequence'].values[0]
            if len(predict_behavior_sequence) > window_size: # 序列长度满足最小的长度时
                if start_and_end_time:
                    description_time = f"{start_and_end_time[0]}到{start_and_end_time[1]}之间，"
                    description_agent = str(decode_predict_file[decode_predict_file['block_id'] == block_id][agent_field].values[0]).strip('[]') + "执行了一条操作行为序列："
                    description_behaviors = str(predict_behavior_sequence) + "。 "
                    description_window = f"依据滑动窗口大小设置为{window_size}，"
                    description_model = "经AI模型识别和分析后得出："
                    description_abnormal_sub_behavior_sequence = ""
                    description_conj = ""
                    description_abnormal_param_value = ""
                    description_abnormal_business_intention = ""

                    # 生成行为序列异常描述
                    if abnormal_mark[0] == 1:
                        description_abnormal_sub_behavior_sequence = f"该操作行为序列出现行为序列异常，出现异常的子序列为"
                        abnormal_behavior_sequence = decode_sequence_abnormal_file[decode_sequence_abnormal_file['abnormal_block_ids'] == block_id]
                        abnormal_behavior_sub_sequences = abnormal_behavior_sequence['abnormal_behavior_sub_sequences'].values[0]
                        predict_values_lists = abnormal_behavior_sequence['predict_values'].values[0]
                        predict_pros_lists = eval(abnormal_behavior_sequence['predict_pros'].values[0])

                        # 把预测值与概率值合并在一起，形成预测值概率分布，并对每条异常子序列生成对应的描述。
                        for predict_value_list, predict_pro_list, abnormal_behavior_sub_sequence in zip(predict_values_lists, predict_pros_lists, abnormal_behavior_sub_sequences):
                            predict_values_pros = []
                            for predict_value, predict_pro in zip(predict_value_list, predict_pro_list):
                                predict_values_pros.append(f'{predict_value}：{predict_pro}')
                            description_abnormal_sub_behavior_sequence += f'，子序列{abnormal_behavior_sub_sequence}（此子序列中最后一个行为的预测值概率分布为：{predict_values_pros}）'
                        description_abnormal_sub_behavior_sequence += "。"

                    # 生成参数值异常描述
                    if abnormal_mark[1] == 1:
                        if abnormal_mark[0] == 1:
                            description_abnormal_sub_behavior_sequence += "并且"
                        description_abnormal_sub_behavior_sequence += "该操作行为序列出现参数值异常，出现参数值异常的行为有"

                        # 判断第几个行为出现参数值异常
                        log_id_sequence = eval(decode_predict_file[decode_predict_file['block_id'] == block_id]['log_id_sequence'].values[0])
                        log_ids_param = param_abnormal_file[logid_field]
                        # print(f'{log_id_sequence}:{log_ids_param}')
                        index_log = 1
                        abnormal_positions = []
                        for log_id in log_id_sequence:
                            if log_id in log_ids_param.values:
                                abnormal_positions.append(index_log)
                            index_log += 1
                        index_position = 0
                        for abnormal_position in abnormal_positions:
                            if index_position < len(abnormal_positions) - 1:
                                description_abnormal_sub_behavior_sequence += f"第{abnormal_position}个行为，"
                            else:
                                description_abnormal_sub_behavior_sequence += f"第{abnormal_position}个行为。"
                            index_position += 1

                    # 生成业务意图异常描述
                    if abnormal_mark[2] == 1:
                        abnormal_behavior_sequence = decode_business_intent_abnormal_file[decode_business_intent_abnormal_file['abnormal_block_ids'] == block_id]
                        abnormal_behavior_sub_sequences = abnormal_behavior_sequence['abnormal_behavior_sub_sequences'].values[0]

                        if abnormal_mark[0] == 1 or abnormal_mark[1] == 1:
                            description_abnormal_sub_behavior_sequence += "除此之外，子序列"

                        # 对每条异常子序列生成描述
                        index_sub = 0
                        for abnormal_behavior_sub_sequence in abnormal_behavior_sub_sequences:
                            if index_sub < len(abnormal_behavior_sub_sequences) - 1:
                                description_abnormal_sub_behavior_sequence += f"{abnormal_behavior_sub_sequence}，"
                            else:
                                description_abnormal_sub_behavior_sequence += f"{abnormal_behavior_sub_sequence}"
                            index_sub += 1
                        description_abnormal_sub_behavior_sequence += "不符合AI模型学习到的业务模式，出现业务意图异常。"

                    behavior_sequence_description = (description_time + description_agent + description_behaviors +
                                                     description_window     +
                                                     description_model + description_abnormal_sub_behavior_sequence +
                                                     description_conj + description_abnormal_param_value +
                                                     description_abnormal_business_intention)
                    behavior_sequence_descriptions.append(behavior_sequence_description)
                    abnormal_block_ids.append(block_id)
                    description_ids.append(f'des{did}')
                    did += 1
                    # print(behavior_sequence_description)
    # 写入文件
    data = {"description_ids": description_ids,
            "block_ids": abnormal_block_ids,
            "recognition_descriptions": behavior_sequence_descriptions
    }
    df_des = pd.DataFrame(data)
    df_des.to_csv(f'{get_parent_path()}/service/impl/tmpdata/RecognitionDescribe/recognition_describe.csv', encoding='utf-8', index=False)

    del df_des
