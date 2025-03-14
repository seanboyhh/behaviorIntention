'''
Created on 2024年4月7日

@author: 13507
'''

import pandas as pd
import numpy as np

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import csv_or_json_data
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import get_max_behavior_encoding
from behavior_intention.cn.edu.xidian.sai.service.impl.common_tool import is_two_dimensional_list_contained


# 构建共现矩阵
def create_cooccurrence_matrix(distance, behavior_sequence):
    # 创建一个初始化矩阵
    line_num = int(get_max_behavior_encoding())
    df = pd.DataFrame(np.zeros((line_num, line_num)))
    df.index += 1
    df.columns += 1
    
    # 获得行为数据并转换成列表形式
    behavior_sequences = csv_or_json_data(behavior_sequence)['behavior_sequence']
    behavior_sequences2 = []
    for behavior_sequence in behavior_sequences:
        if behavior_sequence != '[]':
            behavior_sequences2.append(eval(behavior_sequence))
    behavior_sequences = behavior_sequences2
    
    # 计算行为共现
    for i  in range(line_num):
        i += 1
        for j in range(line_num): 
            j += 1
            frequency_d_i = 0
            frequency_d_i_j = 0
            for behavior_sequence in behavior_sequences:
                frequency_d_i += behavior_sequence.count(i)
                for index, behavior in enumerate(behavior_sequence):
                    if behavior == i:
                        if (index + distance <= len(behavior_sequence) - 1) and (behavior_sequence[index + distance] == j):
                            frequency_d_i_j += 1
                        else:
                            continue
                    else:
                        continue 
            # 计算比例值，即概率值       
            df.loc[i, j] = frequency_d_i_j / frequency_d_i if frequency_d_i != 0 else 0
    df.to_csv(get_parent_path() + '/service/impl/tmpdata/CooccurrenceMatrix/CooccurrenceMatrix_d' + str(distance) + '.csv', index = True)

    del df

# 根据行为之间的共现情况计算行为之间的关联程度，即根据不同的距离分别构建共现矩阵
def calculate_behaviors_cooccurrence_degree(max_distance):
    behavior_sequence = f"{get_parent_path()}/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_train.csv"
    # 构建从距离为1到max_distance的共现矩阵，步长为1
    for i in range(max_distance):
        distance = i + 1 
        create_cooccurrence_matrix(distance, behavior_sequence)

def business_pattern_mining(single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold):
    new_single_business_pattern = single_business_pattern
    vacancy_num = max_distance + 1 - len(single_business_pattern) # 可插入的空位数
    
    if vacancy_num <= 0:
        return all_business_pattern
    
    # 基于共现矩阵挖掘业务模式
    if vacancy_num >0: # 至少需要一个空位
        # print(f'len(single_business_pattern)={len(single_business_pattern)}, vacancy_num={vacancy_num}')
        for _ in range(vacancy_num): 
            for j in range(max_behavior_encoding):
                candidate_behavior = j + 1 # 候选行为
                trail_detect_count = 0 # 尾插检测计数器，记录超过阈值的次数
                head_detect_count = 0 # 头插检测计数器，记录超过阈值的次数
                
                if vacancy_num > 0:
                    # 尾插法计数
                    for detect_behavior in reversed(single_business_pattern): # detect_behavior为检测行为
                        print(f'尾插计数, single_business_pattern={single_business_pattern}, len(single_business_pattern)={len(single_business_pattern)}, trail_detect_count={trail_detect_count}')
                        if trail_detect_count < max_distance:
                            if cooccurrence_matrices[trail_detect_count].loc[detect_behavior, candidate_behavior] >= threshold:
                                trail_detect_count += 1
                            else:
                                break # 提前退出循环
                        else:
                            break
                        
                    # 头插法计数
                    for detect_behavior in single_business_pattern:
                        print(f'头插计数, single_business_pattern={single_business_pattern}, len(single_business_pattern)={len(single_business_pattern)}, head_detect_count={head_detect_count}')
                        if head_detect_count < max_distance:
                            if cooccurrence_matrices[head_detect_count].loc[candidate_behavior, detect_behavior] >= threshold:
                                head_detect_count += 1
                            else:
                                break
                        else:
                            break
                            
                # 尾插
                if trail_detect_count == len(single_business_pattern):
                    print(f'尾插, single_business_pattern={single_business_pattern}, len(single_business_pattern)={len(single_business_pattern)}, head_detect_count={head_detect_count}')
                    # new_single_business_pattern = single_business_pattern
                    new_single_business_pattern.append(candidate_behavior)
                    business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                
                # 头插
                if head_detect_count == len(single_business_pattern):
                    print(f'头插, single_business_pattern={single_business_pattern}, len(single_business_pattern)={len(single_business_pattern)}, head_detect_count={head_detect_count}')
                    # new_single_business_pattern = single_business_pattern
                    new_single_business_pattern.insert(0, candidate_behavior)
                    business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                
                if vacancy_num > 0:
                    # 双插
                    if trail_detect_count != len(single_business_pattern) and head_detect_count != len(single_business_pattern):
                        for k in range(max_behavior_encoding):
                            candidate_behavior1 = k + 1
                            for l in range(max_behavior_encoding):
                                candidate_behavior2 = l + 1
                                double_trail_detect_count = 0 # 双尾插检测计数器
                                double_head_detect_count = 0 # 双头插检测计数器
                                
                                # 双尾插法计数
                                for detect_behavior in reversed(single_business_pattern):
                                    print(f'双尾插计数, single_business_pattern={single_business_pattern}, len(single_business_pattern)={len(single_business_pattern)}, double_trail_detect_count={double_trail_detect_count}')
                                    if double_trail_detect_count < max_distance:
                                        if cooccurrence_matrices[double_trail_detect_count].loc[detect_behavior, candidate_behavior1] + cooccurrence_matrices[double_trail_detect_count].loc[detect_behavior, candidate_behavior2] >= threshold: 
                                            double_trail_detect_count += 1
                                        else:
                                            break
                                    else:
                                        break
                                
                                # 双头插法计数
                                for detect_behavior in single_business_pattern:
                                    print(f'双头插计数, single_business_pattern={single_business_pattern}, len(single_business_pattern)={len(single_business_pattern)}, double_head_detect_count={double_head_detect_count}')
                                    #print(f'single_business_pattern={single_business_pattern} ,single_business_pattern的长度={len(single_business_pattern)} ,double_head_detect_count={double_head_detect_count}')
                                    if double_head_detect_count < max_distance:
                                        if cooccurrence_matrices[double_head_detect_count].loc[candidate_behavior1, detect_behavior] + cooccurrence_matrices[double_head_detect_count].loc[candidate_behavior2, detect_behavior] >= threshold: 
                                            double_head_detect_count += 1
                                        else:
                                            break
                                    else:
                                        break
                                    
                                # 双尾插
                                if double_trail_detect_count == len(single_business_pattern):
                                    # 一次插两个
                                    if cooccurrence_matrices[0].loc[candidate_behavior1, candidate_behavior2] >= threshold:
                                        if vacancy_num > 1: # 至少需要两个空位
                                            print('双尾插，一次两个')
                                            # new_single_business_pattern = single_business_pattern
                                            new_single_business_pattern.extend([candidate_behavior1, candidate_behavior2])
                                            business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                                    
                                    # 分开各插一次
                                    else:
                                        # new_single_business_pattern = single_business_pattern
                                        new_single_business_pattern_temp = single_business_pattern
                                        new_single_business_pattern.append(candidate_behavior1)
                                        business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                                        new_single_business_pattern = new_single_business_pattern_temp
                                        new_single_business_pattern.append(candidate_behavior2)
                                        business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                                                                                                       
                                # 双头插
                                if double_head_detect_count == len(single_business_pattern):
                                    if cooccurrence_matrices[0].loc[candidate_behavior1, candidate_behavior2] >= threshold:
                                        if vacancy_num > 1:
                                            print('test')
                                            print(f'两个行为：{[candidate_behavior1, candidate_behavior2]}')
                                            # new_single_business_pattern = single_business_pattern
                                            print(f'插入前的new_single_business_pattern: {new_single_business_pattern}')
                                            new_single_business_pattern = [candidate_behavior1, candidate_behavior2] + new_single_business_pattern
                                            print(f'插入后的new_single_business_pattern: {new_single_business_pattern}')
                                            business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                                    else:
                                        # new_single_business_pattern = single_business_pattern
                                        new_single_business_pattern_temp = single_business_pattern
                                        new_single_business_pattern.insert(0, candidate_behavior1)
                                        business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                                        new_single_business_pattern = new_single_business_pattern_temp
                                        new_single_business_pattern.insert(0, candidate_behavior2)
                                        business_pattern_mining(new_single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
                                
                                if double_trail_detect_count != len(single_business_pattern) and double_head_detect_count != len(single_business_pattern):
                                    if not is_two_dimensional_list_contained(new_single_business_pattern, all_business_pattern):
                                        print(f'待插入的业务模式为：{new_single_business_pattern}')
                                        all_business_pattern.append(new_single_business_pattern)
                                    break
                                                  
            return all_business_pattern                   

'''
根据共现矩阵挖掘业务模式
threshold为共现阈值
'''
def all_business_pattern_mining(max_distance, threshold):
    # 读取共现矩阵
    cooccurrence_matrices = []
    for i in range(max_distance):
        matrix_name = get_parent_path() + '/service/impl/tmpdata/CooccurrenceMatrix/CooccurrenceMatrix_d' + str(i+1) + '.csv'
        cooccurrence_matrice = pd.read_csv(matrix_name, index_col=0)
        cooccurrence_matrice.columns = cooccurrence_matrice.columns.astype(int)
        cooccurrence_matrices.append(cooccurrence_matrice)
    
    max_behavior_encoding = int(get_max_behavior_encoding())
    
    single_business_pattern = []  # 初始业务模式
    all_business_pattern = [] # 初始化业务模式列表
    all_business_patterns = []
    
    # 业务挖掘
    for start_num in range(1, max_behavior_encoding + 1):
        single_business_pattern = [start_num]
        all_business_pattern = business_pattern_mining(single_business_pattern, all_business_pattern, cooccurrence_matrices, max_distance, max_behavior_encoding, threshold)
        all_business_patterns += all_business_pattern
                        
    print(f'业务模式: {all_business_patterns}')
    
    # 存储业务模式
    business_pattern_ids = []
    for i in range(len(all_business_patterns)):
        business_pattern_ids.append('bp' + str(i + 1))
    data = {'business_pattern_id': business_pattern_ids,
            'business_pattern_sequence': all_business_patterns
            }
    
    df = pd.DataFrame(data)
    
    df.to_csv(get_parent_path() + '/service/impl/tmpdata/BussinessPattern/bussiness_pattern.csv', index = False)

    del df







