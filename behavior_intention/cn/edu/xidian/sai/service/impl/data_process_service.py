'''
Created on May 18, 2024

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import save_param_values_from_train_dao
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import get_param_values_from_train_dao
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import get_max_behavior_encoding_dao, save_max_behavior_encoding_dao
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import log_partition, log_partition2

'''
获得行为编码的最大编码
'''
def get_max_behavior_encoding():
    return get_max_behavior_encoding_dao()

'''
存储
'''
def save__max_behavior_encoding(behavior_field):
    save_max_behavior_encoding_dao(behavior_field)

def save_param_values_from_train_service(field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n):
    save_param_values_from_train_dao(field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n)

def get_param_values_from_train_service():
    return get_param_values_from_train_dao()

def data_divide_service(field_by, logid_field, behavior_field, time_delta ,train_or_predict_mark):
    if len(field_by) == 2:
        log_partition(field_by, logid_field, behavior_field, time_delta ,train_or_predict_mark)
    elif len(field_by) == 1:
        log_partition2(field_by, logid_field, behavior_field, train_or_predict_mark)