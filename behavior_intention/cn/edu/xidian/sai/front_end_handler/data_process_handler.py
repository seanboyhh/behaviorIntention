'''
Created on Jan 18, 2024

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import data_divide_service
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import save_param_values_from_train_service
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import get_param_values_from_train_service

def data_divide_handler(field_by, logid_field, behavior_field, time_delta, train_or_predict_mark):
    data_divide_service(field_by, logid_field, behavior_field, time_delta, train_or_predict_mark)

def save_param_values_from_train_handler(field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n):
    save_param_values_from_train_service(field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n)

def get_param_values_from_train_handler():
    return get_param_values_from_train_service()



# 测试用例代码块
if __name__ == '__main__':
    field_by = ["src_account"]
    logid_field = "log_uuid"
    behavior_field = "generic_result_action"
    time_delta = 6000 # 时间差值，单位为分钟
    time_field = "datetime"
    interval_standard = 30 * 60  # 单位为秒
    para_field = ["src_account", "src_device_name", "src_device_ip"]
    train_or_predict_mark = 'p'
    data_divide_handler(field_by, logid_field, behavior_field, time_delta, train_or_predict_mark)
    
    # param_values = get_param_values_from_train_handler()

    # save_param_values_from_train_handler(field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field)


    
    
    
    
    
    
    