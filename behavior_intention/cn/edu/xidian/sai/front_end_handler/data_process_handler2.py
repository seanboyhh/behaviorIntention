'''
Created on Jan 18, 2024

@author: 13507
'''

#from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import to_timestamp
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import log_time_interval
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import generate_param_value, delete_no_vectorize_and_save
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path

def generate_param_value_handler(behavior_file, time_field, interval_standard, behavior_field, logid_field, train_or_predict_mark, para_field):
    if train_or_predict_mark == 't':
        input_file_path = f"{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_train.csv"
    elif train_or_predict_mark == 'p':
        vectorized_csv = f"{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_predict.csv"
        delete_no_vectorize_and_save(vectorized_csv, behavior_file, para_field)

        input_file_path = f"{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_predict2.csv"
    log_time_interval(input_file_path, time_field, interval_standard)
    return generate_param_value(behavior_field, logid_field, train_or_predict_mark, para_field)



# 测试用例代码块
# if __name__ == '__main__':
#     time_field="log_create_time"
#     interval_standard=30*60 # 单位为秒
#     behavior_field="program_name"
#     logid_field="uuid"
#     para_field=["http_httpagent", "http_response_size", "http_httpmethod", "http_url_externalurl", "src_device_ip", "http_httpstatus"]
#     train_or_predict_mark = 't'
#     generate_param_value_handler(behavior_field, logid_field, train_or_predict_mark, para_field)
    
    
    
    
    