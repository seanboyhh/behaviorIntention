'''
Created on Feb 24, 2024

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_describe_service import behavior_sequence_description as bsd
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import get_param_values_from_train_service

def behavior_sequence_describe_handler(behavior_file_path):
    predict_file_path = get_parent_path() + "/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_predict.csv"
    sequence_abnormal_file_path = get_parent_path() + "/service/impl/tmpdata/Anormaly/behavior_sequence_anormal.csv"
    param_abnormal_file_path = get_parent_path() + "/service/impl/tmpdata/Anormaly/para_anormal.csv"
    business_intent_abnormal_file_path = get_parent_path() + "/service/impl/tmpdata/Anormaly/business_behavior_sequence_anormal.csv"
    encode_map_file_path = get_parent_path() + "/service/impl/tmpdata/encode_field_mapping.json"
    param_values = get_param_values_from_train_service()
    behavior_field = param_values['behavior_field']
    logid_field = param_values['logid_field']
    time_field = param_values['time_field']
    agent_field = param_values['agent_field']

    bsd(behavior_file_path, predict_file_path, sequence_abnormal_file_path, param_abnormal_file_path,
        business_intent_abnormal_file_path,
        encode_map_file_path, behavior_field, agent_field, logid_field, time_field)



# 测试用例代码块
if __name__ == '__main__':
    behavior_file_path = "D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211.csv"
    behavior_sequence_describe_handler(behavior_file_path)
    
    
    
    
    
    