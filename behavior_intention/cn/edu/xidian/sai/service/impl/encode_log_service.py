'''
Created on Jan 12, 2025

@author: 13507
'''


from behavior_intention.cn.edu.xidian.sai.dao.impl.extract_log_dao import create_mapping_from_csv_dao
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import get_param_values_from_train_dao


def create_mapping_from_csv_service(behavior_file):
    # 获取需要编码的字段名
    require_encode_field = get_param_values_from_train_dao()['require_encode_field']

    create_mapping_from_csv_dao(behavior_file, require_encode_field)