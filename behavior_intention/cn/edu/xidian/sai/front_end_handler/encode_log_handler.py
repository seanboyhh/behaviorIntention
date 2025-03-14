'''
Created on Nov 10, 2023

@author: 13507
'''

#from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.service.impl.encode_log_service import create_mapping_from_csv_service
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import save_max_behavior_encoding_dao

def encode_log_handler(behavior_file, behavior_field):
    # 函数调用, 编码
    create_mapping_from_csv_service(behavior_file)

    # 函数调用，保存最大的行为编码
    save_max_behavior_encoding_dao(behavior_field)

# 测试用例代码块
if __name__ == '__main__':
    behavior_file = 'D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211-20w.csv'
    behavior_field = "generic_result_action"
    encode_log_handler(behavior_file, behavior_field)
    