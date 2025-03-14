'''
Created on Mar 6, 2024

@author: 13507
'''

# from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import to_timestamp
# from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import log_time_interval
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_knowlege_graph_service import get_and_save_triples_from_log
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_knowlege_graph_service import generate_knowlege_graphs
from behavior_intention.cn.edu.xidian.sai.service.impl.data_process_service import get_param_values_from_train_service

def behavior_knowlege_graph_handler(original_log_file_path):
    # 获取训练阶段存储的参数设置
    param_values_from_train  = get_param_values_from_train_service()

    # 提取三元组
    get_and_save_triples_from_log(original_log_file_path, param_values_from_train['logid_field'], param_values_from_train['agent_field'], param_values_from_train['behavior_field'], param_values_from_train['para_field'])
    
    # 生成图谱
    generate_knowlege_graphs()


# 测试用例代码块
if __name__ == '__main__':
    original_log_file_path = 'D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211-20w.csv'
    behavior_knowlege_graph_handler(original_log_file_path)
    
    
    