'''
Created on 2024年4月3日

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.main_api.behavior_data_preprocess_main_api import behavior_data_preprocess_predict_main_api
from behavior_intention.cn.edu.xidian.sai.front_end_handler.behavior_sequence_predict_handler import behavior_sequence_predict_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.param_value_predic_handler import param_value_predic_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.business_behavior_sequence_predict_handler import business_behavior_sequence_predict_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.data_process_handler import get_param_values_from_train_handler
from  behavior_intention.cn.edu.xidian.sai.front_end_handler.next_behavior_predict_handler import next_behavior_predict_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.next_behavior_train_handler import next_behavior_train_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.behavior_sequence_describe_handler import behavior_sequence_describe_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.behavior_knowlege_graph_handler import behavior_knowlege_graph_handler

'''
@function
    行为序列和意图识别.
@param 
     behavior_file：识别阶段所使用的行为日志数据，与训练是所使用的数据同属于一类数据(可以是同一个数据)。
'''
def anomaly_detect_predict_main_api(behavior_file):
    # 获取函数参数值
    param_values = get_param_values_from_train_handler()
    field_by = param_values['field_by']
    logid_field = param_values['logid_field']
    behavior_field = param_values['behavior_field']
    time_delta = float(param_values['time_delta'])
    time_field = param_values['time_field']
    interval_standard = float(param_values['interval_standard'])
    para_field = param_values['para_field']
    
    # 识别数据预处理
    predict_data, predict_behavior_logid = behavior_data_preprocess_predict_main_api(behavior_file, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field)
    
    # 基于顺序的行为序列识别
    behavior_sequence_predict_handler()

    # 基于参数值的行为序列识别
    param_value_predic_handler(behavior_file, logid_field, predict_data, predict_behavior_logid)
    del predict_data, predict_behavior_logid
    
    # 业务意图识别
    business_behavior_sequence_predict_handler()

    # 行为意图识别子模型训练
    next_behavior_train_handler()
    # 行为意图识别
    next_behavior_predict_handler()

    # 生成识别结果自然语言描述
    behavior_sequence_describe_handler(behavior_file)

    # 生成辅助知识图谱,若需要，启动该功能
    # behavior_knowlege_graph_handler(behavior_file)

    print('识别完成')




# 测试代码块---接口调用示例
if __name__ == '__main__':
    behavior_file = 'D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211.csv'  # 训练阶段使用的行为日志数据
    anomaly_detect_predict_main_api(behavior_file)
    
    
    
    
    
