# coding=utf-8
# /usr/bin/env python3

import sys

from behavior_intention.cn.edu.xidian.sai.service.impl.param_value_predict_service import param_value_predict
#from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import csv_or_json_data
from behavior_intention.cn.edu.xidian.sai.front_end_handler.data_process_handler2 import generate_param_value_handler

sys.setrecursionlimit(2000)

def param_value_predic_handler(behavior_file, logid_field, predict_data, predict_behavior_logid):
    # 提取未编码的检测原始数据
    predict_original_data=csv_or_json_data(behavior_file)
    
    # 基于参数值的行为序列识别
    print("基于参数值的行为序列识别")
    param_value_predict(predict_data, predict_original_data, predict_behavior_logid, logid_field)



# 测试用例代码块
if __name__ == '__main__':
    time_field = "datetime"
    interval_standard = 30*60 # 单位为秒
    behavior_file='D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211-10w.csv'
    behavior_field="generic_result_action"
    logid_field="log_uuid"
    para_field=["src_account", "src_device_name", "src_device_ip","src_device_mac", "srcdisktypename", "dstdisktypename"]
    train_or_predict_mark = 'p'
    # 提取predict
    print("提取predict数据")
    predict_data, predict_behavior_logid = generate_param_value_handler(behavior_file, time_field, interval_standard, behavior_field, logid_field, train_or_predict_mark, para_field)

    param_value_predic_handler(behavior_file, logid_field, predict_data, predict_behavior_logid)












