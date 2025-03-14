'''
Created on Jan 26, 2024

@author: 13507
'''

import sys
import os
import time

from behavior_intention.cn.edu.xidian.sai.service.impl.param_value_train_service import param_value_train
from behavior_intention.cn.edu.xidian.sai.service.impl.param_value_predict_service import prefer_value_get
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.front_end_handler.data_process_handler2 import generate_param_value_handler 

sys.setrecursionlimit(2000)

def param_value_train_handler(train_data):
    # train_log_file = get_parent_path()+"/service/impl/logdata/Linux/train.log"
    # test_log_file = get_parent_path()+"/service/impl/logdata/Linux/train.log"

    # 创建中间文件的存放目录
    tmpdata_path = ["struct", "EventNpy", "SpellResult", "ParamData", "ParamModel", "ExecutePathModel"]
    for path in tmpdata_path:
        if not os.path.exists(f"{get_parent_path()}/service/impl/tmpdata/{path}"):
            os.makedirs(f"{get_parent_path()}/service/impl/tmpdata/{path}")

    # 提取未编码的检测原始数据
    # predict_original_data=csv_or_json_data(predict_original_input_file_path)
    
    # 参数值向量异常检测
    print("参数值向量异常训练")
    start_time = time.time()
    param_value_train(train_data)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time}秒")
    #写入参照值
    print("写入参照值")
    prefer_value_get(train_data)
    #param_value(df_train_log, df_test_log)


# 测试用例代码块
if __name__ == '__main__':
    behavior_file = 'D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211.csv'
    time_field = "datetime"
    interval_standard=30*60 # 单位为秒
    behavior_field="generic_result_action"
    logid_field="log_uuid"
    para_field = ["src_account", "src_device_name", "src_device_ip","src_device_mac", "srcdisktypename", "dstdisktypename"]
    train_or_predict_mark = 't'
    trian_data, _ = generate_param_value_handler(behavior_file, time_field, interval_standard, behavior_field, logid_field, train_or_predict_mark, para_field)
    
    param_value_train_handler(trian_data)
    
    
    
    
    
    
    
    
    
    
    
    