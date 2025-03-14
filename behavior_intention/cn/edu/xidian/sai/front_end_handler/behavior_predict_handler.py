'''
Created on Dec 18, 2023

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import behavior_predict
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import get_filtrated_coded_log_data
from behavior_intention.cn.edu.xidian.sai.dao.impl.extract_log_dao import extract_log_info
from behavior_intention.cn.edu.xidian.sai.dao.impl.extract_log_dao import create_mapping_from_csv_based_on_txt
from behavior_intention.cn.edu.xidian.sai.dao.impl.extract_log_dao import load_mapping_from_file, apply_mapping_to_csv
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import X_predict_uniformize, X_zoom
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_describe_service import single_behavior_description as sbd
import json
import numpy as np



# 测试代码块
if __name__ == '__main__':
    # 函数调用，从测试数据中抽取所需要的字段
    title_file_path = 'D:/data/project/xdProjects/ShanghaiProject/require_extract_field.txt'
    log_file_path = 'D:/data/project/xdProjects/ShanghaiProject/dlp_predict.log'
    output_csv_path = 'D:/data/project/xdProjects/ShanghaiProject/dlp_process_predict01.csv'
    extract_log_info(title_file_path, log_file_path, output_csv_path)
    
    # 函数调用，提取需要向量化的字段，并生成映射表
    result_mapping = create_mapping_from_csv_based_on_txt('D:/data/project/xdProjects/ShanghaiProject/dlp_process_predict01.csv','D:/data/project/xdProjects/ShanghaiProject/require_encod_field.txt')
    print(result_mapping)
    # 保存数字映射到文件
    file_path = 'D:/data/project/xdProjects/ShanghaiProject/encode_field_mapping_predict.json'
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(result_mapping, file, ensure_ascii=False, indent=4)
    
    # 函数调用，生成向量化的特征
    result_mapping = load_mapping_from_file('D:/data/project/xdProjects/ShanghaiProject/encode_field_mapping_predict.json')
    csv_file = 'D:/data/project/xdProjects/ShanghaiProject/dlp_process_predict01.csv'
    output_csv_file = 'D:/data/project/xdProjects/ShanghaiProject/dlp_process_predict02.csv'
    # 应用映射并保存到CSV文件
    apply_mapping_to_csv(csv_file, result_mapping, output_csv_file)
    
    # 待预测数据的地址
    file_path="D:/data/project/xdProjects/ShanghaiProject/dlp_process_predict02.csv"
    # 获取待预测的数据
    X_predict=get_filtrated_coded_log_data(file_path)
    file_path="D:/data/project/xdProjects/ShanghaiProject/dlp_process02.csv"
    # 获取训练集数据
    X_train=get_filtrated_coded_log_data(file_path)
    
    # 数据归一化
    X_predict=X_predict_uniformize(X_train,X_predict)
    # 数据特征放大或缩小
    weight=np.array([1,(2**0.5)])
    X_predict=X_zoom(weight, X_predict)
    
    # 获取模型保存数据的地址
    model_path=get_parent_path()+'\\source\\model1.pkl'
    #输入的数据以备与行为标签组成新的数据
    input_file_path="D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process_predict01.csv,D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process_predict02.csv"
    #生成带标签数据的路径
    out_file_path="D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process_predict03_01.csv,D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process_predict03_02.csv"
    behavior_predict(X_predict, model_path, input_file_path, out_file_path) #预测
    
    # 描述
    #csv文件路径以及存储路径
    file_path="D:/data/project/xdProjects/ShanghaiProject/dlp_process_predict03_01.csv,D:/data/project/xdProjects/ShanghaiProject/single_behavior_description_predict.txt"
    sbd(file_path)
    
    
    
    
    
    