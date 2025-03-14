'''
Created on Nov 10, 2023

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.dao.impl.extract_log_dao import apply_mapping_to_csv

def vectorize_log_handler(csv_file, time_field, train_or_predict_mark):
    #功能： 特征向量化
    # 调用函数
    # 加载映射
    
    # 应用映射并保存到CSV文件
    apply_mapping_to_csv(csv_file, time_field, train_or_predict_mark)


# 测试用例代码块
if __name__ == '__main__':
    # csv_file: 行为日志文件
    csv_file = 'D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211-10w.csv'
    time_field = 'datetime'
    train_or_predict_mark = 'p'
    vectorize_log_handler(csv_file, time_field, train_or_predict_mark)
    
