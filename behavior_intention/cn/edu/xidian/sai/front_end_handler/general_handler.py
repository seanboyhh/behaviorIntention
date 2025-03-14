'''
Created on Dec 18, 2023

@author: 13507
'''

import numpy as np
#import torch
# from cn.edu.xidian.sai.service.impl.behavior_classify import uniformize
# from cn.edu.xidian.sai.service.impl.behavior_classify import X_predict_uniformize
# from cn.edu.xidian.sai.dao.impl.get_log_data_dao_impl import get_filtrated_coded_log_data
# from cn.edu.xidian.sai.service.impl.behavior_classify import X_zoom
#from sklearn import preprocessing as pp
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
#import pickle
import platform
import tensorflow as tf
import sys
import pandas as pd



# 测试代码块
if __name__ == '__main__':
    out_file_path = f"{get_parent_path()}/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_predict.csv"
    df = pd.read_csv(out_file_path)
    print(df["behavior_sequence"])

    # data = np.random.normal(0, 1, 100)
    # print(data)
    # print(type(data))
    #tf.test.gpu_device_name()
    # print(f"python路径为： {sys.executable}")
    # print(platform.python_version())
    # print(tf.__version__)
    # gpus = tf.config.list_physical_devices('GPU')
    # print(gpus)
    # print(tf.test.gpu_device_name())
    # print(tf.config.experimental.set_visible_devices)
    # print('GPU:', tf.config.experimental.list_physical_devices('GPU'))
    # print('CPU:', tf.config.experimental.list_physical_devices(device_type='CPU'))
    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.test.is_gpu_available())
    # # 输出可用的GPU数量
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # print(tf.version)
    # enc = pp.OneHotEncoder()———
    # X=np.array([['张三','男','工程师'],
    #            ['李四','男','经济师'],
    #            ['小红','女','教师'],
    #            ['王华','女','税务师']]
    #     )
    # enc.fit(X)
    # res=enc._get_tags()
    # result=enc.transform(X).toarray()
    # print(result)
    # print(enc.transform(np.array([['李四','男','经济师']])).toarray())
    # # 存储模型结果
    # with open(get_parent_path()+'\\source\\one_hot1.pkl', 'wb') as f:
    #     pickle.dump(enc, f)
    #     f.close()
    # # 加载存储的模型
    # with open(get_parent_path()+'\\source\\one_hot1.pkl', 'rb') as f:
    #     one_hot1=pickle.load(f)
    #     f.close()
    # print(one_hot1.transform(np.array([['李四','男','经济师']])).toarray())
    # file_path="D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process02.csv"
    # field_name=['log_priority', 'generic_opt_content', 'src_person_status', 'src_person_ctpositionname', 'src_person_type', 'src_device_dept', 'data_level']
    # X_train=get_filtrated_coded_log_data(file_path, field_name)
    # X_test=numpy.array([[4,4,4,4,4,4,4],
    #                     [20,20,20,20,20,20,20]
    #                         ])
    # X_train_test=numpy.append(X_train,X_test, axis=0)# 纵向按行附加
    # i=X_train.shape[0]; j=X_test.shape[0]# 获取矩阵行数
    # get_X_test=X_train_test[i:i+j] 
    # print(f'训练集数据和测试集数据上下拼接的结果为：\n {X_train_test}')
    # print(f'从拼接的数据集取出测试集的结果为：\n {get_X_test}')
    # print(f'这是训练集归一化后的结果：\n {uniformize(X_train)}')
    # print(f'合并数据归一化的结果为： \n {uniformize(X_train_test)}')
    # print(f'从合并数据中获取测试集归一化的结果为：\n {X_predict_uniformize(X_train,X_test)}')
    # X=numpy.array([[1,2,3,4,],
    #                 [5,6,7,8],
    #                 [10,11,12,13],
    #                 [14,15,16,17]]).astype(float)
    # weight=numpy.array([1,1,(2**0.5)])
    # print(X_zoom(weight, X))
    
    
    
    
    
    