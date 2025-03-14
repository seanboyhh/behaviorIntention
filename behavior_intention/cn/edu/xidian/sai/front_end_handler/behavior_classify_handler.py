'''
Created on Nov 25, 2023

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_classification_label#, kmeans_plot
from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import get_filtrated_coded_log_data
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import uniformize, X_zoom


# 测试用例代码块
if __name__ == '__main__':
    file_path="D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process02.csv"
    #输入的数据以备与行为标签组成新的数据
    input_file_path="D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process01.csv,D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process02.csv"
    #生成带标签数据的路径
    out_file_path="D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process03_01.csv,D:\\data\\project\\xdProjects\\ShanghaiProject\\dlp_process03_02.csv"
    X_train=get_filtrated_coded_log_data(file_path)# 获得向量化后的训练集数据
    X_train=uniformize(X_train)# 对数据归一化处理
    weight=[1, 2**0.5]# 设置特征的权重，是放大还是缩小
    X_train=X_zoom(weight, X_train)
    get_classification_label(X_train,input_file_path,out_file_path)# 分类
    #model = Kmeans(k=2)
    #model.fit(filtrated_coded_log_data)
    #kmeans_plot(model)
    # model.fit(blobs(centers=3, random_state=1, n_features=2))
    # kmeans_plot(model)
    # model.fit(blobs(centers=3, random_state=3, n_features=3))
    # kmeans_plot(model)
    