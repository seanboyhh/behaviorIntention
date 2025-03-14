'''
Created on Nov 25, 2023

@author: 13507
'''

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing as pp
import pandas as pd
import pickle
import os
from sklearn.neural_network import MLPClassifier

class Kmeans:
    def __init__(self, k=2, tolerance=0.01, max_iter=300):
        self.k = k
        self.tol = tolerance
        self.max_iter = max_iter
        self.features_count = -1
        self.classifications = None
        self.centroids = None

    def fit(self, data):
        """
        :param data: numpy数组，约定shape为：(数据数量，数据维度)
        :type data: numpy.ndarray
        """
        self.features_count = data.shape[1]
        # 初始化聚类中心（维度：k个 * features种数）
        self.centroids = np.zeros([self.k, data.shape[1]])
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # 清空聚类列表
            self.classifications = [[] for i in range(self.k)]
            # 对每个点与聚类中心进行距离计算
            for feature_set in data:
                # 预测分类
                classification = self.predict(feature_set)
                # 加入类别
                self.classifications[classification].append(feature_set)

            # 记录前一次的结果
            prev_centroids = np.ndarray.copy(self.centroids)

            # 更新中心
            for classification in range(self.k):
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # 检测相邻两次中心的变化情况
            for c in range(self.k):
                if np.linalg.norm(prev_centroids[c] - self.centroids[c]) > self.tol:
                    break

            # 如果都满足条件（上面循环没break），则返回
            else:
                return

    def predict(self, data):
        # 距离
        distances = np.linalg.norm(data - self.centroids, axis=1)
        # 最小距离索引
        return distances.argmin()

def blobs(n_samples=300, n_features=2, centers=1, cluster_std=0.60, random_state=0):
    points, _ = make_blobs(n_samples=n_samples,
                           n_features=n_features,
                           centers=centers,
                           cluster_std=cluster_std,
                           random_state=random_state)
    return points

# 数据归一化处理
def uniformize(X):
    #MinMax方式
    min_max_scaler = pp.MinMaxScaler()
    X_min_max = min_max_scaler.fit_transform(X)# 按列归一化
    return X_min_max

# 对新的测试(预测)集数据进行动态归一化处理
def X_predict_uniformize(X_train, X_predict):
    X_train_predict=np.append(X_train,X_predict, axis=0)# 合并两个数据集，纵向按行附加
    i=X_train.shape[0]; j=X_predict.shape[0]# 获取矩阵行数
    # 对合并的数据集进行归一化，并取回归一化后的测试（预测）集数据
    get_X_predict=uniformize(X_train_predict)[i:i+j]
    return get_X_predict

# 对数据集进行的某些特征值进行放大或缩小
def X_zoom(weight, X):
    # weight: 为数组，指明数据集X中各个特征是放大还是缩小
    for i, element in enumerate(weight):
        X[:, i] *= element
    return X

#获得目录
def get_parent_path():
    current_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
    return parent_dir

#获得行为分类标签并存入
def get_classification_label(X, input_file_path, out_file_path):
    # 创建K-Means对象
    kmeans = KMeans(n_clusters=2)  # 设置聚类簇数为2
    # 训练K-Means模型
    kmeans.fit(X)
    # 存储模型结果
    with open(get_parent_path()+'\\source\\model1.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        f.close()
    # 根据聚类结果将行为分类
    behavior_labels = kmeans.labels_.tolist()
    input_file=input_file_path.split(","); out_file=out_file_path.split(",")
    input_file0=input_file[0]; input_file1=input_file[1]
    outfile0=out_file[0]; outfile1=out_file[1]
    df=pd.read_csv(input_file0)
    # 在新列中添加数据
    df['authority_label'] = behavior_labels
    # 保存修改后的CSV文件
    df.to_csv(outfile0, index=False)
    df=pd.read_csv(input_file1)
    df['authority_label'] = behavior_labels
    df.to_csv(outfile1, index=False)
    print(f"行为授权标签为：{behavior_labels}")

# 对新行为的分类进行预测
def behavior_predict(X, model_path, input_file_path, out_file_path):
    # 加载存储的模型
    with open(get_parent_path()+'\\source\\model1.pkl', 'rb') as f:
        model1=pickle.load(f)
        f.close()
        # 根据聚类结果将行为分类
        behavior_labels = model1.predict(X).tolist() # 如果需要可以设置参数sample_weight
        input_file=input_file_path.split(","); out_file=out_file_path.split(",")
        input_file0=input_file[0]; input_file1=input_file[1]
        outfile0=out_file[0]; outfile1=out_file[1]
        df=pd.read_csv(input_file0)
        # 在新列中添加数据
        df['authority_label'] = behavior_labels
        # 保存修改后的CSV文件
        df.to_csv(outfile0, index=False)
        df=pd.read_csv(input_file1)
        df['authority_label'] = behavior_labels
        df.to_csv(outfile1, index=False)
        print(behavior_labels)


# 分类器
def classify(X, y):
    clt=MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(50,100),random_sate=1)
    

def kmeans_plot(kmeans_model):
    """
    简单可视化2d或3d kmeans聚类结果，不是算法必须的，直接使用即可。

    :param kmeans_model: 训练的kmeans模型
    :type kmeans_model: Kmeans | FastKmeans
    """
    plt.style.use('ggplot')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # 2D
    if kmeans_model.features_count == 2:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot()

        for i in range(kmeans_model.k):
            color = colors[i%len(colors)]

            for feature_set in kmeans_model.classifications[i]:
                ax.scatter(feature_set[0], feature_set[1], marker="x", color=color, s=50, linewidths=1)

        for centroid in kmeans_model.centroids:
            ax.scatter(centroid[0], centroid[1], marker="o", color="k", s=50, linewidths=3)
    # 3D
    elif kmeans_model.features_count == 3:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        for i in range(kmeans_model.k):
            color = colors[i%len(colors)]

            for feature_set in kmeans_model.classifications[i]:
                ax.scatter(feature_set[0], feature_set[1], feature_set[2], marker="x", color=color, s=50, linewidths=1)

        for centroid in kmeans_model.centroids:
            ax .scatter(centroid[0], centroid[1], centroid[2], marker="o", color="k", s=50, linewidths=3)
    plt.show()


