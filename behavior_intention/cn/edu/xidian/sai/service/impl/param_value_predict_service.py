'''
Created on Jan 29, 2024

@author: 13507
'''
import os
import json

from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import csv_or_json_data
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import get_param_values_from_train_dao
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.service.impl.common_tool import training_data_generate, mean_squared_error_modified
from behavior_intention.cn.edu.xidian.sai.service.impl.common_tool import load
from _io import open
import gc

'''
predict_data: 预测集数据
predict_original_data: 原始预测集数据
predict_behavior_logid: 与预测集数据有关的行为-日志id字典
logid_field: 日志id字段名
'''
def param_value_predict(predict_data, predict_original_data, predict_behavior_logid, logid_field):
    # 对predict数据进行异常检测
    model_dict=model_get(predict_data)
    anormal_logid_list = []
    #refer_value=100000
    # 取得异常检测的阈值
    with open(f"{get_parent_path()}/service/impl/tmpdata/refer_value.txt") as f:
        refer_values=json.load(f)

    # 取得偏差区间缩放的比例值
    zoom_value = float(get_param_values_from_train_dao()['zoom_value'])

    for behavior_id, params in predict_data.items():
        # 获取当前行为类别对应参数值的误差
        mses=get_errors(model_dict, behavior_id, params)
        print(f"行为类别：{behavior_id}基于参数值的识别结果：{mses}")
        if(len(mses)>0):
            for i in range(len(mses)):
                # 如果大于阈值，则认为是异常的，这时候溯源到具体日志，保存在df_anormal中
                refer_value = refer_values[str(behavior_id)]
                if mses[i] > zoom_value * refer_value: # 此参照值可以根据实际情况乘以一个实数进行放大和缩小
                    logid = predict_behavior_logid[behavior_id][i+3]
                    anormal_logid_list.append(logid)

    df_abnormal = predict_original_data.loc[predict_original_data[logid_field].isin(anormal_logid_list)]
    df_abnormal2 = csv_or_json_data(f"{get_parent_path()}/service/impl/tmpdata/Anormaly/para_anormal.csv")
    df_abnormal2 = df_abnormal2._append(df_abnormal)
    df_abnormal2.to_csv(f"{get_parent_path()}/service/impl/tmpdata/Anormaly/para_anormal.csv", index=False)

    del model_dict, df_abnormal, df_abnormal2
    gc.collect()

def model_get(predict_data):
    # 对每个日志键进行训练，模型放在model_dict中，eventID为键，model为值
    model_dict = {}
    # 依次加载每个日志键的日志参数组成的矩阵，依次检测
    for behavior_id in predict_data.keys():
        model_file = f"{get_parent_path()}/service/impl/tmpdata/ParamModel/{behavior_id}.h5"
        if os.path.exists(model_file):
            model_dict[behavior_id] = load(model_file)
            continue
    return model_dict
'''
在对预测数据进行检测时，计算误差值，这里的误差为均方差
'''
def get_errors(model_dict, behavior_id, params):
    n_steps = 5
    mses=[]
    # 如果test的日志键还没有模型生成，那么跳过对此日志键的检测
    # 如果检测日志的同日志键的日志条目个数小于n_steps，则跳过检测，因为数据太少，一个滑动窗口都没有
    if (behavior_id in model_dict.keys()) and (len(params) >= n_steps):
        X_predict, Y_predict = training_data_generate(params, n_steps)
        yhat = model_dict[behavior_id].predict(X_predict)
        mses = mean_squared_error_modified(Y_predict, yhat)
    return mses

'''
获得异常检测需要的参照值，当检测时的误差大于这个参照值时就判定为异常，这里的参照值是训练集里最大的误差值或最小值或平均值，
根据实际情况设定
'''
def prefer_value_get(train_data):
    model_dict=model_get(train_data)
    mses_dict={}
    for behavior_id, params in train_data.items():
        mses=get_errors(model_dict, behavior_id, params)
        if(len(mses)>0):
            mses_dict[behavior_id]=max(mses)
    with open(f"{get_parent_path()}/service/impl/tmpdata/refer_value.txt", 'w') as f:
        # 写入参照值
        json.dump(mses_dict, f)

'''
求列表的平均值
'''
def mean(lst):
    if len(lst) >0:
        return sum(lst)/len(lst)
    else:
        return 0
    





    