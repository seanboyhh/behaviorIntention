'''
Created on 2024年4月3日

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.front_end_handler.behavior_sequence_train_handler import behavior_sequence_train_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.param_value_train_handler import param_value_train_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.business_behavior_sequence_mining_handler import business_behavior_sequence_mining_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.business_behavior_sequence_train_handler import business_behavior_sequence_train_handler
from behavior_intention.cn.edu.xidian.sai.main_api.behavior_data_preprocess_main_api import behavior_data_preprocess_train_main_api

'''
@function
    行为意图异常识别模型训练。
@param
    behavior_file: 训练时使用的行为日志文件，格式为csv格式, csv文件需关闭只读模式。
    field_by：依据分割的字段名，格式为列表。本模型提供两种分割方式，（1）按照时间字段和施动者字段进行划分，
            field_by为包含两个元素的列表，第一个字段为时间字段，第二个字段为施动者字段；（2）按照施动者字段
            进行划分，field_by为包含施动者字段的一个元素的列表。
    logid_field：日志唯一id字段名。
    behavior_field：指定行为对应的字段名。
    time_delta：分割的时间间隔，单位为分钟。如果field_by参数为一个元素的列表(即按照第二种方式划分)，time_delta设置为0
    time_field：行为日志中的时间字段名。
    interval_standard：时间间隔标准，即前后两个行为或日志条目时间间隔超过此标准后时间间隔设置为0，单位为秒。
    para_field：一组参数值字段名，格式为列表。即除行为和时间字段以外，可能对异常识别结果有影响的字段。
    max_distance：任意一个业务模式挖掘的最大步长，即任意一个业务模式的最大长度。
    threshold：业务模式挖掘的阈值，即行为序列共现的关联程度。阈值越大，表明挖掘出的业务模式关联程度越强，反之亦然。
    window_size: 滑动窗口大小，数据类型为整型。滑动窗口大小即依据多少个长度的行为序列作为训练和识别的标准，
                建议设置为3-20之间。滑动窗口的大小会直接影响识别效果，请妥善设置。一般情况下，行为类型数量越少
                、完成一个业务的所需要的行为数量越少，滑动窗口大小应该设置比较小，反之亦然。
    zoom_value: 参数值异常训练和识别的偏差值的缩放比例，格式为浮点型。由于训练数据不是纯的正常数据，需要对模型学习到的合理偏差
                进行缩放。zoom_value为小于1的时候，合理的偏差区间进行缩小，参数值异常识别出的异常行为会多，反之亦然。
    top_n: 在基于顺序的行为序列识别中，以预测值中前n个最高概率的预测值为正常标准。类型为int，最小值为1，最大值为行为类别数。
            top_n越大，检测的异常就越少，反之亦然。
    
'''
def anomaly_detect_train_main_api(behavior_file, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, max_distance, threshold, window_size, zoom_value, top_n):
    # 训练数据预处理
    param_value_train_data, _ = behavior_data_preprocess_train_main_api(behavior_file, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n)
    
    # 基于顺序的行为序列识别子模型训练
    behavior_sequence_train_handler()

    # 基于参数值的行为序列识别子模型训练
    param_value_train_handler(param_value_train_data)
    del param_value_train_data

    # 业务意图子模型训练
    business_behavior_sequence_mining_handler(max_distance, threshold)  # 业务模式挖掘
    business_behavior_sequence_train_handler()

    print('训练完成')



# 测试代码块---接口调用示例
if __name__ == '__main__':
    behavior_file = 'D:/data/project/xdProjects/ShanghaiProject/20241218/终端文件操作日志_20241211.csv'  # 训练阶段使用的行为日志数据
    field_by = ["src_account"]
    logid_field = "log_uuid"
    behavior_field = "generic_result_action"
    time_delta = 0  # 时间差值，单位为分钟
    time_field = "datetime"
    interval_standard = 30 * 60  # 单位为秒
    para_field = ["src_account", "src_device_name", "src_device_ip", "src_device_mac", "srcdisktypename",
                  "dstdisktypename"]
    max_distance = 12
    threshold = 0.4
    window_size = 2
    zoom_value = 0.85
    top_n = 8
    anomaly_detect_train_main_api(behavior_file, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, max_distance, threshold, window_size, zoom_value, top_n)
    
    
    
    
    
