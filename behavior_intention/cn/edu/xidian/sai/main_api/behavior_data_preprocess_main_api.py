'''
Created on 2024年9月23日

@author: 13507
'''

import os

# from behavior_intention.cn.edu.xidian.sai.front_end_handler.extract_log_handler import extract_log_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.encode_log_handler import encode_log_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.vectorize_log_handler import vectorize_log_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.data_process_handler import data_divide_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.data_process_handler2 import generate_param_value_handler
from behavior_intention.cn.edu.xidian.sai.front_end_handler.data_process_handler import save_param_values_from_train_handler
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path

# 训练数据预处理模块
def behavior_data_preprocess_train_main_api(behavior_file, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n):
    train_or_predict_mark = 't'

    # 创建中间文件的存放目录
    tmpdata_path = ["Anormaly", "BehaviorTriple", "BusinessLog", "BussinessModel", "BusinessPattern", "CooccurrenceMatrix"
                    , "DataPreprocess", "EventNpy", "ExecutePathModel", "IntermediateFile", "KnowlegeGraph",
                    "NextBehaviorBasedData", "ParamData", "ParamModel", "SeqData","SeqLog", "SeqModel", "SpellResult",
                    "struct", "TmpLog"]
    for path in tmpdata_path:
        if not os.path.exists(f"{get_parent_path()}/service/impl/tmpdata/{path}"):
            os.makedirs(f"{get_parent_path()}/service/impl/tmpdata/{path}")

    # 保存函数的参数值
    save_param_values_from_train_handler(field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field, window_size, zoom_value, top_n)
    
    '''
    @function:
        提取原始日志中需要的字段并保持为csv格式文件，具体字段由用户指明。如果需要此功能模块，把相应的代码注释去掉即可。
    @param:
        title_file_path: 用户指定的需要提前的字段名文件路径，格式为.txt,文件中每一行为一个字段名。
        log_file_path: 原始日志文件，格式为.log，类似于json格式。
        output_csv_path: 提取后文件路径，若不使用，会启用默认路径。
    '''
    # title_file_path = ''
    # log_file_path = ''
    # extract_log_handler(title_file_path, log_file_path)
    
    '''
    @function:
        对行为日志数据进行编码（特征向量化）。
    @param: 
        behavior_file: 行为日志文件，格式为csv格式。
        require_encode_field: 需要编码的字段，即需要参与运算的字段，由调用者指定，文件类型为txt格式，每一行对应一个指定的字段名。
    '''
    # 编码
    encode_log_handler(behavior_file, behavior_field)

    # 向量化
    vectorize_log_handler(behavior_file, time_field, train_or_predict_mark)
    
    '''
    @function:
        对编码后的行为日志数据按照时间进行分割，生成行为序列异常训练需要的数据。
    @param:
        field_by：依据分割的时间字段名。
        logid_field：日志唯一id字段名。
        behavior_field：指定行为对应的字段名。
        time_delta：分割的时间间隔，单位为分钟。
    '''
    data_divide_handler(field_by, logid_field, behavior_field, time_delta, train_or_predict_mark)
    
    '''
    @function:
        生成参数值异常训练需要的数据。
    @param:
        time_field：行为日志中的时间字段名
        interval_standard：时间间隔标准，即前后两个行为或日志条目时间间隔超过此标准后时间间隔设置为0，单位为秒
        behavior_field：指定行为对应的字段名。
        logid_field：日志唯一id字段名。
        train_or_predict_mark：指明生成的数据是训练数据还是预测数据，t代表训练数据，p代表预测数据。
        para_field：一组参数值字段名，格式为列表。
    '''
    return generate_param_value_handler(behavior_file, time_field, interval_standard, behavior_field, logid_field, train_or_predict_mark, para_field)

# 预测数据预处理模块
def behavior_data_preprocess_predict_main_api(behavior_file, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field):

    train_or_predict_mark = 'p'

    #数据向量化
    vectorize_log_handler(behavior_file, time_field, train_or_predict_mark)

    # 数据分割
    data_divide_handler(field_by, logid_field, behavior_field, time_delta, train_or_predict_mark)

    return generate_param_value_handler(behavior_file, time_field, interval_standard, behavior_field, logid_field, train_or_predict_mark, para_field)
    
# 测试代码块---调用示例
# if __name__ == '__main__':
#     # 训练数据预处理
#     behavior_file = 'D:/data/project/xdProjects/ShanghaiProject/20231229/security_log_202312290928.csv' # 训练阶段使用的行为日志数据
#     require_encode_field = 'D:/data/project/xdProjects/ShanghaiProject/20231229/require_encod_field.txt'
#     field_by="log_create_time"
#     logid_field="uuid"
#     behavior_field="program_name"
#     time_delta=30 # 时间差值，单位为分钟
#     time_field = "log_create_time"
#     interval_standard = 30*60 # 单位为秒
#     para_field = ["http_httpagent", "http_response_size", "http_httpmethod", "http_url_externalurl", "src_device_ip", "http_httpstatus"]
#     behavior_data_preprocess_train_main_api(behavior_file, require_encode_field, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field)
    
    # 预测数据预处理，除了预测数据外，其他参数设置与训练阶段的参数保持一致
    # behavior_file = 'D:/data/project/xdProjects/ShanghaiProject/20231229/security_log_202312290928.csv' # 训练阶段使用的行为日志数据
    # require_encode_field = 'D:/data/project/xdProjects/ShanghaiProject/20231229/require_encod_field.txt'
    # field_by="log_create_time"
    # logid_field="uuid"
    # behavior_field="program_name"
    # time_delta=30 # 时间差值，单位为分钟
    # time_field = "log_create_time"
    # interval_standard = 30*60 # 单位为秒
    # para_field = ["http_httpagent", "http_response_size", "http_httpmethod", "http_url_externalurl", "src_device_ip", "http_httpstatus"]
    # behavior_data_preprocess_predict_main_api(behavior_file, field_by, logid_field, behavior_field, time_delta, time_field, interval_standard, para_field)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    