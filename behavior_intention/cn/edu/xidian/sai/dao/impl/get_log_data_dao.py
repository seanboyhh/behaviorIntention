'''
Created on Nov 24, 2023

@author: 13507
'''
import json 
import pandas as pd
import os
import chardet

def get_all_log_data(file_path):
    file_extension = os.path.splitext(file_path)[1] # 通过分隔符"."获取文件后缀名
    if file_extension==".log" or file_extension==".json":
        cout=1
        cout2=1
        list1=[]
        for line in open(file_path,"r",encoding="utf-8"):
            content=json.loads(line)
            if "data_tag" in content:
                list1.append(content["data_tag"])
                #print(str(cout)+": "+content["data_tag"])
                cout2=cout2+1
            cout=cout+1
        unique_list = list(set(list1))   
        for node in unique_list:
            print(node)
        print("不重复的data_tag数量为："+str(len(unique_list)))
        print("有data_tag标签的数量为："+str(cout2)+"; "+"占比为："+str(cout2/cout*100)+"%")
    if file_extension==".csv":
        list1=[]
        count=0
        df = pd.read_csv(file_path)
        selected_rows=df[['program_name']]
        data_array=selected_rows.values
        for node in data_array:
            list1.append(node[0])
        unique_list = list(set(list1))
        for node in unique_list:
            count+=1
            print(str(count)+":"+node)
        print("不重复的program_name数量为："+str(count))
        
#获得筛选后和编码后的日志数据以备分类
def get_filtrated_coded_log_data(file_path, field_name):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 指定某些列的某些行,如果需要指定某些行，使用.iloc[a:b]
    selected_rows = df[field_name]
    # 获取选取字段的值
    data_array = selected_rows.values
    print(data_array)
    return data_array

'''
获取文件的编码格式
'''
def get_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    return encoding

#获得csv文件
def csv_or_json_data(file_path):
    encoding = get_file_encoding(file_path)

    file_extension = os.path.splitext(file_path)[1]
    if file_extension==".csv":
        return pd.read_csv(file_path, encoding=encoding)
    elif file_extension==".json":
        return pd.read_json(file_path, encoding=encoding)
    else:
        return None
    