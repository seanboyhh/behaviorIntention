'''
Created on Nov 10, 2023

@author: 13507
'''

import json
import csv

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import convert_to_utf8, save_Json_file, csv_sort_values_by_time
from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import get_file_encoding

#字段提取：从dlp.log文件中提取出所需要的字段信息（require_extract_field.txt文件），并写入dlp_process01.csv文件中
#cqzd.txt文件为所需要提取出来的字段名
#后续若需要添加字段信息，只需在require_extract_field.txt文件中添加即可
def extract_log_info(title_file_path, log_file_path, output_csv_path = f'{get_parent_path}/service/impl/tmpdata/DataPreprocess/dlp_process01.csv'):
    try:
        # 从标题行名称文件中读取标题行
        with open(title_file_path, 'r', encoding='utf-8') as title_file:
            # 逐行读取标题，去除空格并添加到标题列表中
            titles = [line.strip() for line in title_file]

        # 打开log文件和CSVimport csv文件
        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)

                # 写入CSV文件的标题行
                csv_writer.writerow(titles)

                # 逐行读取log文件并提取相应的信息写入CSV文件
                for line in log_file:
                    try:
                        # 解析JSON数据
                        log_data = json.loads(line)

                        # 提取感兴趣的信息
                        row_data = [log_data.get(title, "") for title in titles]

                        # 写入CSV文件的一行数据
                        csv_writer.writerow(row_data)

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue

    except FileNotFoundError:
        print(f"File not found: {title_file_path} or {log_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

#创建映射：对提取出字段信息的csv文件选择部分字段创建数字映射（require_encod_field.txt），并将映射保存到result_mapping.json文件中，方便后续调用require_encod_field.txt文件为所需创建映射的字段名。
#后续若需要增加为某一字段名创建数字映射，即可直接增加到require_encod_field.txt文件中。
#若想要固定映射，仅需在result_mapping.json中进行修改
def create_mapping_from_csv_dao(csv_file, titles_to_map):
    # 在行为日志文件中为每个字段值创建数字映射
    mappings = {}
    
    # 获取文件编码格式,并转换成utf-8
    file_encoding = get_file_encoding(csv_file)
    print(f'训练的原始csv文件编码格式为：{file_encoding}')
    if file_encoding!= 'utf-8':
        print('由于原始csv文件编码格式为非utf-8格式，开始转换为utf-8格式')
        convert_to_utf8(csv_file)
    
    # 读取CSV文件内容
    with open(csv_file, 'r', newline='', encoding = 'utf-8') as file:
        reader = csv.DictReader(file)
        csv_content = list(reader)  # 将CSV内容存储在列表中

    for title in titles_to_map:
        values_mapping = {}
        unique_values = list(set(row[title] for row in csv_content))
        for index, value in enumerate(unique_values):
            # 交换键值对的顺序
            values_mapping[index + 1] = value
        mappings[title] = values_mapping

    # 保存编码后的数据
    out_file_path = f'{get_parent_path()}/service/impl/tmpdata/encode_field_mapping.json'
    save_Json_file(out_file_path, mappings)

    del mappings

#数据数字化：读取result_mapping.json映射关系，并按照上一步骤的数字映射，将dlp_process01.csv文件中的数据进行数字化
# 从JSON文件加载result_mapping
def load_mapping_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        mapping = json.load(file)
        return mapping

# 应用映射到CSV文件
def apply_mapping_to_csv(csv_file, time_field, train_or_predict_mark):
    # 获取文件编码格式,并转换成utf-8
    file_encoding = get_file_encoding(csv_file)
    print(f'原始csv文件编码格式为：{file_encoding}')
    if file_encoding != 'utf-8':
        print('由于原始csv文件编码格式为非utf-8格式，开始转换为utf-8格式')
        convert_to_utf8(csv_file)

    # 对原始csv文件按照时间排序
    csv_sort_values_by_time(csv_file, time_field)

    result_mapping = load_mapping_from_file(f'{get_parent_path()}/service/impl/tmpdata/encode_field_mapping.json')
    if train_or_predict_mark == 't':
        output_csv_file = f'{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_train.csv'
    elif train_or_predict_mark == 'p':
        output_csv_file = f'{get_parent_path()}/service/impl/tmpdata/vectorized_behavior_log_predict.csv'
    
    # 用于存储映射后的数据
    mapped_data = []

    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # 针对每一行数据
        for row in reader:
            mapped_row = {}
            # 对于每个标题，应用映射
            for title, value in row.items():
                if title in result_mapping:
                    # 使用映射后的值作为键来查找原始值
                    original_value = next((k for k, v in result_mapping[title].items() if v == value), value)
                    mapped_row[title] = original_value
                else:
                    mapped_row[title] = value  # 如果没有映射，则保留原始值
            mapped_data.append(mapped_row)

    # 写入映射后的数据到输出CSV文件
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=mapped_data[0].keys())
        writer.writeheader()
        writer.writerows(mapped_data)

    del mapped_data

# 从原始的文件中提取所需的字段
def extract_log_filed():
    pass