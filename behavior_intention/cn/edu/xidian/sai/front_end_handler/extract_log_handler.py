'''
Created on Nov 10, 2023

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.dao.impl.extract_log_dao import extract_log_info

def extract_log_handler(title_file_path, log_file_path):
    # 函数调用
    extract_log_info(title_file_path, log_file_path)

if __name__ == '__main__':
    extract_log_handler('D:/data/project/xdProjects/ShanghaiProject/require_extract_field.txt', 'D:/data/project/xdProjects/ShanghaiProject/dlp.log')
    