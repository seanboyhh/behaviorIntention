'''
Created on 2024年4月8日

@author: 13507
'''

from behavior_intention.cn.edu.xidian.sai.service.impl.business_behavior_sequence_mining_service import calculate_behaviors_cooccurrence_degree as cbcd
from behavior_intention.cn.edu.xidian.sai.service.impl.business_behavior_sequence_mining_service import all_business_pattern_mining as abpm

def business_behavior_sequence_mining_handler(max_distance, threshold):
    # 构建行为共现矩阵
    cbcd(max_distance)
    
    # 挖掘行为业务模式
    abpm(max_distance, threshold)


# 测试用例代码块
if __name__ == '__main__':
    max_distance = 10
    threshold = 0.4
    business_behavior_sequence_mining_handler(max_distance, threshold)
    
    