'''
Created on 2024年4月1日

@author: 13507
'''

from rdflib import Graph, Namespace, Literal, URIRef
from sympy.codegen.ast import continue_

#from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import csv_or_json_data
from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path

'''
从日志文件中提取三元组并存储
log_id_field: 日志条目唯一id
'''
def get_and_save_triple_from_log(log_data, partition_name, agent_field, behavior_field, parameter_fields):
    name_space = "https://www.xidian.edu.cn/"
    ex = Namespace(name_space)
    g = Graph()
    
    for _, row in log_data.iterrows():
        # 指名第几组三元组
        triple_group = "group1"
        
        # 施动者-行为三元组
        subject = str(row[agent_field]).replace(" ", "---") if " " in str(row[agent_field]) else str(row[agent_field])
        subject = subject.replace("\\", "--") if "\\" in subject else subject
        subject = subject + '***' + triple_group
        predicate = "does"
        object = str(row[behavior_field]).replace(" ", "---") if " " in str(row[behavior_field]) else str(row[behavior_field])
        g.add((URIRef(ex[subject]), ex[predicate], Literal(object)))
        
        # 行为-参数名三元组，参数名-参数值三元组
        triple_group = "group2"
        triple_group1 = "group3"
        for parameter_field in parameter_fields:
            # 行为-参数名三元组
            subject = str(row[behavior_field]).replace(" ", "---") if " " in str(row[behavior_field]) else str(row[behavior_field])
            subject = subject + '***' + triple_group
            predicate = "has"
            object = str(parameter_field).replace(" ", "---") if " " in str(parameter_field) else str(parameter_field)
            g.add((URIRef(ex[subject]), ex[predicate], Literal(object)))
            
            # 参数名-参数值三元组
            subject = object + '***' + triple_group1
            predicate = "value_as"
            object = str(row[parameter_field]).replace(" ", "---") if " " in str(row[parameter_field]) else str(row[parameter_field])
            g.add((URIRef(ex[subject]), ex[predicate], Literal(object)))
            
    
    # 序列化存储
    triple_file_name = str(partition_name)+".ttl"
    with open(get_parent_path() + "/service/impl/tmpdata/BehaviorTriple/"+triple_file_name, 'wb') as f:
        try:
            g.serialize(f, format="turtle")
        except Exception as e:
            print(e)
            pass
    # query = prepareQuery('SELECT ?s ?p ?o WHERE {?s ?p ?o}')




