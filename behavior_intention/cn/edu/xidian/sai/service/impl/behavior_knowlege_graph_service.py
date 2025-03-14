'''
Created on Mar 5, 2024

@author: 13507
'''

import json
import pandas as pd
#import random
from rdflib import Graph
# from rdflib. plugins. sparql import prepareQuery
from pyvis.network import Network
from pyvis.options import Layout

from behavior_intention.cn.edu.xidian.sai.service.impl.behavior_classify_service import get_parent_path
from behavior_intention.cn.edu.xidian.sai.dao.impl.behavior_knowlege_graph_dao import get_and_save_triple_from_log
from behavior_intention.cn.edu.xidian.sai.dao.impl.get_log_data_dao import csv_or_json_data
from behavior_intention.cn.edu.xidian.sai.dao.impl.data_process_dao import get_filenames

# 移除命名空间前缀
def remove_namespace(uri, name_space):
    uri = str(uri) if type(uri) != str else uri
    if uri.startswith(name_space):
        return uri.replace(name_space, "", 1)
    return uri

'''
为每一个日志分块（一个完整的任务或同一用户）分别构建相应的三元组.
partition_file_path: 包含划分信息的文件，可以是按照任务划分的，也可以是按照用户划分的.
log_id_field: 原始日志文件中日志唯一id字段名.
'''
def get_and_save_triples_from_log(original_log_file_path, log_id_field, agent_field, behavior_field, parameter_fields):
    partition_file_path = f'{get_parent_path()}/service/impl/tmpdata/SeqData/log_entry_to_behavior_block_predict.csv'
    original_file = csv_or_json_data(original_log_file_path)
    partition_file = csv_or_json_data(partition_file_path)
    for _, row in partition_file.iterrows():
        partition_name = row['block_id']
        
        # 选择同一日志分块的日志数据
        selected_rows = pd.DataFrame(columns=original_file.columns)
        elements = eval(row['log_id_sequence'])
        for element in elements:
            selected_row = original_file[original_file[log_id_field]==element]
            selected_rows = pd.concat([selected_rows, selected_row], axis=0)
        
        # 为日志分块构建三元组
        if not selected_rows.empty:
            get_and_save_triple_from_log(selected_rows, partition_name, agent_field, behavior_field, parameter_fields)
        
'''
使用pyvis生成知识图谱
'''
def generate_knowlege_graph1(triple_file_path, graph_file_name):
    
    # 网络布局
    #layout = Layout()
    # layout.set_edge_minimization(True)
    # layout.set_separation(30)
    
    # 创建网络对象
    net = Network(height="1000px", width="100%", notebook=True, directed=True)
    
    # 定义节点样式
    # node_style = {
    #     "shape": "dot",
    #     "color": "#FFA500"
    # }
     
    # 定义边样式
    # edge_style1 = {
    #     "width": 1,
    #     "arrows": {"to": True},
    #     "smooth": False,
    #     "color": "#70DB93"
    # }
    # edge_style2 = {
    #     "width": 1,
    #     "arrows": {"to": True},
    #     "smooth": False,
    #     "color": "#00FFFF"
    # }
    
    # 加载RDF文件
    g = Graph()
    g.parse(triple_file_path)
    
    # 生成随机颜色列表
    # color_str = "lightgreen yellow lightblue green red gold orange olive darkgreen springgreen yellowgreen darkcyan blue " \
    #             "darkblue lightslateblue mediumpurple cornflowerblue deepskyblue lightcyan lightskyblue "
    
    # 遍历所有主语-谓语-客体关系
    for s, p, o in g:
        # 去掉命名空间
        name_space = "https://www.xidian.edu.cn/"
        s = remove_namespace(s, name_space).replace("---", " ") if "---" in remove_namespace(s, name_space) else remove_namespace(s, name_space)
        s = s.replace("--", "\\") if "--" in s else s
        p = remove_namespace(p, name_space).replace("---", " ") if "---" in remove_namespace(p, name_space) else remove_namespace(p, name_space)
        o = o.replace("---", " ") if "---" in o else o
        
        # 取得分组标记
        s_list = s.split('***')
        s = s_list[0]
        triple_group =s_list[1] 
        
        #color_list = [random.choice(color_str.split())][0]
        # group1为施动者-行为三元组，group2为行为-参数名三元组，group3为参数名-参数值三元组
        if triple_group == "group1":
            # 添加节点
            net.add_node(str(s), color="red")  # , size='10', **node_style
            net.add_node(str(o), color="lightgreen")  # , size='10', **node_style
            
            # 添加边
            net.add_edge(str(s), str(o), label=p)  # , **edge_style
        elif triple_group == "group2":
            net.add_node(str(s), color="lightgreen")
            net.add_node(str(o), color="lightblue")
            
            net.add_edge(str(s), str(o), label=p)
        
        elif triple_group == "group3":
            net.add_node(str(s), color="lightblue")
            net.add_node(str(o), color="lightslateblue")
            
            net.add_edge(str(s), str(o), label=p)
            
    # 控制按钮
    # net.show_buttons(filter_= None)

    # 设置布局算法
    options = {
      "manipulation": {
        "enabled": True,
        "initiallyActive": True
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3650,
          "springLength": 190,
          "springConstant": 0.05,
          "damping": 0.99
        },
        "minVelocity": 0.75
      }
  }
    
    # 装载样式
    options = json.dumps(options)
    net.set_options(options)
    # net.toggle_drag_nodes(True)
    
    # 显示图形化结果并存储图谱
    out_kg_path = get_parent_path() + "/service/impl/tmpdata/KnowlegeGraph/" + graph_file_name + ".html"
    net.show(out_kg_path, notebook=True)
    
    # 返回图谱路径
    return out_kg_path

'''
为每一个三元组生成各自的知识图谱
'''
def generate_knowlege_graphs(tripe_file_directory=get_parent_path() + "/service/impl/tmpdata/BehaviorTriple"):
    knowlege_graphs = []
    
    # 获取指定目录下的所有文件的文件名
    filenames = get_filenames(tripe_file_directory)
    
    for filename in filenames:
        triple_file_path = tripe_file_directory+"/"+filename
        filename = filename.split('.')[0]
        knowlege_graph = generate_knowlege_graph1(triple_file_path, filename)
        knowlege_graphs.append(knowlege_graph)
    
    # 返回图谱的路径
    return knowlege_graphs
  





