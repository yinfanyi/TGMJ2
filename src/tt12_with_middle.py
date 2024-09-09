"""适用topology_TSR_flexible_strut_ball_foot_v1.xlsx，相比之前版本，xlsx的表单名字改了

"""

import xml.etree.ElementTree as ET
from typing import Optional,Dict,Optional, List
import re
import ast

import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt

# vscode可以在settings.json将src的文件添加到路径，使编译器可以识别
# "python.analysis.extraPaths": [
#         "./src"
#     ]

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

from tensegrity import Tensegrity
from xlsxConverter import XLSXConverter

class TT12WithMiddle(Tensegrity):
    def __init__(self, xlsx_path, param_length_1=0.7, param_length_2=0.35, param_length_3=0, param_length_4=0.7, radius=0.2):        

        self.radius = radius
        xlsxConverter = XLSXConverter(xlsx_path)
        N = xlsxConverter.get_Node_array(param_length_1=param_length_1, 
                                         param_length_2=param_length_2, 
                                         param_length_3=param_length_3,
                                         param_length_4=param_length_4)  # param_length_4=self.radius*2
        Cb_in = xlsxConverter.get_Cb_in_array()
        Cs_in = xlsxConverter.get_Cs_in_array(sheet_name='cables_strut_to_strut')

        self.alt_node_label_list = xlsxConverter.get_alt_node_label_list()

        self.TT_mid_connected_array = xlsxConverter.get_TT_mid_element_connected_array(sheet_name='cables_strut_to_sphere')
        self.sphere_central_node_label_list = xlsxConverter.get_sphere_central_node_label_list()
        self.strut2strut_string_stiffness_type = xlsxConverter.get_TT_strut2strut_string_stiffness_type(sheet_name='cables_strut_to_strut')
        super().__init__(N, Cb_in, Cs_in)
        self.move_nodes(0, 0, 1)
    
    def generate_shpere_element(self, mass, density, radius, middle_node_label:str, connected_node_label_list:Optional[List[str]]=None, label:Optional[str]=None):
        return self.create_item('Ball', 
                                {'mass':mass, 'density':density, 
                                 'radius':radius,
                                 'middle_node_label':middle_node_label, 
                                'connected_node_label_list':connected_node_label_list, 
                                'label':label})
    
    # @overload 增加一个栏目，用于写中点的位置
    def generate_Bar_element(self, mass, density, from_node_label=None, to_node_label=None, alt_node_label:Optional[str] = None, label:Optional[str]=None):
        return self.create_item('Bar', 
                                {'mass':mass, 'density':density, 
                                 'from_node_label':from_node_label,
                                 'alt_node_label': alt_node_label,
                                 'to_node_label':to_node_label, 'label':label})

    # @overload 有了中点
    def fill_Bar_data(self):
        if self.C_b is None or self.C_s is None or self.N is None:
            print('数据不完全')
            return False
        B = np.dot(self.N, self.C_b.T)
        for j in range(B.shape[1]):
            x_str= str(np.argmax(self.C_b[j, :] == -1))
            y_str = str(np.argmax(self.C_b[j, :] == 1))
            self.add(self.generate_Bar_element(mass=0.08, density=0,
                                            from_node_label=f'node_{x_str}',
                                            to_node_label=f'node_{y_str}',
                                            alt_node_label=self.alt_node_label_list[j],
                                            label=f'bar_{x_str}_{y_str}'))
    
    # @overload 张拉整体结构的弹簧刚度有两种
    def fill_String_data(self, stiffness1, stiffness2):
        if self.C_b is None or self.C_s is None or self.N is None:
            print('数据不完全')
            return False
        
        S = np.dot(self.N, self.C_s.T)
        for j in range(S.shape[1]):
            if self.strut2strut_string_stiffness_type[j] == 1:
                stiffness = stiffness1
                color = '1 0.5 0.5 0.7'
                ori_length = 0.22
            else:
                stiffness = stiffness2
                color = '0.7 0 0 0.7'
                ori_length = 0.10
            self.add(self.generate_String_element(stiffness=stiffness,
                                                ori_length=ori_length,
                                                color=color,
                                                from_node_label=f'node_{np.argmax(self.C_s[j, :] == -1)}',
                                                to_node_label=f'node_{np.argmax(self.C_s[j, :] == 1)}',
                                                label=f'string_{np.argmax(self.C_s[j, :] == -1)}_{np.argmax(self.C_s[j, :] == 1)}'))
        
    def fill_TT_Mid_connected_String_data(self):
        for i in range(self.TT_mid_connected_array.shape[0]):
            self.add(self.generate_String_element(stiffness=67000,
                                                ori_length=0.17,
                                                color='1 1 0 0.7',
                                                from_node_label=f'node_{self.TT_mid_connected_array[i,0]-1}',
                                                to_node_label=f'node_{self.TT_mid_connected_array[i,1]-1}',
                                                label=f'string_{self.TT_mid_connected_array[i,0]-1}_{self.TT_mid_connected_array[i,1]-1}'))

    def find_node_position(self, df, label):
        # node_row = df[(df['Type'] == 'Node') & df['Data'].apply(lambda x: ast.literal_eval(x)['label'] == label)]
        node_row = df[(df['Type'] == 'Node')]
        for _, row in node_row.iterrows():
            data_dict = row['Data']
            if data_dict.get('label') == label:
                return data_dict.get('position')
        else:
            return None

    def get_midpoint_coords(self, coord1, coord2):
        return [(coord1[i] + coord2[i]) / 2 for i in range(3)]
    
    def get_mid_element_Node_list(self):
        mid_element_node_label_list = []
        for i in range(self.TT_mid_connected_array.shape[0]):
            mid_element_node_label_list.append(f'node_{self.TT_mid_connected_array[i,1]-1}')
        return list(set(mid_element_node_label_list))

    def fill_Ball_data(self):
        mid_element_node_label_list = self.get_mid_element_Node_list()
        self.add(self.generate_shpere_element(mass=0.4, density=0, radius=self.radius,
                                            middle_node_label=self.sphere_central_node_label_list[0],
                                            connected_node_label_list=mid_element_node_label_list,
                                            label='Outside_sphere'))

        self.add(self.generate_shpere_element(mass=6.6, density=0, radius=0.08,
                                            middle_node_label=self.sphere_central_node_label_list[1],
                                            label='Inside_sphere'))
    
    def fill_all_data(self):
        super().fill_Node_data()
        self.fill_Bar_data()
        self.fill_String_data(stiffness1=771, stiffness2=2000)
        self.fill_TT_Mid_connected_String_data()
        self.fill_Ball_data()