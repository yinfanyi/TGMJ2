from typing import List

import pandas as pd
import numpy as np

class XLSXConverter():
    def __init__(self, xlsx_path) -> None:
        self.xlsx_path = xlsx_path

    def get_Node_array(self, **kwargs)->np.array:
        df = pd.read_excel(self.xlsx_path, sheet_name='nodes')
        N = df[['X', 'Y', 'Z']].to_numpy()
        N_eval = np.array([[eval(expr, globals(), kwargs) for expr in row] for row in N])
        return N_eval.T

    def get_Cb_in_array(self, sheet_name='struts')->np.array:
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        Cb_in = df[['imarker', 'jmarker']].to_numpy()
        return Cb_in

    def get_Cs_in_array(self, sheet_name='cables1')->np.array:
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        Cs_in = df[['imarker', 'jmarker']].to_numpy()
        return Cs_in
    
    def get_cables_type_list(self, sheet_name='cables1'):
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        name_and_types = df[['imarker', 'jmarker', 'type']].to_numpy()
        cables_type_list = []
        for i in range(name_and_types.shape[0]):
            imarker = name_and_types[i, 0]
            jmarker = name_and_types[i, 1]
            type = name_and_types[i,2]
            cables_type_list.append([f'string_{imarker-1}_{jmarker-1}', type])
        return cables_type_list
    
    # 获取每个杆中间的节点的名称，目前是每个杆都有中间的节点，如果每个杆没有中间节点，该如何写？
    def get_alt_node_label_list(self, sheet_name='struts')->List:
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        kmarker_list = df['kmarker'].to_list()
        alt_node_label_list = []
        for kmarker in kmarker_list:
            alt_node_label_list.append(f'node_{kmarker-1}')
        return alt_node_label_list
    
    def get_sphere_central_node_label_list(self, sheet_name='spheres'):
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        centermarker_list = df['centermarker'].to_list()
        sphere_central_node_label_list = []
        for centermarker in centermarker_list:
            sphere_central_node_label_list.append(f'node_{centermarker-1}')
        return sphere_central_node_label_list
    
    def get_TT_mid_element_connected_array(self, sheet_name='cables2'):
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        coonected_array = df[['imarker', 'jmarker']].to_numpy()
        return coonected_array
    
    def get_TT_strut2strut_string_stiffness_type(self, sheet_name='cables1'):
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        stiffness_type_list = df['type'].tolist()
        return stiffness_type_list
    
if __name__ == "__main__":
    xlsx_path = './data/xlsx/topology_TSR_flexible_strut_ball_foot_v1_R.xlsx'
    xlsxConverter = XLSXConverter(xlsx_path)
    # print(xlsxConverter.get_Node_array(param_length_1=1, param_length_2=1, param_length_3=1))  
    # print(xlsxConverter.get_Cb_in_array())    
    # print(xlsxConverter.get_Cs_in_array())    
    # print(xlsxConverter.get_alt_node_label_list())
    # print(xlsxConverter.get_TT_mid_element_connected_array())
    a=xlsxConverter.get_TT_strut2strut_string_stiffness_type(sheet_name='cables_strut_to_strut')
    print(a)
