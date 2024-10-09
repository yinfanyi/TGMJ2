
import re
import ast
import os
import sys
import time 
import xml.etree.ElementTree as ET
from typing import List, Optional
import multiprocessing  
from multiprocessing import Pool  
import random
import json

import pandas as pd
import pickle
import numpy as np
import sympy as sp
import keyboard
import matplotlib.pyplot as plt
import nevergrad as ng
from concurrent import futures

import mujoco.viewer as mj_viewer
import mujoco as mj
from dm_control import mjcf

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

def dis(n1:np.array, n2:np.array):
    """
    :param n1: node1
    :param n2: node2
    :return: distance of two nodes
    """
    d = np.sqrt(np.dot(n1-n2, n1-n2))
    return d

def process_equilibrium(lx, ly, lz, a, z, f1, f21, f31, f41, version="v8"):
    """
    :param lx: 前进方向杆长度的一半
    :param ly: 高度方向杆长度的一半
    :param lz: 屏幕法向杆长度的一半
    :param a:  杆间距的一半
    :param z:  连接点的位置
    :param f1: type1 弹簧力大小
    :param f21: type2 弹簧力大小 (21)
    :param f31: type5 弹簧力大小 (31)
    """
    # unknown variables
    if version == "v7":
        f22, f23, f32, f33, f42, f43, fb1, fb2, fb3 = sp.symbols('f22 f23 f32 f33 f42 f43 fb1 fb2 fb3')
    elif version == "v8":
        f22, f23, f31, f32, f33, fb1, fb2, fb3 = sp.symbols('f22 f23 f31 f32 f33 fb1 fb2 fb3')
        f41 = 0
        f42 = 0
        f43 = 0

    # equilibrium of x-dir node
    n_x1 = np.array([lx, a, a])
    n_x2 = np.array([lx, -a, a])
    n_x4 = np.array([lx, a, -a])
    n_y1 = np.array([a, ly, a])
    n_z1 = np.array([a, a, lz])
    n_y4 = np.array([a, ly, -a])
    n_z4 = np.array([a, -a, lz])
    n_sx = np.array([a+z, 0, 0])
    n_xb = np.array([-lx, a, a])

    len1 = dis(n_x1, n_x2)
    len21 = dis(n_y1, n_x1)
    len23 = dis(n_z1, n_x1)
    len41 = dis(n_y4, n_x1)
    len43 = dis(n_z4, n_x1)
    len31 = dis(n_sx, n_x1)
    lenb1 = dis(n_xb, n_x1)

    eq_x = f1 / len1 * (n_x2 + n_x4 - 2 * n_x1)  + \
           f21 / len21 * (n_y1 - n_x1) + f23 / len23 * (n_z1 - n_x1) + \
           f41 / len41 * (n_y4 - n_x1) + f43 / len43 * (n_z4 - n_x1) + \
           f31 / len31 * (n_sx - n_x1) + fb1 / lenb1 * (n_x1 - n_xb)
    
    # equilibrium of y-dir node
    n_y1 = np.array([a, ly, a])
    n_y2 = np.array([-a, ly, a])
    n_y4 = np.array([a, ly, -a])
    n_x1 = np.array([lx, a, a])
    n_z1 = np.array([a, a, lz])
    n_x4 = np.array([lx, a, -a])
    n_z2 = np.array([-a, a, lz])
    n_sy = np.array([0, a+z, 0])
    n_yb = np.array([a, -ly, a])

    len22 = dis(n_z1, n_y1)
    len42 = dis(n_z2, n_y1)
    len32 = dis(n_sy, n_y1)
    lenb2 = dis(n_yb, n_y1)

    eq_y = f1 / len1 * (n_y2 + n_y4 - 2 * n_y1) + \
           f21 / len21 * (n_x1 - n_y1) + f22 / len22 * (n_z1 - n_y1) + \
           f41 / len41 * (n_x4 - n_y1) + f42 / len42 * (n_z2 - n_y1) + \
           f32 / len32 * (n_sy - n_y1) + fb2 / lenb2 * (n_y1 - n_yb)
    # equilibrium of z-dir node
    n_z1 = np.array([a, a, lz])
    n_z2 = np.array([-a, a, lz])
    n_z4 = np.array([a, -a, lz])
    n_x1 = np.array([lx, a, a])
    n_y1 = np.array([a, ly, a])
    n_y2 = np.array([-a, ly, a])
    n_x2 = np.array([lx, -a, a])
    n_sz = np.array([0, 0, a+z])
    n_zb = np.array([a, a, -lz])
    len33 = dis(n_sz, n_z1)
    lenb3 = dis(n_zb, n_z1)

    eq_z = f1 / len1 * (n_z2 + n_z4 - 2 * n_z1) + \
           f23 / len23 * (n_x1 - n_z1) + f22 / len22 * (n_y1 - n_z1) + \
           f42 / len42 * (n_y2 - n_z1) + f43 / len43 * (n_x2 - n_z1) + \
           f33 / len33 * (n_sz - n_z1) + fb3 / lenb3 * (n_z1 - n_zb)
    
    eq_x1 = sp.Eq(eq_x[0], 0)
    eq_x2 = sp.Eq(eq_x[1], 0)
    eq_x3 = sp.Eq(eq_x[2], 0)
    eq_y1 = sp.Eq(eq_y[0], 0)
    eq_y2 = sp.Eq(eq_y[1], 0)
    eq_y3 = sp.Eq(eq_y[2], 0)

    if version == "v7":
        eq_z1 = sp.Eq(eq_z[0], 0)
        eq_z2 = sp.Eq(eq_z[1], 0)
        eq_z3 = sp.Eq(eq_z[2], 0)
    elif version == "v8":
        eq_z1 = sp.Eq(f22-f22, 0)
        eq_z2 = sp.Eq(f31-f33, 0)
        eq_z3 = sp.Eq(fb1-fb3, 0)
    eqs = [eq_x1, eq_x2, eq_x3, eq_y1, eq_y2, eq_y3, eq_z1, eq_z2, eq_z3]

    # solve
    if version == "v7":
        sol = sp.solve(eqs, (f22, f23, f32, f33, f42, f43, fb1, fb2, fb3))
        # set force value
        force_val = [
            f1, f21, float(sol[f22]), float(sol[f23]),
            f31, float(sol[f32]), float(sol[f33]),
            f41, float(sol[f42]), float(sol[f43]),
            float(sol[fb1]), float(sol[fb2]), float(sol[fb3])]
    elif version == "v8":
        sol = sp.solve(eqs, (f22, f23, f31, f32, f33, fb1, fb2, fb3))
        # set force value
        force_val = [
            f1, f21,
            float(sol[f22]), float(sol[f23]),
            float(sol[f31]), float(sol[f32]), float(sol[f33]),
            f41, f42, f43,
            float(sol[fb1]), float(sol[fb2]), float(sol[fb3])]
        
        len_val = [len1, len21, len22, len23, len31, len32, len33]

    # set_force_value(model_adams, force_val)
    return force_val, len_val
    # set current length
    # lens = [len1, len21, len22, len23, len31, len32, len33, len41, len42, len43, 2*lx, 2*ly, 2*lz]

def get_linspace(x, interval, num):
    lower_bound = x - interval
    if lower_bound < 0:
        lower_bound = 0
    upper_bound = x + interval
    values = np.linspace(lower_bound, upper_bound, num)
    return values

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
    
    def get_TT_strut2sphere_string_stiffness_type(self, sheet_name='cables_strut_to_sphere'):
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        stiffness_type_list = df['type'].tolist()
        return stiffness_type_list

class TT12WithMiddle():
    def __init__(self, 
                 xlsx_path, 
                 param_length_1=0.7, 
                 param_length_2=0.35, 
                 param_length_3=0, 
                 param_length_4=0.7, 
                 param_length_5=1,
                 param_length_6=0.6,
                 radius=0.2, 
                 z=50,
                 f1=125,
                 f21=250,
                 k1=1000,
                 k2=2000,
                 k3=2000,
                 k4=2000,
                 k5=1000000,
                 k6=1000000,
                 k7=1000000,
                 displacement=(0,0,0.38)):        

        self.radius = radius
        self.xlsxConverter = XLSXConverter(xlsx_path)
        self.N = self.xlsxConverter.get_Node_array(param_length_1=param_length_1, 
                                         param_length_2=param_length_2, 
                                         param_length_3=param_length_3,
                                         param_length_4=param_length_4,
                                         param_length_5=param_length_5,
                                         param_length_6=param_length_6,)  # param_length_4=self.radius*2
        self.z = z
        self.lx = param_length_5/2*1000
        self.lz = param_length_4/2*1000
        self.xlsx_path = xlsx_path
        self.f1 = f1
        self.f21 = f21
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.k7 = k7

        Cb_in = self.xlsxConverter.get_Cb_in_array()
        Cs_in = self.xlsxConverter.get_Cs_in_array(sheet_name='cables_strut_to_strut')
        self.C_b = self.tenseg_ind2C(Cb_in)
        self.C_s = self.tenseg_ind2C(Cs_in)
        self.alt_node_label_list = self.xlsxConverter.get_alt_node_label_list()
        self.TT_mid_connected_array = self.xlsxConverter.get_TT_mid_element_connected_array(sheet_name='cables_strut_to_sphere')
        self.sphere_central_node_label_list = self.xlsxConverter.get_sphere_central_node_label_list()
        self.move_nodes(displacement[0], displacement[1], displacement[2])

        self._time_step = 1e-6
        self._gravity = '0 0 -10'
        self.initial_time_step = 1e-5
        self._floor_solimp = '0.269 0.071 0.065 0.5 2'
        self._floor_solref = '-5773.1 -68.76'
        self._floor_friction = "0.5 0.005 0.0001"
        self._floor_pos = '0 0 0'
        self.tongue = -100
        self.initial_velocity = 70

        self.bar_edge_shape = None

        self.tt_model_df_list = []

        self._node_label_with_class = []

        self.mjcf_model = mjcf.RootElement(model='TT12_environment')
        
        self._set_default_values()
        self._set_visual_attributes()
        self._create_textures_and_materials()
        self._add_floor_and_lights()
    
    def export_to_xml_file(self, export_xml_file_path:str, file_name:str, is_correct_keyframe=True):
        mjcf.export_with_assets(self.mjcf_model, 
                                export_xml_file_path, 
                                file_name)
        xml_path = export_xml_file_path + file_name
        self.delete_external_body(xml_path)
        self.correct_sensor_refname(xml_path)
        self.correct_keyframe(xml_path, is_correct_keyframe)
    
    def delete_external_body(self, xml_path:str):
        """因为子模型含有自由关节，加入到世界后，出现错误：free joint can only be used on top level Element，
        该错误无法通过mjcf库内部的函数解决，因此生成xml文件后，对其进行后处理，将每个子模型内的element移到上一层，
        并将包裹子模型的body类型的element删掉。为了检测这类body，统一其命名格式:'tt_model/'或者'tt_model_xx'，
        其中xx是任意数字

        Args:
            xml_path (str): 模型的xml文件路径
        """        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        pattern = re.compile(r'tt_model_\d+/')
        for body in worldbody.findall('body'):
            if pattern.match(body.attrib['name']) or body.attrib['name'] == 'tt_model/':
                for child in list(body):
                    worldbody.append(child)
                worldbody.remove(body)

        self.prettify(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    def correct_sensor_refname(self, xml_path:str):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sensor = root.find('sensor')
        for framepos in sensor.findall('framepos'):
            refname = framepos.attrib.get('refname')
            # print(refname)
            if refname.endswith('world_site'):  
                framepos.attrib['refname'] = 'world_site'
        
        for frameangvel in sensor.findall('frameangvel'):
            refname = frameangvel.attrib.get('refname')
            # print(refname)
            if refname.endswith('world_site'):  
                frameangvel.attrib['refname'] = 'world_site'
        self.prettify(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    def correct_keyframe(self, xml_path:str, is_correct_keyframe=True):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if is_correct_keyframe:
            model = mj.MjModel.from_xml_path(xml_path)
            data = mj.MjData(model)
            mj.mj_resetData(model, data)
            while data.time < 0.1:
                mj.mj_step(model, data)
            # print(data.qpos)
            keyframe = root.find('keyframe')
            for key in keyframe.findall('key'):
                name = key.attrib.get('name')
                if name.endswith('initial_state'):
                    key.attrib['qpos'] = ' '.join(map(str, data.qpos))
        option = root.find('option')
        option.attrib['timestep'] = str(self.time_step)
        self.prettify(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)

    def prettify(self, elem, level=0):
        indent = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for elem in elem:
                self.prettify(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

    def plot_tensegrity_structure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.N[0, :], self.N[1, :], self.N[2, :], c='r', marker='o')
        for i, pos in enumerate(self.N.T):
            label = f'Node {i}'
            ax.text(pos[0], pos[1], pos[2], label)

        B = np.dot(self.N, self.C_b.T)  # bar
        for j in range(B.shape[1]):
            x_str= str(np.argmax(self.C_b[j, :] == -1))
            y_str = str(np.argmax(self.C_b[j, :] == 1))
            from_pos = self.N[:, int(x_str)]
            to_pos = self.N[:, int(y_str)]
            ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], [from_pos[2], to_pos[2]], c='b')
        
        S = np.dot(self.N, self.C_s.T)  # string
        for j in range(S.shape[1]):
            x_str= str(np.argmax(self.C_s[j, :] == -1))
            y_str = str(np.argmax(self.C_s[j, :] == 1))
            from_pos = self.N[:, int(x_str)]
            to_pos = self.N[:, int(y_str)]
            ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], [from_pos[2], to_pos[2]], c='g')
        
        for i in range(self.TT_mid_connected_array.shape[0]):
            x_str= str(self.TT_mid_connected_array[i, 0]-1)
            y_str = str(self.TT_mid_connected_array[i, 1]-1)
            from_pos = self.N[:, int(x_str)]
            to_pos = self.N[:, int(y_str)]
            ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], [from_pos[2], to_pos[2]], c='y')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
    
    def find_node_coordinate(self, label):
        match = re.search(r'node_(\d+)', label) 
        return self.N[:, int(match.group(1))]
        
    def generate_tt_model(self, tt_model_name='tt_model'):
        """在世界中生成子模型

        Args:
            tt_model_name (str, optional): 子模型的名字，需要tt_model_xx的格式，xx是数字
            . Defaults to 'tt_model'.
            重要！每个新模型需要命名不同旧模型的名字，否则新的模型会基于原有模型进行运算，导致错误
            pos (str, optional): 子模型在世界坐标的位置. Defaults to '0 0 0'.

        Raises:
            ValueError: 需要符合命名格式
        """        
        valid_pattern = re.compile(r'tt_model_\d+')
        if tt_model_name != 'tt_model' and not valid_pattern.match(tt_model_name):
            raise ValueError("Invalid 'tt_model_name' format. It must be 'tt_model' or 'tt_model_<number>/'")

        tt_model =  mjcf.RootElement(model=tt_model_name)
        tt_model.default.tendon.set_attributes(
            limited=True,
            range="0 10",
            width="0.005",
            rgba="1 0 0 1",
            damping="0.1"
        )
 
        self.generate_all_bars(tt_model)
        self.generate_middle_platform(tt_model)
        self.generate_all_tendons(tt_model)
        self.add_exclude_contacts(tt_model)
        self.add_actuator(tt_model)
        self.add_sensor(tt_model)
        self.add_keyframe(tt_model)
    
        self.mjcf_model.attach(tt_model)

    def generate_all_bars(self, tt_model:mjcf.element.RootElement):
        B = np.dot(self.N, self.C_b.T)  # bar
        for j in range(B.shape[1]):
            x_str= str(np.argmax(self.C_b[j, :] == -1))
            y_str = str(np.argmax(self.C_b[j, :] == 1))
            from_pos = self.N[:, int(x_str)]
            to_pos = self.N[:, int(y_str)]

            midpoint_coords = [(from_pos[i] + to_pos[i]) / 2 for i in range(3)]
            
            from_coord_local = [x - y for x, y in zip(from_pos, midpoint_coords)]
            to_coord_local = [x - y for x, y in zip(to_pos, midpoint_coords)]
            fromto_coords_local_str = " ".join(map(str, from_coord_local + to_coord_local))
            
            bar_middle_site = tt_model.worldbody.add(
                'site', name=f'middle_site_{x_str}_{y_str}', pos=midpoint_coords, rgba='0 0 0 0')
            
            new_model = mjcf.RootElement(model=f'rod{x_str}_{y_str}')
            new_model.worldbody.add('geom', name=f'geom{x_str}_{y_str}', type='cylinder', fromto=from_coord_local + to_coord_local, mass=0.0225, size='0.012')
            new_model.worldbody.add('site', name=f'node_{x_str}', pos=from_coord_local)
            new_model.worldbody.add('site', name=f'node_{y_str}', pos=to_coord_local)
            new_model.worldbody.add('site', name=self.alt_node_label_list[j], pos='0 0 0')

            self._node_label_with_class.append(f'rod{x_str}_{y_str}/node_{x_str}')
            self._node_label_with_class.append(f'rod{x_str}_{y_str}/node_{y_str}')
            self._node_label_with_class.append(f'rod{x_str}_{y_str}/{self.alt_node_label_list[j]}')

            new_model.worldbody.add('geom', 
                                    name=f'geom_edge_{x_str}', type='sphere', 
                                    pos=from_coord_local, mass=str(0.0113), condim='6',
                                    size='0.03', rgba="0 0.9 0 0.5")
                
            new_model.worldbody.add('geom', 
                                name=f'geom_edge_{y_str}', type='sphere', 
                                pos=to_coord_local, mass=str(0.0113), condim='6',
                                size='0.03', rgba="0 0.9 0 0.5")
            
            attachment_frame = bar_middle_site.attach(new_model)
            attachment_frame.add('freejoint')
        
    def generate_middle_platform(self, tt_model:mjcf.element.RootElement):
        mid_element_node_label_list = []
        for i in range(self.TT_mid_connected_array.shape[0]):
            mid_element_node_label_list.append(f'node_{self.TT_mid_connected_array[i,1]-1}')

        df = pd.read_excel(self.xlsx_path, sheet_name='spheres')
        centermarker_list = df['centermarker'].to_list()
        connected_node_label_list = list(set(mid_element_node_label_list))
        
        middle_node_coord = self.N[:, int(centermarker_list[0])-1]

        middle_element_site = tt_model.worldbody.add(
                'site', name=f'middle_element_site', pos=middle_node_coord, rgba='0 0 0 0')

        new_model = mjcf.RootElement(model='middle_platform')
        for connected_node_label in connected_node_label_list:
            connected_node_coord = self.find_node_coordinate(connected_node_label)
            connected_coord_local = [x - y for x, y in zip(connected_node_coord, middle_node_coord)]
            connected_node_coord_local_str = " ".join(map(str, connected_coord_local))
            new_model.worldbody.add('site', name=connected_node_label, pos=connected_node_coord_local_str)
            self._node_label_with_class.append(f'middle_platform/{connected_node_label}')

        new_model.worldbody.add('site', name='outside_site', pos='0 0 0', size='0.2', rgba='1 0.9 0 0.5')
        
        new_body = new_model.worldbody.add('body', name=f'inside_ball', pos='0 0 0')
        new_body.add('inertial', pos='0 0 0', mass='6.6', diaginertia='0.12 0.12 0.12')
        # new_body.add('joint', type='hinge', name='x_control', axis="1 0 0", armature="1.0", damping='0.1')
        new_body.add('joint', type='hinge', name='x_control', axis="1 0 0")
        new_body.add('joint', type='hinge', name='y_control', axis="0 1 0")
        new_body.add('joint', type='hinge', name='z_control', axis="0 0 1")
        new_body.add('site', name='inside_site', pos="0 0 0", type='box', size='0.1 0.1 0.1', rgba='0 0.9 0 1')
        
        attachment_frame = middle_element_site.attach(new_model)
        attachment_frame.add('freejoint')
        attachment_frame.add('inertial', pos='0 0 0', mass='0.7', diaginertia='0.02 0.02 0.02')

    def generate_all_tendons(self, tt_model:mjcf.element.RootElement):
        a=350/2
        f1 = self.f1
        f21 = self.f21
        f31 = 160
        f41 = 250
        lx=self.lx
        ly=350
        lz=self.lz
        z = self.z
        
        result, len_val = process_equilibrium(lx=lx,
                        ly=ly,
                        lz=lz,
                        a=a,
                        z=z,
                        f1=f1,
                        f21=f21,
                        f31=f31,
                        f41=f41)
        
        f1, f21, f22, f23, f31, f32, f33, f41, f42, f43, fb1, fb2, fb3 = result

        # print(f'f1: {f1}, f21: {f21}, f22: {f22}, f23: {f23}, f31: {f31}, f32: {f32}, f33: {f33}\nf41: {f41}, f42: {f42}, f43: {f43}, fb1: {fb1}, fb2: {fb2}, fb3: {fb3}')

        # stiffness1 = 1000
        # stiffness21 = 2000
        # stiffness22 = 2000
        # stiffness23 = 2000
        # stiffness31 = 1000000
        # stiffness32 = 1000000
        # stiffness33 = 1000000

        stiffness1 = self.k1
        stiffness21 = self.k2
        stiffness22 = self.k3
        stiffness23 = self.k4
        stiffness31 = self.k5
        stiffness32 = self.k6
        stiffness33 = self.k7
        
        dx1 = f1/stiffness1
        dx21 = f21/stiffness21
        dx22 = f22/stiffness22
        dx23 = f23/stiffness23
        dx31 = f31/stiffness31
        dx32 = f32/stiffness32
        dx33 = f33/stiffness33

        length1 = len_val[0]*0.001
        length21 = len_val[1]*0.001
        length22 = len_val[2]*0.001
        length23 = len_val[3]*0.001
        length31 = len_val[4]*0.001
        length32 = len_val[5]*0.001
        length33 = len_val[6]*0.001

        ori_length1 = abs(dx1-length1)
        ori_length21 = abs(dx21-length21)
        ori_length22 = abs(dx22-length22)
        ori_length23 = abs(dx23-length23)
        ori_length31 = abs(dx31-length31)
        ori_length32 = abs(dx32-length32)
        ori_length33 = abs(dx33-length33)

        # print(f'ori_length1: {ori_length1}, ori_length21: {ori_length21}, ori_length22: {ori_length22}, ori_length23: {ori_length23}, ori_length31: {ori_length31}, ori_length32: {ori_length32}, ori_length33: {ori_length33}')
        
        S = np.dot(self.N, self.C_s.T)  # string
        strut2strut_string_stiffness_type = self.xlsxConverter.get_TT_strut2strut_string_stiffness_type(sheet_name='cables_strut_to_strut')
        strut2sphere_string_stiffness_type = self.xlsxConverter.get_TT_strut2sphere_string_stiffness_type(sheet_name='cables_strut_to_sphere')
        for j in range(S.shape[1]):
            if strut2strut_string_stiffness_type[j] == 1:
                stiffness = stiffness1
                color = '1 0.5 0.5 0.7' # 粉
                ori_length = ori_length1
                ori_length = ori_length1
            elif strut2strut_string_stiffness_type[j] == 2:
                stiffness = stiffness21
                stiffness = stiffness23
                color = '0.7 0 0 0.7'
                ori_length = ori_length21   # 红
                ori_length = ori_length23   # 红
            elif strut2strut_string_stiffness_type[j] == 3:
                stiffness = stiffness22
                stiffness = stiffness21

                color = '0 0.7 0 0.7'
                ori_length = ori_length22   # 绿
                ori_length = ori_length21   # 绿
            elif strut2strut_string_stiffness_type[j] == 4:
                stiffness = stiffness23
                stiffness = stiffness22
                color = '0 0 0.7 0.7'   # 蓝
                ori_length = ori_length23
                ori_length = ori_length22

            x_str= str(np.argmax(self.C_s[j, :] == -1))
            y_str = str(np.argmax(self.C_s[j, :] == 1))
            string_name = f'td{x_str}_{y_str}'
            spatial = tt_model.tendon.add('spatial', name=string_name, stiffness=stiffness,
                                       springlength=ori_length, rgba=color)
            
            from_node_label_with_class = next((elem for elem in self._node_label_with_class if elem.endswith(f'/node_{x_str}')), None)
            to_node_label_with_class = next((elem for elem in self._node_label_with_class if elem.endswith(f'/node_{y_str}')), None)
            spatial.add('site', site=from_node_label_with_class)
            spatial.add('site', site=to_node_label_with_class)

        for i in range(self.TT_mid_connected_array.shape[0]):
            if strut2sphere_string_stiffness_type[i] == 5:
                color = '0.7 0.7 0 0.7' # 黄
                ori_length = ori_length31
                ori_length = ori_length31
                stiffness = stiffness31
                stiffness = stiffness31
            elif strut2sphere_string_stiffness_type[i] == 6:
                color = '0.7 0 0.7 0.7' # 紫
                ori_length = ori_length32
                ori_length = ori_length33
                stiffness = stiffness32
                stiffness = stiffness33
            elif strut2sphere_string_stiffness_type[i] == 7:
                color = '0 0.7 0.7 0.7' # 青
                ori_length = ori_length33
                ori_length = ori_length32
                stiffness = stiffness33
                stiffness = stiffness32
            x_str= str(self.TT_mid_connected_array[i, 0]-1)
            y_str = str(self.TT_mid_connected_array[i, 1]-1)
            string_name = f'td{x_str}_{y_str}'
            spatial = tt_model.tendon.add('spatial', name=string_name, stiffness=stiffness,
                                       springlength=ori_length, rgba=color)
            from_node_label_with_class = next((elem for elem in self._node_label_with_class if elem.endswith(f'/node_{x_str}')), None)
            to_node_label_with_class = next((elem for elem in self._node_label_with_class if elem.endswith(f'/node_{y_str}')), None)
            spatial.add('site', site=from_node_label_with_class)
            spatial.add('site', site=to_node_label_with_class)
        return

    def add_exclude_contacts(self, tt_model:mjcf.element.RootElement):
        rod_name_list = []
        B = np.dot(self.N, self.C_b.T)
        for j in range(B.shape[1]):
            x_str= str(np.argmax(self.C_b[j, :] == -1))
            y_str = str(np.argmax(self.C_b[j, :] == 1))
            body_name=f'rod{x_str}_{y_str}'
            rod_name_list.append(body_name)

        for i in range(len(rod_name_list)):
            for j in range(i+1, len(rod_name_list)):
                body1 = rod_name_list[i]
                body2 = rod_name_list[j]
                tt_model.contact.add('exclude', name=f'exclude_{body1}_{body2}', body1=f'{body1}/', body2=f'{body2}/')
    
    def add_actuator(self, tt_model:mjcf.element.RootElement):
        tt_model.actuator.add('motor',name="torque_x_ctrl", joint="middle_platform/x_control", gear="1", ctrlrange="-200 200", ctrllimited="true")
        tt_model.actuator.add('velocity',name="velocity_x_ctrl", joint="middle_platform/x_control", kv="0", ctrlrange="-200 200")
    
    def add_sensor(self, tt_model:mjcf.element.RootElement):
        tt_model.sensor.add('framepos',name="pos_outside_ball", objtype="body", objname="middle_platform/", reftype="site", refname="world_site")
        tt_model.sensor.add('framepos',name="pos_inside_ball", objtype="body", objname="middle_platform/inside_ball", reftype="site", refname="world_site")
        
        tt_model.sensor.add('frameangvel',name="angvel_outside_ball", objtype="body", objname="middle_platform/", reftype="site", refname="world_site")
        tt_model.sensor.add('frameangvel',name="angvel_inside_ball", objtype="body", objname="middle_platform/inside_ball", reftype="site", refname="world_site")
        
        tt_model.sensor.add('frameangacc',name="angacc_outside_ball", objtype="body", objname="middle_platform/")
        tt_model.sensor.add('frameangacc',name="angacc_inside_ball", objtype="body", objname="middle_platform/inside_ball")
        
        tt_model.sensor.add('accelerometer',site='middle_platform/inside_site')
        
        # tt_model.sensor.add('framepos',name="pos_outside_ball", joint="middle_platform/x_control", kv="10", ctrlrange="0 100")
    
    def add_keyframe(self, tt_model:mjcf.element.RootElement):
        qpos = "0.179416 0.179418 0.365036 0.976395 -5.81186e-05 9.11291e-05 0.21599 -0.179416 0.179418 0.365036 0.976395 -5.81186e-05 -9.11291e-05 -0.21599 -0.179416 -0.179415 0.365036 0.802284 1.56984e-05 -0.000106937 0.596943 0.179416 -0.179415 0.365036 0.802284 1.56984e-05 0.000106937 -0.596943 0.179421 1.72124e-06 0.544061 1 -2.92112e-11 2.67372e-07 -3.52729e-10 0.179412 1.7212e-06 0.185228 1 -4.24989e-11 2.44424e-07 4.32981e-10 -0.179412 1.7212e-06 0.185228 1 -4.24987e-11 -2.44424e-07 -4.32971e-10 -0.179421 1.72124e-06 0.544061 1 -2.92113e-11 -2.67372e-07 3.52739e-10 0 0.179422 0.544061 1 0 0 0 0 -0.179419 0.544061 1 0 0 0 0 -0.17941 0.185228 1 0 0 0 0 0.179414 0.185228 1 0 0 0 0 1.72122e-06 0.364638 1 -4.643e-11 0 0 822.353 0 0"
        qvel = " ".join(["0"] * 78)
        qvel = qvel + f' {self.initial_velocity} 0 0'
        ctrl = f"{self.tongue} 0"
        tt_model.keyframe.add('key', name='initial_state', time='0', qpos=qpos, qvel=qvel, ctrl=ctrl)

        qpos='0.17933 0.223141 0.274582 0.920472 -0.386277 0.023045 0.0546742 -0.17933 0.223141 0.274582 0.920472 -0.386277 -0.023045 -0.0546742 -0.179412 -0.0309028 0.527741 0.907542 -0.377382 -0.0707502 -0.170131 0.179412 -0.0309028 0.527741 0.907542 -0.377382 0.0707502 0.170131 0.179412 0.223032 0.527741 0.924406 -0.381409 1.87586e-05 -3.10178e-06 0.17933 -0.0310121 0.274582 0.92557 -0.378366 0.0117246 -0.00469259 -0.17933 -0.0310121 0.274582 0.92557 -0.378366 -0.0117246 0.00469259 -0.179412 0.223032 0.527741 0.924406 -0.381409 -1.87641e-05 3.10404e-06 9.57909e-13 0.349963 0.400962 1 4.00996e-14 1.09915e-15 -7.35262e-12 -2.77579e-12 0.0960647 0.654625 1 5.8886e-14 1.05046e-15 -7.35306e-12 -6.50961e-12 -0.157834 0.400962 1 -3.76275e-14 6.27321e-16 -7.35271e-12 -2.77591e-12 0.0960647 0.147512 1 5.11088e-14 4.47506e-17 -7.35383e-12 -2.77586e-12 0.0960647 0.400984 0.92388 -0.382683 2.81394e-12 -6.79313e-12 -349.074 2.37314e-11 -5.88057e-12'
        qvel = " ".join(["0"] * 78)
        qvel = qvel + f' {self.initial_velocity} 0 0'
        ctrl = "-100 0"
        tt_model.keyframe.add('key', name='initial_state1', time='0', qpos=qpos, qvel=qvel, ctrl=ctrl)

    def tenseg_ind2C(self, C_ind):
        """
        Creates a connectivity matrix from input index notation array and node matrix.

        Inputs:
            C_ind: index connectivity array (m x 2 array for m members)
            Nodes: node matrix (3 x n array for n nodes)

        Outputs:
            C_mat: connectivity matrix (m x n matrix satisfying M = N*C')

        Example: If given four nodes (N is a 3x4 matrix), and you want one bar to
        be the vector from node 1 to 3, and another bar from node 2 to 4, input
        C_ind would be: C_ind = np.array([[1, 3], [2, 4]])

        C_b = tenseg_ind2C(np.array([[1, 3], [2, 4]]), N)
        """
        nmembers = C_ind.shape[0]  # Number of members being created
        n = self.N.shape[1]  # Number of nodes in the structure

        # Initialize connectivity matrix
        C_mat = np.zeros((n, nmembers))

        for i in range(nmembers):  # Go through each member
            # Get indices for start and end points
            side1 = C_ind[i, 0]
            side2 = C_ind[i, 1]

            # Put -1 at index for start point, 1 at index for end point
            C_mat[side1 - 1, i] = -1
            C_mat[side2 - 1, i] = 1

        return C_mat.T
    
    def move_nodes(self, x, y, z):
        """使节点整体移动

        Args:
            x (_type_): x方向移动距离
            y (_type_): y方向移动距离
            z (_type_): z方向移动距离
        """
        self.N = self.N + np.array([[x], [y], [z]])
    
    def _set_default_values(self) -> None:
        self.mjcf_model.compiler.angle = 'degree'
        self.mjcf_model.compiler.coordinate = "local"
        self.mjcf_model.compiler.inertiafromgeom = "true"
        self.mjcf_model.compiler.autolimits = "true"

        self.mjcf_model.option.timestep = self.initial_time_step
        # self.mjcf_model.option.integrator = 'RK4'
        self.mjcf_model.option.integrator = 'Euler'
        self.mjcf_model.option.gravity = self._gravity
        self.mjcf_model.option.flag.energy = 'enable'
        self.mjcf_model.option.flag.override = 'disable'

        self.mjcf_model.default.joint.damping = 0
        # self.mjcf_model.default.joint.damping = 2
        self.mjcf_model.default.joint.type = 'hinge'
        self.mjcf_model.default.geom.type = 'capsule'
        self.mjcf_model.default.site.type = 'sphere'
        self.mjcf_model.default.site.size = '0.01'

        self.mjcf_model.default.tendon.set_attributes(
            limited=True,
            range="0 10",
            width="0.005",
            rgba="1 0 0 1",
            damping="0.1"
        )

    def _set_visual_attributes(self) -> None:
        self.mjcf_model.visual.headlight.set_attributes(
            ambient=[.6, .6, .6],
            diffuse=[.3, .3, .3],
            specular=[.1, .1, .1]
        )

    def _create_textures_and_materials(self) -> None:
        self.mjcf_model.asset.add('texture', name='skybox', type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7",
                                  rgb2="0 0 0", width="512", height="3072")
        chequered = self.mjcf_model.asset.add('texture', name='2d', type='2d', builtin='checker', width=300,
                                              height=300, rgb1=[.2, .3, .4], rgb2=[.1, .2, .3])
        grid = self.mjcf_model.asset.add('material', name='grid', texture=chequered, texrepeat=[5, 5], reflectance=.2)
        self.mjcf_model.worldbody.add('geom', name='floor', 
                                      type='plane', size=[0, 0, .125], condim='6',
                                      material=grid,
                                      friction=self._floor_friction,
                                      solimp=self._floor_solimp,
                                      solref=self._floor_solref,
                                      pos=self._floor_pos,
                                      priority='1')

    def _add_floor_and_lights(self) -> None:
        for x in [-2, 2]:
            self.mjcf_model.worldbody.add('light', name=f'light_{x}', pos=[x, -1, 3], dir=[-x, 1, -2])
        self.mjcf_model.worldbody.add('site', name='world_site', pos='0 0 0')

    @property
    def floor_pos(self):
        return self._floor_pos

    @floor_pos.setter
    def floor_pos(self, floor_pos):
        self._floor_pos = floor_pos
        floor = self.mjcf_model.find(namespace='geom',identifier='floor')
        floor.set_attributes(pos = floor_pos)

    @property
    def floor_solimp(self):
        return self._floor_solimp

    @floor_solimp.setter
    def floor_solimp(self, floor_solimp):
        self._floor_solimp = floor_solimp
        floor = self.mjcf_model.find(namespace='geom',identifier='floor')
        floor.set_attributes(solimp = floor_solimp)

    @property
    def floor_solref(self):
        return self._floor_solref

    @floor_solref.setter
    def floor_solref(self, floor_solref):
        self._floor_solref = floor_solref
        floor = self.mjcf_model.find(namespace='geom',identifier='floor')
        floor.set_attributes(solref = floor_solref)
    
    @property
    def floor_friction(self):
        return self._floor_friction

    @floor_friction.setter
    def floor_friction(self, floor_friction):
        self._floor_friction = floor_friction
        floor = self.mjcf_model.find(namespace='geom',identifier='floor')
        floor.set_attributes(friction = floor_friction)

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, time_step):
        self._time_step = time_step
        # self.mjcf_model.option.timestep = self._time_step

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, gravity):
        self._gravity = gravity
        self.mjcf_model.option.gravity = self._gravity

class record_data():
    def __init__(self) -> None:
        self.time_datas = []
        self.kinetic_energy_datas = []
        self.potential_energy_datas = []
        self.total_energy_datas = []
        
        self.ctrl_v_x_datas = []
        self.actual_v_x_datas = []
        self.outside_v_x_datas = []

        self.outside_x_pos_datas = []
        self.outside_y_pos_datas = []
        self.outside_z_pos_datas = []
        self.intside_x_pos_datas = []

        self.angvel_inside_ball_datas = []
        self.angvel_outside_ball_datas = []

        self.angacc_inside_ball_datas = []
        self.angacc_outside_ball_datas = []
        
        self.torque_datas = []
        # plt.figure()
    
    def clear_data(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key + '_datas'):
                getattr(self, key + '_datas').clear()

    def bind_data(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key + '_datas'):
                getattr(self, key + '_datas').append(value)
    
    def plot_data_simultaneous(self):
        # 实时绘制
        row = 4
        column = 1
        plt.clf()
        # plt.grid(True, linewidth=0.5)
        # 位置
        plt.subplot(row, column, 1)
        plt.grid(True)
        plt.plot(self.time_datas, self.outside_x_pos_datas, label='outside_x_pos')
        plt.plot(self.time_datas, self.intside_x_pos_datas, label='inside_x_pos')
        plt.xlabel('Time')
        plt.ylabel('x_pos')
        plt.legend()
        plt.draw()
        # 速度
        plt.subplot(row, column, 2)
        plt.grid(True)
        plt.plot(self.time_datas, self.angvel_outside_ball_datas, label='angvel_outside')
        plt.plot(self.time_datas, self.angvel_inside_ball_datas, label='angvel_inside')
        plt.xlabel('Time')
        plt.ylabel('angvel')
        plt.legend()
        plt.draw()
        # 加速度
        plt.subplot(row, column, 3)
        plt.grid(True)
        plt.plot(self.time_datas, self.angacc_outside_ball_datas, label='angacc_outside')
        plt.plot(self.time_datas, self.angacc_inside_ball_datas, label='angacc_inside')
        plt.xlabel('Time')
        plt.ylabel('angacc')
        plt.legend()
        plt.draw()
        # 控制
        plt.subplot(row, column, 4)
        plt.grid(True)
        plt.plot(self.time_datas, self.ctrl_v_x_datas, label='ctrl_v_x')
        # plt.plot(self.time_datas, self.torque_datas, label='torque_x')
        plt.xlabel('Time')
        plt.ylabel('control_variable')
        plt.legend()
        plt.draw()
        plt.pause(0.001)
    
    def plot_data(self, attr_name_list):
        # 最后绘制图案
        non_empty_lists = [attr_name+'_datas' for attr_name in attr_name_list if hasattr(self, attr_name + '_datas')]  
        num_subplots = len(non_empty_lists)
        
        fig, axs = plt.subplots(num_subplots, 1, figsize=(5, 3*num_subplots))
        for i, attr_name in enumerate(non_empty_lists):
            y_data = getattr(self, attr_name)
            axs[i].grid(True)
            axs[i].plot(self.time_datas, y_data, label=attr_name)
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(attr_name)
            axs[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_all_data(self):
        # 最后绘制图案
        non_empty_lists = [attr_name for attr_name in dir(self) if attr_name.endswith('_datas') and attr_name != 'time_datas' and getattr(self, attr_name)]
        num_subplots = len(non_empty_lists)
        
        fig, axs = plt.subplots(num_subplots, 1, figsize=(5, 3*num_subplots))
        for i, attr_name in enumerate(non_empty_lists):
            y_data = getattr(self, attr_name)
            axs[i].grid(True)
            axs[i].plot(self.time_datas, y_data, label=attr_name)
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(attr_name)
            axs[i].legend()

        plt.tight_layout()
        plt.show()
    
    def save_data(self, filename='output.csv'):  
        # 查找所有非空数据列表，排除 time_datas  
        non_empty_lists = [attr_name for attr_name in dir(self)   
                        if attr_name.endswith('_datas')   
                        and attr_name != 'time_datas'   
                        and getattr(self, attr_name)]  
        
        # 创建一个字典来存储数据  
        data_dict = {'Time': self.time_datas}  
        
        # 将非空数据添加到字典中  
        for attr_name in non_empty_lists:  
            data_dict[attr_name] = getattr(self, attr_name)  
        
        df = pd.DataFrame(data_dict)  
        df.to_csv(filename, index=False)  

class TT12_Control():
    def __init__(self, mjcf_file_name=None, keyframe_id=None, xml_string=None) -> None:
        self.mjcf_file_name = mjcf_file_name

        self.Hz = 20
        if xml_string is None:
            self.model = mj.MjModel.from_xml_path(self.mjcf_file_name)  # MuJoCo model
        else:
            self.model = mj.MjModel.from_xml_string(xml_string)  # MuJoCo model
        self.data = mj.MjData(self.model)
        self.keyframe_id = keyframe_id
        self.recorded_data = record_data()
        self.is_control = True
        self.init()

    def init(self):
        # self.model = mj.MjModel.from_xml_path(self.mjcf_file_name)
        # self.data = mj.MjData(self.model)             
        if self.keyframe_id is None:
          mj.mj_resetData(self.model, self.data)
        else:
            mj.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        self.set_velocity_servo("tt_model_0/velocity_x_ctrl", 0)
        mj.mj_forward(self.model, self.data)
    
    def set_velocity_servo(self, actuator_name, kv):
        actuator_no = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        self.model.actuator_gainprm[actuator_no, 0] = kv
        self.model.actuator_biasprm[actuator_no, 2] = -kv
        
    def set_actuator_value(self, actuator_name:str, value:float):
        velocity_x_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        self.data.ctrl[velocity_x_actuator_id] = value
        print(self.data.ctrl[velocity_x_actuator_id])
        print('actuator_no', velocity_x_actuator_id)
        print('value', value)

    def controller(self):
        velocity_x_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "tt_model_0/velocity_x_ctrl")
        torque_x_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "tt_model_0/torque_x_ctrl")
        # self.data.xfrc_applied[13, :] = [0,0,0,-100,0,0]
        if abs(self.data.sensor('tt_model_0/angvel_inside_ball').data[0] - self.data.sensor('tt_model_0/angvel_outside_ball').data[0]) < 0.1:
            # print('111')
            self.data.ctrl[torque_x_actuator_id] = 0
            self.set_velocity_servo("tt_model_0/velocity_x_ctrl", 100)
            self.data.ctrl[velocity_x_actuator_id] = 0
            return False
            # self.data.xfrc_applied[13, :] = [0,0,0,0,0,0]
            # print('time: ', self.data.time)
        # else:
        #     self.set_velocity_servo("tt_model_0/velocity_x_ctrl", 0)
        # if self.data.time > 0.6:
            # self.data.xfrc_applied[13, :] = [0,0,0,0,0,0]
        return True
        
    def reload(self):
        # self.model = mj.MjModel.from_xml_path(self.xml_path)
        # self.data = mj.MjData(self.model)             
        self.init()
        self.recorded_data.clear_data()

    def simulate_step(self):
        current_simstart = self.data.time   # 当前一帧的开始时间
        while (self.data.time - current_simstart < 1.0/self.Hz):
            if self.is_control:
                self.is_control = self.controller()
            mj.mj_step(self.model, self.data)

    def simulate(self, is_render=True, stop_time=10):
        is_paused = False
        if is_render:
            with mj_viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
                viewer.cam.lookat = np.array([-0.00, -0.86, 0.23])
                viewer.cam.azimuth = 177
                viewer.cam.distance = 3
                viewer.cam.elevation = -7
                while viewer.is_running():
                    if keyboard.is_pressed('esc'):  
                        print("Exiting simulation...")  
                        break
                    if self.data.time > stop_time:
                        # Stop the simulation after 20 simulated seconds  
                        print("Exiting simulation...")  
                        break
                    if keyboard.is_pressed('p'):
                        # Pause the simulation  
                        if not is_paused:  
                            is_paused = True  
                            print("Simulation paused. Press 'P' to resume.")  
                        else:  
                            is_paused = False  
                            print("Simulation resumed.")  
                        while keyboard.is_pressed('p'):  
                            pass
                    if keyboard.is_pressed('r'):
                        # Reset the simulation  
                        self.init()
                    # if keyboard.is_pressed('a'):
                    #     self.test()
                    #     while keyboard.is_pressed('a'):  
                    #         pass
                    # Only step the simulation if not paused  
                    if not is_paused:  
                        self.recorded_data.bind_data(time=self.data.time,
                                                # 位置                     
                                                outside_y_pos=self.data.sensor('tt_model_0/pos_outside_ball').data[1],
                                                outside_z_pos=self.data.sensor('tt_model_0/pos_outside_ball').data[2],
                                                intside_x_pos=self.data.sensor('tt_model_0/pos_inside_ball').data[0],
                                                # 速度
                                                angvel_inside_ball = self.data.sensor('tt_model_0/angvel_inside_ball').data[0],
                                                angvel_outside_ball = self.data.sensor('tt_model_0/angvel_outside_ball').data[0],
                                                # 加速度
                                                angacc_inside_ball = self.data.sensor('tt_model_0/angacc_inside_ball').data[0],
                                                angacc_outside_ball = self.data.sensor('tt_model_0/angacc_outside_ball').data[0],
                                                # 控制变量
                                                ctrl_v_x=self.data.ctrl[1],
                                                torque=self.data.ctrl[0])
                        self.simulate_step()
                        # print(self.data.ten_length)
                    viewer.sync()
        else:
            while self.data.time < stop_time:
                self.recorded_data.bind_data(time=self.data.time,
                                                # 位置                     
                                                # outside_x_pos=self.data.sensor('tt_model_0/pos_outside_ball').data[0],
                                                outside_y_pos=self.data.sensor('tt_model_0/pos_outside_ball').data[1],
                                                outside_z_pos=self.data.sensor('tt_model_0/pos_outside_ball').data[2],
                                                # intside_x_pos=self.data.sensor('tt_model_0/pos_inside_ball').data[0],
                                                # 速度
                                                angvel_inside_ball = self.data.sensor('tt_model_0/angvel_inside_ball').data[0],
                                                angvel_outside_ball = self.data.sensor('tt_model_0/angvel_outside_ball').data[0],
                                                # 加速度
                                                angacc_inside_ball = self.data.sensor('tt_model_0/angacc_inside_ball').data[0],
                                                angacc_outside_ball = self.data.sensor('tt_model_0/angacc_outside_ball').data[0],
                                                # 控制变量
                                                ctrl_v_x=self.data.ctrl[1],
                                                torque=self.data.ctrl[0])
                self.simulate_step()

    def test(self):
        mj.mj_setKeyframe(self.model, self.data, 0)
        print('qpos')
        print(self.data.qpos)
        print('--------------')
        print('qvel')
        print(self.data.qvel)
        print('--------------')
        print('ctrl')
        print(self.data.ctrl)
        mj.mj_printModel(self.model, './data/xml/print_model.xml')
        # velocity_x_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "tt_model_0/velocity_x_ctrl")
        # torque_x_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "tt_model_0/torque_x_ctrl")

        # print('----------1111---------------')
        # print(self.model.actuator_gainprm)
        # print(self.model.actuator_biasprm)
        # print('---------------------------------')
        # print(self.data.ctrl)
        # print('---------------------------------')
        # print(torque_x_actuator_id)
        # print(velocity_x_actuator_id)
        # print('---------------------------------')

    def viewer_scene(self):
        def load_callback(m=None, d=None):
            m = mj.MjModel.from_xml_path(self.mjcf_file_name)
            d = mj.MjData(m)
            return m, d
        mj_viewer.launch(loader=load_callback)

class ParameterRecognize:
    def __init__(self) -> None:
        self.i_list = [0, 20, 40, 60, 80, 99]
        self.z_list = np.linspace(0, 165, 100)

        self.get_adams_data()
    
    def get_adams_data(self, analysis_directory=f'./tests/0929/', save_processed_data=False):
        df_dict = np.load(f'{analysis_directory}/ana_all_data.npy', allow_pickle=True).item()
        self.data_dict = {}
        for i in self.i_list:
            filtered_df = df_dict[f'{i+1}'][(df_dict[f'{i+1}']['Time'] <= 1)]
            filtered_df = filtered_df.interpolate(method='linear', limit_direction='both').iloc[::10]  
            filtered_df.reset_index(drop=True, inplace=True)
            filtered_df = filtered_df[['Time', 'height_center', 'disp_x_exter_sphere', 'ang_vel_exter_sphere']]
            filtered_df['height_center'] = filtered_df['height_center'] * 0.001
            filtered_df['disp_x_exter_sphere'] = filtered_df['disp_x_exter_sphere'] * 0.001 - 1
            self.data_dict[f'{i+1}'] = filtered_df
        # 保存self.data_dict
        if save_processed_data:
            np.save(f'{analysis_directory}/processed_data_dict.npy', self.data_dict)
    
    def update_parameter_in_xml(self, xml_file, stiffness, damping, d0, d_width, width, torsional_friction, rolling_friction):
        tree = ET.parse(xml_file)
        worldbody = tree.find('worldbody')
        geoms = worldbody.findall('geom')
        for geom in geoms:
            if geom.get('name') == 'floor':
                geom.set('solref', f'{-stiffness} {-damping}')
                geom.set('solimp', f'{d0} {d_width} {width} 0.5 2')
                geom.set('friction', f'0.5 {torsional_friction} {rolling_friction}')
        return ET.tostring(tree.getroot(), encoding='unicode')
    
    def simulate_with_specific_index(self, index,
                                         stiffness=4360.39, 
                                         damping=138, 
                                         d0=0.96988, 
                                         d_width=0.10222, 
                                         width=0.055103, 
                                         torsional_friction=0.001924,
                                         rolling_friction=0.539, 
                                         stop_time=1,
                                         is_render=False):
        '''
            z的index, 取值范围是0-99    
        '''
        z = self.z_list[index]
        
        working_dir = './tests/0929'
        xlsx_path = f'{working_dir}/topology_TSR_flexible_strut_ball_foot_v8.xlsx'
        export_xml_file_path = f'{working_dir}/tmp_xml/'
        export_xml_file_name = f"{index}.xml"
        xml_path = export_xml_file_path + export_xml_file_name

        a=350/2*0.001
        param_length_1=0.7
        param_length_2=0.35
        param_length_3=0
        param_length_4=0.9
        param_length_5=0.9
        param_length_6 = (2*z*0.001+a*2)

        xml_string = self.update_parameter_in_xml(xml_path, stiffness, damping, d0, d_width, width, torsional_friction, rolling_friction)

        tt12_control = TT12_Control(keyframe_id=0, xml_string=xml_string)  
        tt12_control.Hz = 1010
        tt12_control.is_control = True  
        tt12_control.simulate(is_render=is_render, stop_time=stop_time)
        # tt12_control.recorded_data.plot_all_data()
        return tt12_control.recorded_data.time_datas, tt12_control.recorded_data.outside_z_pos_datas, tt12_control.recorded_data.outside_y_pos_datas, tt12_control.recorded_data.angvel_outside_ball_datas  

    def plot_comparison(self, stiffness, damping, d0, d_width, width, torsional_friction, rolling_friction, index=0, analysis_directory=f'./tests/0929/', stop_time=0.5):
        df_dict = np.load(f'{analysis_directory}/ana_all_data.npy', allow_pickle=True).item()
        adams_df = df_dict[f'{index+1}'][(df_dict[f'{index+1}']['Time'] <= stop_time)]
        adams_df.loc[:, 'height_center'] = adams_df['height_center'] * 0.001
        adams_df.loc[:, 'disp_x_exter_sphere'] = adams_df['disp_x_exter_sphere'] * 0.001 - 1
        # simulated_time_list, simulated_outside_z_pos_list, simulated_outside_y_pos_list, simulated_angvel_outside_ball_list = self.simulate_with_specific_index(self.i_list[index], stiffness, damping, d0, d_width, width, torsional_friction, rolling_friction, is_render=False)
        simulated_time_list, simulated_outside_z_pos_list, simulated_outside_y_pos_list, simulated_angvel_outside_ball_list = self.simulate_with_specific_index(index, stiffness, damping, d0, d_width, width, torsional_friction, rolling_friction, stop_time=stop_time, is_render=False)
        # tongue = self.tongue_list[self.i_list[index]]
        z = self.z_list[index]
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle(f'z={z}')
        axs[0].plot(simulated_time_list, simulated_outside_z_pos_list, label='mujoco')  
        axs[0].plot(adams_df['Time'], adams_df['height_center'], label='adams')  
        axs[0].set_title('height(mm) vs time')  
        axs[0].set_xlabel('time (s)')  
        axs[0].set_ylabel('height (mm)')  

        # 绘制水平位置(mm)的变化  
        axs[1].plot(simulated_time_list, simulated_outside_y_pos_list, label='mujoco')
        axs[1].plot(adams_df['Time'], adams_df['disp_x_exter_sphere'], label='adams')
        axs[1].set_title('forward position(mm) vs time')  
        axs[1].set_xlabel('time (s)')  
        axs[1].set_ylabel('forward position (mm)')  

        # 绘制角速度(rad/s)的变化  
        axs[2].plot(simulated_time_list, simulated_angvel_outside_ball_list, label='mujoco')  
        axs[2].plot(adams_df['Time'], adams_df['ang_vel_exter_sphere'], label='adams')  
        axs[2].set_title('angular velocity(rad/s) vs time')  
        axs[2].set_xlabel('time (s)')  
        axs[2].set_ylabel('angular velocity (rad/s)')  

        for ax in axs:
            ax.legend() 
            ax.grid() 
        # 调整布局  
        plt.tight_layout()  

        # 显示图形  
        plt.show()  

    def plot_recognize_result(self, result_file='./tests/0929/optimizer_log'):
        from scipy.interpolate import UnivariateSpline  
        def plot_data(line_number_list, data_list, title, subplot_position):  
            plt.subplot(2, 4, subplot_position)  
            plt.scatter(line_number_list, data_list, s=1.2)
            spline = UnivariateSpline(line_number_list, data_list, s=1000)  # s为平滑因子  
            # 生成拟合曲线的数据  
            x_fit = np.linspace(min(line_number_list), max(line_number_list), 100)  
            y_fit = spline(x_fit)  
            # 绘制拟合曲线  
            plt.plot(x_fit, y_fit, color='red', label='Spline Fit')  
            plt.title(title)  
            plt.grid()  
        import json  
        line_number_list = []  
        loss_list = []  
        d0_list = []  
        d_width_list = []  
        width_list = []  
        stiffness_list = []  
        damping_list = []  
        rolling_friction_list = []  
        torsional_friction_list = []
        least_loss = 9999
        with open(result_file, 'r') as file:  
            for line_number, line in enumerate(file):  
                if line_number < 1001:
                    continue
                data = json.loads(line.strip())  
                line_number_list.append(line_number) 
                current_loss = data.get("#loss")
                if current_loss < least_loss:
                    least_loss = current_loss
                    index = line_number
                loss_list.append(current_loss)  
                d0_list.append(data.get("d0"))  
                d_width_list.append(data.get("d_width"))  
                width_list.append(data.get("width"))  
                stiffness_list.append(data.get("stiffness"))  
                damping_list.append(data.get("damping"))  
                rolling_friction_list.append(data.get("rolling_friction"))  
                torsional_friction_list.append(data.get("torsional_friction"))  

        print('最小的loss的index:', index, 'loss:', least_loss)
        plt.figure(figsize=(12,24))  
        plot_data(line_number_list, d0_list, 'd0 History', 1)  
        plot_data(line_number_list, d_width_list, 'd_width History', 2)  
        plot_data(line_number_list, width_list, 'width History', 3)  
        plot_data(line_number_list, stiffness_list, 'stiffness History', 4)  
        plot_data(line_number_list, damping_list, 'damping History', 5)  
        plot_data(line_number_list, rolling_friction_list, 'rolling_friction History', 6)
        plot_data(line_number_list, torsional_friction_list, 'torsional_friction History', 7)
        plot_data(line_number_list, loss_list, 'Loss History', 8)  

        plt.tight_layout()  
        plt.show()  

    def get_single_error(self, index=0, stiffness=4360.39, 
                        damping=138, 
                        d0=0.96988, 
                        d_width=0.10222, 
                        width=0.055103, 
                        rolling_friction=0.539, 
                        torsional_friction=0.001924):
        data_dict_inedx = self.i_list[index]+1
        adams_df = self.data_dict[f'{data_dict_inedx}']
        simulated_time_list, simulated_outside_z_pos_list, simulated_outside_y_pos_list, simulated_angvel_outside_ball_list = self.simulate_with_specific_index(index=self.i_list[index], is_render=False, stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, rolling_friction=rolling_friction, torsional_friction=torsional_friction)
        interpolated_values_angvel = np.interp(adams_df['Time'], simulated_time_list, simulated_angvel_outside_ball_list)
        interpolated_values_height = np.interp(adams_df['Time'], simulated_time_list, simulated_outside_z_pos_list)
        interpolated_values_forward_pos = np.interp(adams_df['Time'], simulated_time_list, simulated_outside_y_pos_list)
        ROBOT_HALF_WIDTH = 0.35
        real_pose = np.array([adams_df['height_center'], adams_df['disp_x_exter_sphere']])  # 乘以0.5使水平移动的影响变小
        mujoco_pose = np.array([interpolated_values_height, interpolated_values_forward_pos])
        e_pos = np.sum(np.linalg.norm(real_pose - mujoco_pose, axis=0))/ROBOT_HALF_WIDTH/15  # 18和77是为了让这两个误差算出来差不多
        e_vel = np.linalg.norm(adams_df['ang_vel_exter_sphere'] - interpolated_values_angvel, axis=0)/80
        return e_pos, e_vel

    def loss_function(self, stiffness, damping, d0, d_width, width, torsional_friction, rolling_friction):
        error = 0
        for i in range(len(self.i_list)):
            e_pos, e_vel = self.get_single_error(index=i, stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, rolling_friction=rolling_friction, torsional_friction=torsional_friction)
            error += e_pos + e_vel
        return error
    
    def experiment_with_optimizer(self, logger_path='./tests/0929/optimizer_log'):
        instrum = ng.p.Instrumentation(
        stiffness=ng.p.Scalar(lower=3000, upper=10000),
        damping=ng.p.Scalar(lower=80, upper=500),
        d0=ng.p.Scalar(lower=0.5, upper=0.999),
        d_width=ng.p.Scalar(lower=0.0001, upper=0.5),
        width=ng.p.Scalar(lower=0.00001, upper=0.1),
        torsional_friction=ng.p.Scalar(lower=0.0001, upper=0.5),
        rolling_friction=ng.p.Scalar(lower=0.0001, upper=0.5),
        )

        # child = instrum.spawn_child()
        # child.value = ((), {'d0': 0.9433598140425187, 'd_width': 0.1395079405492468, 'width': 0.05208016638200587, 'stiffness': 4724.630709636312, 'damping': 91.11742334723378, 'rolling_friction': 0.5985903152412875, 'torsional_friction': 0.02774301352299431})
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=1000, num_workers=28)

        logger = ng.callbacks.ParametersLogger(logger_path)
        optimizer.register_callback("tell",  logger)
        
        start_time = time.time()
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(self.loss_function, executor=executor, batch_mode=False, verbosity=2)

        print('loss:',recommendation.loss)
        print('value:',recommendation.value[1])
        print('total time: ', time.time() - start_time)

    def sensitivity_analysis(self):
        result = {'stiffness': 5693.293409913486, 'damping': 467.8923522753243, 'd0': 0.9242124266838425, 'd_width': 0.02431798300083489, 'width': 0.052154498290043386, 'torsional_friction': 0.04024805720625445, 'rolling_friction': 0.007583432626563187}
        stiffness = result['stiffness']
        damping = result['damping']
        d0 = result['d0']
        d_width = result['d_width']
        width = result['width']
        torsional_friction = result['torsional_friction']
        rolling_friction = result['rolling_friction']

        stiffness_list = np.linspace(3000, 10000, 10)
        damping_list = np.linspace(80, 500, 10)
        d0_list = np.linspace(0.5, 0.999, 10)
        d_width_list = np.linspace(0.01, 0.5, 10)
        width_list = np.linspace(0.00001, 0.1, 10)
        torsional_friction_list = np.linspace(0.0001, 0.6, 10)
        rolling_friction_list = np.linspace(0.0001, 0.6, 10)
        error_list = []
        for stiffness in stiffness_list:
            error = self.loss_function(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, torsional_friction=torsional_friction, rolling_friction=rolling_friction)
            error_list.append(error)
        # 算error方差
        print('stiffness: ',np.std(error_list))
        stiffness = result['stiffness']

        error_list = []
        for damping in damping_list:
            error = self.loss_function(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, torsional_friction=torsional_friction, rolling_friction=rolling_friction)
            error_list.append(error)
        # 算error方差
        print('damping: ',np.std(error_list))
        damping = result['damping']

        error_list = []
        for d0 in d0_list:
            error = self.loss_function(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, torsional_friction=torsional_friction, rolling_friction=rolling_friction)
            error_list.append(error)
        # 算error方差
        print('d0: ',np.std(error_list))
        d0 = result['d0']

        error_list = []
        for d_width in d_width_list:
            error = self.loss_function(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, torsional_friction=torsional_friction, rolling_friction=rolling_friction)
            error_list.append(error)
        # 算error方差
        print('d_width: ',np.std(error_list))
        d_width = result['d_width']

        error_list = []
        for width in width_list:
            error = self.loss_function(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, torsional_friction=torsional_friction, rolling_friction=rolling_friction)
            error_list.append(error)
        # 算error方差
        print('width: ',np.std(error_list))
        width = result['width']

        error_list = []
        for torsional_friction in torsional_friction_list:
            error = self.loss_function(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, torsional_friction=torsional_friction, rolling_friction=rolling_friction)
            error_list.append(error)
        # 算error方差
        print('torsional_friction: ',np.std(error_list))
        torsional_friction = result['torsional_friction']

        error_list = []
        for rolling_friction in rolling_friction_list:
            error = self.loss_function(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, torsional_friction=torsional_friction, rolling_friction=rolling_friction)
            error_list.append(error)
        # 算error方差
        print('rolling_friction: ',np.std(error_list))
        rolling_friction = result['rolling_friction']

class ParallelExperiment:
    def __init__(self) -> None:

        # self.k1_list = np.linspace(600000, 700000, 100)
        f1,f21,z,k1,k2,k3,k4,k5,k6,k7=[1.08570230e+02, 2.15525292e+02, 8.81208515e-01, 6.54814411e+05,
                        6.00253605e+05, 4.49265232e+04, 2.31510442e+05, 2.34302564e+05,
                        8.01972959e+05, 3.05995422e+02]
        
        # self.list = get_linspace(6.54814411e+05, 10000, 50) # k1
        # self.list = get_linspace(6.00253605e+05, 10000, 50) # k2
        # self.list = np.linspace(100, 10000, 50) # k7
        # self.list = get_linspace(k1, 10000, 100)
        # self.list = get_linspace(512930, 2, 50)
        self.list = np.linspace(100, 1000000, 100)  #k1
        self.index_list = [i for i in range(len(self.list))]

        self.stiffness = 5773
        self.damping = 68
        self.d0 = 0.27
        self.d_width = 0.07
        self.width = 0.06
        self.rolling_friction = 0.005
        self.torsional_friction = 0.001

    def simulate_with_specific_index(self, index):
        '''
            k1的index, 取值范围是0-99  
        '''
        working_dir = './tests/1008'
        xlsx_path = f'{working_dir}/topology_TSR_flexible_strut_ball_foot_v8.xlsx'
        export_xml_file_path = f'{working_dir}/tmp_xml/'
        export_xml_file_name = f"{index}.xml"
        xml_path = export_xml_file_path + export_xml_file_name

        f1,f21,z,k1,k2,k3,k4,k5,k6,k7=[1.08570230e+02, 2.15525292e+02, 8.81208515e-01, 6.54814411e+05,
                        6.00253605e+05, 4.49265232e+04, 2.31510442e+05, 2.34302564e+05,
                        8.01972959e+05, 3.05995422e+02]
        
        a=350/2*0.001
        # z=50
        param_length_1=0.7
        param_length_2=0.35
        param_length_3=0
        param_length_4=0.9
        param_length_5=0.9
        param_length_6 = (2*z*0.001+a*2)

        ### 要修改的地方1
        save_data_csv = f'{working_dir}/k1/{index}.csv'
        ### 要修改的地方2
        k1=self.list[index]
        ### 要修改的地方3
        #  增加对应文件夹

        # f1=self.f1_list[index]

        tt12_with_middle = TT12WithMiddle(xlsx_path, 
                                        param_length_1, 
                                        param_length_2, 
                                        param_length_3, 
                                        param_length_4, 
                                        param_length_5, 
                                        param_length_6,
                                        f1=f1,
                                        f21=f21,
                                        z=z,
                                        k1=k1,
                                        k2=k2,
                                        k3=k3,
                                        k4=k4,
                                        k5=k5,
                                        k6=k6,
                                        k7=k7,
                                        displacement=(0, 0, 0.38))
        # tt12_with_middle.gravity='0 0 0'
        # tt12_with_middle.plot_tensegrity_structure()
        tt12_with_middle.time_step = 1e-6
        tt12_with_middle.floor_solimp = f'{self.d0} {self.d_width} {self.width} 0.5 2'
        tt12_with_middle.floor_solref = f'{-self.stiffness} {-self.damping}'
        tt12_with_middle.floor_friction = f'0.5 {self.torsional_friction} {self.rolling_friction}'
        tt12_with_middle.generate_tt_model(tt_model_name='tt_model_0')
        tt12_with_middle.export_to_xml_file(export_xml_file_path, 
                                            export_xml_file_name, 
                                            is_correct_keyframe=True)

        tt12_control = TT12_Control(keyframe_id=0, mjcf_file_name=xml_path)  
        tt12_control.Hz = 1010
        tt12_control.is_control = True  
        tt12_control.simulate(is_render = False, stop_time=1)
        # print('f21:', f2)
        tt12_control.recorded_data.save_data(save_data_csv)
        print(f'{index} finished')

    def run_simulations_specify_name(self, function_name:str, index_list_name:str):  
        if not hasattr(self, function_name):  
            raise ValueError(f"{function_name} is not a valid method of {self.__class__.__name__}")  

        func = getattr(self, function_name)
        list = getattr(self, index_list_name)  

        with multiprocessing.Pool(processes=28) as pool:  
            from functools import partial  
            results = pool.map(partial(func), list) 
        
        return results  

class OptimalParametersExperiment:
    def __init__(self) -> None:
        self.f1_list = np.linspace(50, 130, 50)
        self.f21_list = np.linspace(200, 450, 50)
        self.z_list = np.linspace(0, 165, 100)
        self.k1_list = np.linspace(100, 1000000, 100)
        self.k2_list = np.linspace(100, 1000000, 100)
        self.k3_list = np.linspace(100, 1000000, 100)
        self.k4_list = np.linspace(100, 1000000, 100)
        self.k5_list = np.linspace(100, 1000000, 100)
        self.k6_list = np.linspace(100, 1000000, 100)
        self.k7_list = np.linspace(100, 1000000, 100)
        # self.long_bar_len_list = np.linspace(0.8, 1.2, 50)

        self.stiffness = 5773
        self.damping = 68
        self.d0 = 0.27
        self.d_width = 0.07
        self.width = 0.06
        self.rolling_friction = 0.005
        self.torsional_friction = 0.001

    def simulate_with_specific_params(self, f1=125, f21=250, z=50, k1=1000, k2=2000, k3=2000, k4=2000, k5=1000000, k6=1000000, k7=1000000,
                                         stop_time=1,
                                         is_render=False,
                                         is_new_xml=True,
                                         is_save_data=False,
                                         xml_name=None,
                                         time_step=1e-5,
                                         ):
        
        '''
            z的index, 取值范围是0-99    
        '''
        working_dir = './tests/1008'
        xlsx_path = f'{working_dir}/topology_TSR_flexible_strut_ball_foot_v8.xlsx'
        export_xml_file_path = f'{working_dir}/tmp_xml_1005/'
        name = f"f1-{f1}-f21-{f21}-z-{z}-k1-{k1}-k2-{k2}-k3-{k3}-k4-{k4}-k5-{k5}-k6-{k6}-k7-{k7}"
        # name.replace(".", "_")
        name = f"{name}-{random.randint(1, 10000)}-{random.randint(1, 10000)}"
        export_xml_file_name = f"{name}.xml"
        xml_path = export_xml_file_path + export_xml_file_name

        # 检测是否存在同名文件，如果存在，则重新命名    # 还是会存在问题：它检测的时候，还没有同名文件，它检测后就有了，导致两个
        while os.path.exists(xml_path):
            time.sleep(0.1)
            name = f"{name}-{random.randint(1, 1000)}"
            export_xml_file_name = f"{name}.xml"
            xml_path = export_xml_file_path + export_xml_file_name
            print(xml_path)
        
        if xml_name is not None:
            name = xml_name
            export_xml_file_name = f"{name}.xml"
            xml_path = export_xml_file_path + export_xml_file_name

        save_data_csv = f'{working_dir}/tmp_csv/{name}.csv'

        a=350/2*0.001
        param_length_1=0.7
        param_length_2=0.35
        param_length_3=0
        param_length_4=0.9
        param_length_5=0.9
        param_length_6 = (2*z*0.001+a*2)
        if is_new_xml:
            tt12_with_middle = TT12WithMiddle(xlsx_path, 
                                            param_length_1, 
                                            param_length_2, 
                                            param_length_3, 
                                            param_length_4, 
                                            param_length_5, 
                                            param_length_6,
                                            f1=f1,
                                            f21=f21,
                                            k1=k1,
                                            k2=k2,
                                            k3=k3,
                                            k4=k4,
                                            k5=k5,
                                            k6=k6,
                                            k7=k7,
                                            z=z,
                                            displacement=(0, 0, 0.38))
            # tt12_with_middle.gravity='0 0 0'
            # tt12_with_middle.plot_tensegrity_structure()
            tt12_with_middle.time_step = time_step
            tt12_with_middle.floor_solimp = f'{self.d0} {self.d_width} {self.width} 0.5 2'
            tt12_with_middle.floor_solref = f'{-self.stiffness} {-self.damping}'
            tt12_with_middle.floor_friction = f'0.5 {self.torsional_friction} {self.rolling_friction}'
            tt12_with_middle.generate_tt_model(tt_model_name='tt_model_0')
            tt12_with_middle.export_to_xml_file(export_xml_file_path, 
                                                export_xml_file_name, 
                                                is_correct_keyframe=True)

        tt12_control = TT12_Control(keyframe_id=0, mjcf_file_name=xml_path)  
        tt12_control.Hz = 2000
        tt12_control.is_control = True  
        tt12_control.simulate(is_render=is_render, stop_time=stop_time)
        if is_save_data:
            tt12_control.recorded_data.save_data(save_data_csv)
        # tt12_control.recorded_data.plot_all_data()
        if xml_name is None:
            os.remove(xml_path)

        return tt12_control.recorded_data.time_datas, tt12_control.recorded_data.outside_z_pos_datas, tt12_control.recorded_data.outside_y_pos_datas, tt12_control.recorded_data.angvel_outside_ball_datas  

    def get_error(self, f1=125, f21=250, z=50, k1=1000, k2=2000, k3=2000, k4=2000, k5=1000000, k6=1000000, k7=1000000):
        _, height_list, _, _ = self.simulate_with_specific_params(f1=f1, f21=f21, z=z, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, k7=k7, stop_time=1, is_render=False)
        return 1/max(height_list)
    
    def get_error_ga(self, x):
        f1, f21, z, k1, k2, k3, k4, k5, k6, k7 = x
        return self.get_error(f1=f1, f21=f21, z=z, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, k7=k7)
    
    def experiment_with_optimizer(self, logger_path='./tests/0929/optimizer_log'):
        instrum = ng.p.Instrumentation(
            f1=ng.p.Scalar(lower=50, upper=130),
            f21=ng.p.Scalar(lower=200, upper=450),
            z=ng.p.Scalar(lower=0, upper=165),
            k1=ng.p.Scalar(lower=100, upper=1000000),
            k2=ng.p.Scalar(lower=100, upper=1000000),
            k3=ng.p.Scalar(lower=100, upper=1000000),
            k4=ng.p.Scalar(lower=100, upper=1000000),
            k5=ng.p.Scalar(lower=100, upper=1000000),
            k6=ng.p.Scalar(lower=100, upper=1000000),
            k7=ng.p.Scalar(lower=100, upper=1000000),
        )

        # child = instrum.spawn_child()
        # child.value = ((), {'d0': 0.9433598140425187, 'd_width': 0.1395079405492468, 'width': 0.05208016638200587, 'stiffness': 4724.630709636312, 'damping': 91.11742334723378, 'rolling_friction': 0.5985903152412875, 'torsional_friction': 0.02774301352299431})
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=1000, num_workers=28)

        logger = ng.callbacks.ParametersLogger(logger_path)
        optimizer.register_callback("tell",  logger)
        
        start_time = time.time()
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(self.get_error, executor=executor, batch_mode=True, verbosity=1)

        print('loss:',recommendation.loss)
        print('value:',recommendation.value[1])
        print('total time: ', time.time() - start_time)

    def plot_recognize_result(self, result_file='./tests/0929/optimizer_log'):
        from scipy.interpolate import UnivariateSpline  
        def plot_data(line_number_list, data_list, title, subplot_position):  
            plt.subplot(3, 4, subplot_position)  
            plt.scatter(line_number_list, data_list, s=1.2)
            spline = UnivariateSpline(line_number_list, data_list, s=1000)  # s为平滑因子  
            # 生成拟合曲线的数据  
            x_fit = np.linspace(min(line_number_list), max(line_number_list), 100)  
            y_fit = spline(x_fit)  
            # 绘制拟合曲线  
            plt.plot(x_fit, y_fit, color='red', label='Spline Fit')  
            plt.title(title)  
            plt.grid()  
        import json  
        line_number_list = []  
        loss_list = []  
        f1_list = []  
        f21_list = []  
        z_list = []  
        k1_list = []  
        k2_list = []  
        k3_list = []  
        k4_list = []  
        k5_list = []  
        k6_list = []  
        k7_list = []  

        least_loss = 9999
        with open(result_file, 'r') as file:  
            for line_number, line in enumerate(file):  
                # if line_number < 1001:
                #     continue
                data = json.loads(line.strip())  
                line_number_list.append(line_number) 
                current_loss = data.get("#loss")
                if current_loss < least_loss:
                    least_loss = current_loss
                    index = line_number
                loss_list.append(current_loss)  
                f1_list.append(data.get("f1"))  
                f21_list.append(data.get("f21"))  
                z_list.append(data.get("z"))  
                k1_list.append(data.get("k1"))  
                k2_list.append(data.get("k2"))  
                k3_list.append(data.get("k3"))  
                k4_list.append(data.get("k4"))  
                k5_list.append(data.get("k5"))  
                k6_list.append(data.get("k6"))  
                k7_list.append(data.get("k7"))  

        print('最小的loss的index:', index, 'loss:', least_loss)
        print('f1:', f1_list[index], 'f21:', f21_list[index], 'z:', z_list[index], 'k1:', k1_list[index], 'k2:', k2_list[index], 'k3:', k3_list[index], 'k4:', k4_list[index], 'k5:', k5_list[index], 'k6:', k6_list[index], 'k7:', k7_list[index])
        plt.figure(figsize=(12,24))  
        plot_data(line_number_list, f1_list, 'f1', 1)  
        plot_data(line_number_list, f21_list, 'f21', 2)  
        plot_data(line_number_list, z_list, 'z', 3)  
        plot_data(line_number_list, k1_list, 'k1', 4)  
        plot_data(line_number_list, k2_list, 'k2', 5)  
        plot_data(line_number_list, k3_list, 'k3', 6)  
        plot_data(line_number_list, k4_list, 'k4', 7)  
        plot_data(line_number_list, k5_list, 'k5', 8)  
        plot_data(line_number_list, k6_list, 'k6', 9)  
        plot_data(line_number_list, k7_list, 'k7', 10)  
        plot_data(line_number_list, loss_list, 'Loss History', 11)  

        plt.tight_layout()  
        plt.show()  

class GeneticAlgroithm:
    def __init__(self):
        pass

    def is_converge(self, pop):
        return np.all(pop == pop[0])
    
    def translateDNA(self, 
        pop:list[list[float]],
        params_bound:list[tuple[float,float]],
        dna_num:int, 
        dna_size:int)->list[list[float]]:
        """ Translate DNA to parameter's values
        :param pop:             A matrix of population, which contains pop_num rows and (dna_size * dna_num) columns.
                                Each row represents a population's ALL DNAs, every (dna_size) columns of each row represents a DNA of a population.
        :param params_bound:    A list whose element is a tuple contains the lower and upper bound of a parameter.
        :param dna_num:         Number of DNAs that a population has.
        :param dna_size:        Size of a DNA.
        
        :return:
            A list whose element is also a list that contains each parameter's values.
        """
        sub_dnas = np.hsplit(pop, dna_num)
        vals = []

        for sub_dna, param_bound in zip(sub_dnas, params_bound):
            val_decoded = sub_dna.dot(2 ** np.arange(dna_size)[::-1]) / float(2 ** dna_size - 1) * (param_bound[1] - param_bound[0]) + param_bound[0]
            vals.append(list(val_decoded))

        return vals
    
    def decode_parameters(self,
                      vals: list[list[float]],   
                      params_bound: list[tuple[float, float]],   
                      dna_num: int,   
                      dna_size: int) -> list[list[float]]:  
        """ Decode parameter's values back to DNA representation  
        :param vals:           A list of values for each parameter.  
        :param params_bound:   A list whose element is a tuple contains the lower and upper bound of a parameter.  
        :param dna_num:        Number of DNAs that a population has.  
        :param dna_size:       Size of a DNA.  
        
        :return:  
            A matrix of population, which contains pop_num rows and (dna_size * dna_num) columns.  
        """  
        # Initialize an empty array to store the population  
        pop_num = np.array(vals).shape[1]
        pop = np.zeros((pop_num, dna_size * dna_num))  

        for i, (val, param_bound) in enumerate(zip(vals, params_bound)):  
            # Normalize the value back to a 0-1 range  
            normalized_val = (np.array(val) - param_bound[0]) / (param_bound[1] - param_bound[0])  
            
            # Get binary representation  
            binary_representation = (normalized_val * (2 ** dna_size - 1)).astype(int)  

            # Ensure we convert to binary and pad to dna_size  
            dna_encoded = np.array([list(np.binary_repr(b, width=dna_size)) for b in binary_representation]).astype(int)  
            
            # Flatten and place in the correct position in the population  
            pop[:, i * dna_size:(i + 1) * dna_size] = dna_encoded  
        
        return pop.astype(int)  
    
    def get_params_vals_from_json(self, json_file='./config_gen_1.json'):
        # 读取 JSON 文件  
        with open(json_file, 'r') as file:  
            data = json.load(file)  

        # 创建一个空列表来存储每一行的numpy数组  
        data_list = []  

        # 遍历每个字典条目  
        for item in data:  
            # 提取id以外的所有数值  
            values = [value for key, value in item.items() if key != 'id']  
            # 将提取出的值追加到列表中  
            data_list.append(values)  

        tmp = np.array(data_list)
        tmp[:, -7:] *= 1000  
        return  tmp.T.tolist()  

    def get_fitness(self, pop, params_bound, dna_num, dna_size, func):
        params_vals = self.translateDNA(pop, params_bound, dna_num, dna_size)
        pred = func(params_vals)
        return -(pred - np.max(pred)) + 1e-3  # 要加上一个很小的正数

    def mutation(self, child, dna_num, dna_size, mutation_rate=0.005):
        # 基本位变异算子
        if np.random.rand() < mutation_rate:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, dna_size * dna_num)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转（异或运算符 1与1为0、1与0为1、0与0为0）

    def select(self, pop, pop_size, fitness):  # 描述了从np.arange(POP_SIZE)里选择每一个元素的概率，概率越高约有可能被选中，最后返回被选中的个体即可
        idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True, p=(fitness) / (fitness.sum()))
        return pop[idx]
    
    def crossover_and_mutation(self, pop, pop_size, dna_num, dna_size, crossover_rate=0.95):  # 单点交叉
        """变异和交叉

        Args:
            pop (_type_): _description_
            pop_size (_type_): _description_
            dna_num (_type_): _description_
            dna_size (_type_): _description_
            crossover_rate (float, optional): _description_. Defaults to 0.95.

        Returns:
            _type_: _description_
        """
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[np.random.randint(pop_size)]  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=dna_size * dna_num)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            self.mutation(child, dna_num=dna_num, dna_size=dna_size)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop

    def run(self, pop=None):
        num_params = 10
        np.set_printoptions(threshold=np.inf)

        params_bound = [
            (50, 150),   # f1
            (150, 350),  # f21
            (0, 165),    # z
            (100, 1000000),   # k1
            (100, 1000000),   # k2
            (100, 1000000),   # k3
            (100, 1000000),   # k4
            (100, 1000000),   # k5
            (100, 1000000),   # k6
            (100, 1000000)    # k7
        ]

        dna_num = num_params    # DNA数目也即参数数目
        pop_size = 160           # 种群数目, 可认为是每一次迭代需要跑的仿真数目
        dna_size = 16            # 每个DNA长度, 应该与参数的分辨率有关 
        num_generations = 200     # 迭代次数
        crossover_rate = 0.95   # 交叉概率

        if pop is None:
            pop = np.random.randint(2, size=(pop_size, dna_size * dna_num))  # matrix (POP_SIZE, DNA_SIZE)
        gen = 1

        for gen in range(num_generations):  
            # 计算适应度  
            # if gen <= 99:    # 接着昨天中断的继续算
            #     continue
            fitness = self.get_fitness(pop, params_bound=params_bound, dna_num=dna_num, dna_size=dna_size, func=self.func)  
            # 选择操作  
            pop = self.select(pop, pop_size, fitness)  
            # 交叉与变异操作  
            pop = np.array(self.crossover_and_mutation(pop=pop, pop_size=pop_size, dna_num=dna_num, dna_size=dna_size, crossover_rate=crossover_rate))  
            # 翻译DNA  
            params_vals = self.translateDNA(pop, params_bound=params_bound, dna_num=dna_num, dna_size=dna_size)  
            # 保存参数值  
            min_fitness_index = np.argmin(fitness)  
            print("gen:", gen)  
            params_vals_array = np.array(params_vals)  
            print("best:", params_vals_array[:, min_fitness_index])  
            # print("pop:", pop)  
            with open('./tests/1005/params_vals1004.txt', 'a') as f:  # 使用'a'追加数据  
                f.write(f'gen:{gen} min_fitness_index:{min_fitness_index}\n')  
                f.write(f'best:\n{params_vals_array[:, min_fitness_index]}\n\n')
            with open('./tests/1005/pop1004.txt', 'a') as f:  # 使用'a'追加数据  
                f.write(f'gen:{gen} \n pop:\n{pop}\n\n')  
        
    def simulate_with_params(self, args):  
        # 拆分参数  
        f1, f21, z, k1, k2, k3, k4, k5, k6, k7, floor_param = args  

        # 实例化一次 OptimalParametersExperiment  
        op = OptimalParametersExperiment()  
        op.stiffness = floor_param['stiffness']  
        op.damping = floor_param['damping']  
        op.d0 = floor_param['d0']  
        op.d_width = floor_param['d_width']  
        op.width = floor_param['width']  
        op.rolling_friction = floor_param['rolling_friction']  
        op.torsional_friction = floor_param['torsional_friction']  

        # 进行模拟  
        _, height_list, _, _ = op.simulate_with_specific_params(  
            f1=f1, f21=f21, z=z,   
            k1=k1, k2=k2, k3=k3,   
            k4=k4, k5=k5, k6=k6, k7=k7,   
            stop_time=1, is_render=False  
        )

        # 返回结果  
        return 1 / max(height_list)  

    def func(self, params_vals):  
        f1_list, f21_list, z_list, k1_list, k2_list, k3_list, k4_list, k5_list, k6_list, k7_list = params_vals  
        
        floor_param = {  
            'stiffness': 6136.365161604287,  
            'damping': 488.8856222732014,  
            'd0': 0.8678732296264311,  
            'd_width': 0.29383619213963635,  
            'width': 0.0003514224555948409,  
            'torsional_friction': 0.45001884503858036,  
            'rolling_friction': 0.007074149832385092  
        }  

        # 准备参数  
        args_list = [  
            (f1_list[i], f21_list[i], z_list[i],   
            k1_list[i], k2_list[i], k3_list[i],  
            k4_list[i], k5_list[i], k6_list[i],   
            k7_list[i], floor_param)  
            for i in range(len(f1_list))  
        ]  

        # 使用 multiprocessing.Pool 来进行多进程计算  
        with Pool() as pool:  
            result = pool.map(self.simulate_with_params, args_list)  

        return result  
    
def experiment_with_ga():
    # # 将pop从文件中读取出来
    # with open('./tests/1003/tmp.txt', 'r') as f:  
    #     string = f.read()
    # cleaned_string = string.replace('[', '').replace(']', '')  # 去掉方括号  
    # cleaned_string = cleaned_string.replace('\n', ' ')  # 去掉换行符  
    # data_array = np.fromstring(cleaned_string, sep=' ').reshape(96, 100)  # 100是每行的元素个数  
    params_bound = [
            (50, 150),   # f1
            (150, 350),  # f21
            (0, 165),    # z
            (100, 1000000),   # k1
            (100, 1000000),   # k2
            (100, 1000000),   # k3
            (100, 1000000),   # k4
            (100, 1000000),   # k5
            (100, 1000000),   # k6
            (100, 1000000)    # k7
        ]
    num_params = len(params_bound)  # 参数个数
    dna_num = num_params    # DNA数目也即参数数目
    dna_size = 16            # 每个DNA长度, 应该与参数的分辨率有关 
    ga = GeneticAlgroithm()
    params_vals_initial = ga.get_params_vals_from_json('./tests/1003/config_gen_1.json')
    pop_initial = ga.decode_parameters(params_vals_initial, params_bound, dna_num, dna_size)
    ga.run(pop=pop_initial)

def sensibility_test():
    floor_param = {'stiffness': 6136.365161604287, 'damping': 488.8856222732014, 'd0': 0.8678732296264311, 'd_width': 0.29383619213963635, 'width': 0.0003514224555948409, 'torsional_friction': 0.45001884503858036, 'rolling_friction': 0.007074149832385092}
    pe = ParallelExperiment()
    pe.stiffness = floor_param['stiffness']
    pe.damping = floor_param['damping']
    pe.d0 = floor_param['d0']
    pe.d_width = floor_param['d_width']
    pe.width = floor_param['width']
    pe.rolling_friction = floor_param['rolling_friction']
    pe.torsional_friction = floor_param['torsional_friction']

    # pe.simulate_with_specific_index(0)
    # pe.run_simulations()
    # pe.simulate_with_specific_index_long_bar_len(index=8)
    # pe.simulate_with_specific_index_f2(index=13)
    # pe.simulate_with_specific_index_f2(index=14)
    # pe.run_simulations_specify_name('simulate_with_specific_index_long_bar_len', 'index_long_bar_len_list')
    # pe.run_simulations_specify_name('simulate_with_specific_index_f1', 'index_f1_list')
    # pe.run_simulations_specify_name('simulate_with_specific_index_f2', 'index_f2_list')
    pe.run_simulations_specify_name('simulate_with_specific_index', 'index_list')
    # pe.run_simulations_specify_name('simulate_with_specific_index_k', 'index_f1_list')

def single_experiment():
    floor_param = {
            'stiffness': 6136.365161604287,  
            'damping': 488.8856222732014,  
            'd0': 0.8678732296264311,  
            'd_width': 0.29383619213963635,  
            'width': 0.0003514224555948409,  
            'torsional_friction': 0.45001884503858036,  
            'rolling_friction': 0.007074149832385092  
        }
    
    f1,f21,z,k1,k2,k3,k4,k5,k6,k7=[1.08570230e+02, 2.15525292e+02, 8.81208515e-01, 6.54814411e+05,
                        6.00253605e+05, 4.49265232e+04, 2.31510442e+05, 2.34302564e+05,
                        8.01972959e+05, 3.05995422e+02]
    
    k1 = 511410
    # k1 = 515420
    op = OptimalParametersExperiment()
    op = OptimalParametersExperiment()
    op.stiffness = floor_param['stiffness']  
    op.stiffness = floor_param['stiffness']  
    op.damping = floor_param['damping']  
    op.damping = floor_param['damping']  
    op.d0 = floor_param['d0']  
    op.d0 = floor_param['d0']  
    op.d_width = floor_param['d_width']  
    op.d_width = floor_param['d_width']  
    op.width = floor_param['width']  
    op.width = floor_param['width']  
    op.rolling_friction = floor_param['rolling_friction']  
    op.rolling_friction = floor_param['rolling_friction']  
    op.torsional_friction = floor_param['torsional_friction']  

    # 进行模拟  
    _, height_list, _, _ = op.simulate_with_specific_params(  
        f1=f1, f21=f21, z=z,   
        k1=k1, k2=k2, k3=k3,   
        k4=k4, k5=k5, k6=k6, k7=k7, 
        stop_time=1, is_render=True  ,xml_name='test1007', is_save_data=True, time_step=1e-6
    )
    print(max(height_list))

if __name__ == "__main__":
    # parallel_experiment()
    # ga_experiment()
    # ga = GeneticAlgroithm()
    # ga.run()
    # experiment_with_ga()
    # sensibility_test()
    single_experiment()

