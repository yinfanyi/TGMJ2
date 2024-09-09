# 生成单个模型，生成控制,生成传感器

import re
import ast
import os
import sys
from time import time, sleep
import xml.etree.ElementTree as ET

import pandas as pd
import pickle
import numpy as np
import keyboard
import matplotlib.pyplot as plt

import mujoco.viewer as mj_viewer
import mujoco as mj
from dm_control import mujoco as dmmj
from dm_control import mjcf

from xlsxConverter import XLSXConverter
from tt12_with_middle import TT12WithMiddle


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

class TT12_MJCF():
    def __init__(self) -> None:
        self._time_step = 0.001
        self._gravity = '0 0 -10'

        self._floor_solimp = '0.9 0.95 0.001 0.5 2.2'
        self._floor_solref = '0.02 1'
        self._floor_friction = "0.1 0.005 0.0001"
        self._floor_pos = '0 0 0'

        self.bar_edge_shape = None

        self.tt_model_df_list = []

        self._node_label_with_class = []

        self.mjcf_model = mjcf.RootElement(model='TT12_environment')
        
        
        # self._arena = floors.Floor()

        self._set_default_values()
        self._set_visual_attributes()
        self._create_textures_and_materials()
        self._add_floor_and_lights()
        self.set_bar_edge_shape('sphere')
    
    def generate_tt_model(self, detailed_csv_file_path, tt_model_name='tt_model', pos:list=[0, 0, 0]):
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

        self._load_data(detailed_csv_file_path, name = tt_model_name)

        tt_model =  mjcf.RootElement(model=tt_model_name)
        tt_model.default.tendon.set_attributes(
            limited=True,
            range="0 10",
            width="0.005",
            rgba="1 0 0 1",
            damping="10"
        )
        result = [sublist for sublist in self.tt_model_df_list if sublist[0] == tt_model_name]
        node_df = result[0][1]
        bar_df = result[0][2]
        string_df = result[0][3]
        ball_df = result[0][4]
        for idx, row in node_df.iterrows():
            data_dict = ast.literal_eval(row['Data'])
            position = data_dict['position']    # 张拉模型在自己坐标系下的位置
            # print('position', position)
            updated_position = (position[0] + pos[0], position[1] + pos[1], position[2] + pos[2])
            # updated_position = tuple(val + pos[i] if isinstance(val, float) else val for i, val in enumerate(position))
            # print('updated_position', updated_position)
            data_dict['position'] = updated_position
            node_df.at[idx, 'Data'] = str(data_dict)
        
        self.generate_all_bars(tt_model, bar_df, node_df)
        self.generate_middle_platform(tt_model, ball_df, node_df)
        self.generate_all_tendons(tt_model, string_df)
        self.add_exclude_contacts(tt_model, bar_df)
        self.add_actuator(tt_model)
        self.add_sensor(tt_model)
    
        self.mjcf_model.attach(tt_model)
        
    def _load_data(self, detailed_csv_file_path: str, name='tt_model') -> None:
        data = pd.read_csv(detailed_csv_file_path)
        node_df = data[data['Type'] == 'Node']
        bar_df = data[data['Type'] == 'Bar']
        string_df = data[data['Type'] == 'String']
        ball_df =  data[data['Type'] == 'Ball']
        tt_model = [name, node_df, bar_df, string_df, ball_df]
        self.tt_model_df_list.append(tt_model)

    def _set_default_values(self) -> None:
        self.mjcf_model.compiler.angle = 'degree'
        self.mjcf_model.compiler.coordinate = "local"
        self.mjcf_model.compiler.inertiafromgeom = "true"
        self.mjcf_model.compiler.autolimits = "true"

        self.mjcf_model.option.timestep = self._time_step
        self.mjcf_model.option.integrator = 'RK4'
        self.mjcf_model.option.gravity = self._gravity
        self.mjcf_model.option.flag.energy = 'enable'
        self.mjcf_model.option.flag.override = 'enable'

        self.mjcf_model.default.joint.damping = 2
        self.mjcf_model.default.joint.type = 'hinge'
        self.mjcf_model.default.geom.type = 'capsule'
        self.mjcf_model.default.site.type = 'sphere'
        self.mjcf_model.default.site.size = '0.01'

        self.mjcf_model.default.tendon.set_attributes(
            limited=True,
            range="0 10",
            width="0.005",
            rgba="1 0 0 1",
            damping="10"
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
                                      type='plane', size=[0, 0, .125], 
                                      material=grid,
                                      friction=self._floor_friction,
                                      solimp=self._floor_solimp,
                                      solref=self._floor_solref,
                                      pos=self._floor_pos)

    def _add_floor_and_lights(self) -> None:
        for x in [-2, 2]:
            self.mjcf_model.worldbody.add('light', name=f'light_{x}', pos=[x, -1, 3], dir=[-x, 1, -2])
        self.mjcf_model.worldbody.add('site', name='world_site', pos='0 0 0')

    def get_midpoint_coords(self, coord1, coord2):
        return [(coord1[i] + coord2[i]) / 2 for i in range(3)]

    def extract_node_number(self, node_label):
        match = re.match(r'node_(\d+)', node_label)
        return match.group(1)

    def find_node_coordinates(self, labels, node_df):
        node_coords = []
        for _, node_row in node_df.iterrows():
            node_data_dict = ast.literal_eval(node_row['Data'])
            node_label = node_data_dict['label']
            # print(node_label)
            if any(label == node_label for label in labels):
                node_coords.append(node_data_dict['position'])
                # print(node_coords)
        return node_coords

    def generate_all_bars(self, tt_model:mjcf.element.RootElement, bar_df, node_df):
        for _, row in bar_df.iterrows():
            data_dict = ast.literal_eval(row['Data'])
            label = data_dict['label']
            density = data_dict['density'] 
            mass = data_dict['mass']
            from_node_label = data_dict['from_node_label']
            to_node_label = data_dict['to_node_label']
            alt_node_label = data_dict['alt_node_label']

            node_coords = self.find_node_coordinates([from_node_label, to_node_label, alt_node_label], node_df)
            from_coord, to_coord, alt_node_coord = node_coords

            midpoint_coords = self.get_midpoint_coords(from_coord, to_coord)
            midpoint_coords_str = " ".join(map(str, midpoint_coords))

            from_coord_local = [x - y for x, y in zip(from_coord, midpoint_coords)]
            to_coord_local = [x - y for x, y in zip(to_coord, midpoint_coords)]
            
            fromto_coords_str = " ".join(map(str, from_coord + to_coord))
            fromto_coords_local_str = " ".join(map(str, from_coord_local + to_coord_local))
            from_point_coords_str = " ".join(map(str, from_coord))
            from_point_coords_local_str = " ".join(map(str, from_coord_local))
            to_point_coords_str = " ".join(map(str, to_coord))
            to_point_coords_local_str = " ".join(map(str, to_coord_local))
            alt_point_coords_str = " ".join(map(str, alt_node_coord))

            x_str = self.extract_node_number(from_node_label)
            y_str = self.extract_node_number(to_node_label)

            bar_middle_site = tt_model.worldbody.add(
                'site', name=f'middle_site_{x_str}_{y_str}', pos=midpoint_coords, rgba='0 0 0 0')
            
            new_model = mjcf.RootElement(model=f'rod{x_str}_{y_str}')
            new_geom = new_model.worldbody.add('geom', name=f'geom{x_str}_{y_str}', type='cylinder', fromto=fromto_coords_local_str, mass=str(mass), size='0.012')
            new_model.worldbody.add('site', name=from_node_label, pos=from_point_coords_local_str)
            new_model.worldbody.add('site', name=to_node_label, pos=to_point_coords_local_str)
            new_model.worldbody.add('site', name=alt_node_label, pos='0 0 0')

            self._node_label_with_class.append(f'rod{x_str}_{y_str}/{from_node_label}')
            self._node_label_with_class.append(f'rod{x_str}_{y_str}/{to_node_label}')
            self._node_label_with_class.append(f'rod{x_str}_{y_str}/{alt_node_label}')


            if self.bar_edge_shape == 'box':
                new_model.worldbody.add('geom', 
                                    name=f'geom_edge_{x_str}', type='box', 
                                    pos=from_point_coords_local_str, mass=str(0.0001), 
                                    size='0.02 0.02 0.02', rgba="0 0.9 0 0.5")
                
                new_model.worldbody.add('geom', 
                                    name=f'geom_edge_{y_str}', type='box', 
                                    pos=to_point_coords_local_str, mass=str(0.0001), 
                                    size='0.02 0.02 0.02', rgba="0 0.9 0 0.5")
            elif self.bar_edge_shape == 'sphere':
                new_model.worldbody.add('geom', 
                                    name=f'geom_edge_{x_str}', type='sphere', 
                                    pos=from_point_coords_local_str, mass=str(0.0113), 
                                    size='0.03', rgba="0 0.9 0 0.5")
                
                new_model.worldbody.add('geom', 
                                    name=f'geom_edge_{y_str}', type='sphere', 
                                    pos=to_point_coords_local_str, mass=str(0.0113), 
                                    size='0.03', rgba="0 0.9 0 0.5")
            
            attachment_frame = bar_middle_site.attach(new_model)
            attachment_frame.add('freejoint')

    def generate_middle_platform(self, tt_model:mjcf.element.RootElement, ball_df, node_df):
        data_dict = ast.literal_eval(ball_df.iloc[0]['Data'])
        # print(data_dict)
        label = data_dict['label']
        density = data_dict['density']
        mass = data_dict['mass']
        radius = data_dict['radius']
        middle_node_label = data_dict['middle_node_label']
        connected_node_label_list = data_dict['connected_node_label_list']
        
        middle_node_coord = self.find_node_coordinates([middle_node_label], node_df)[0]
        middle_node_coord_str = " ".join(map(str, middle_node_coord))

        middle_element_site = tt_model.worldbody.add(
                'site', name=f'middle_element_site', pos=middle_node_coord_str, rgba='0 0 0 0')

        new_model = mjcf.RootElement(model='middle_platform')
        for connected_node_label in connected_node_label_list:
            connected_node_coord = self.find_node_coordinates([connected_node_label], node_df)[0]
            connected_coord_local = [x - y for x, y in zip(connected_node_coord, middle_node_coord)]
            connected_node_coord_local_str = " ".join(map(str, connected_coord_local))
            new_model.worldbody.add('site', name=connected_node_label, pos=connected_node_coord_local_str)
            self._node_label_with_class.append(f'middle_platform/{connected_node_label}')

        new_model.worldbody.add('site', name='outside_site', pos='0 0 0', size='0.2', rgba='1 0.9 0 0.5')
        
        new_body = new_model.worldbody.add('body', name=f'inside_ball', pos='0 0 0')
        new_body.add('inertial', pos='0 0 0', mass='6.6', diaginertia='0.12 0.12 0.12')
        new_body.add('joint', type='hinge', name='x_control', axis="1 0 0", armature="1.0", damping='0.1')
        new_body.add('joint', type='hinge', name='y_control', axis="0 1 0")
        new_body.add('joint', type='hinge', name='z_control', axis="0 0 1")
        new_body.add('site', name='inside_site', pos="0 0 0", type='box', size='0.1 0.1 0.1', rgba='0 0.9 0 1')
        
        attachment_frame = middle_element_site.attach(new_model)
        attachment_frame.add('freejoint')
        attachment_frame.add('inertial', pos='0 0 0', mass='0.7', diaginertia='0.02 0.02 0.02')

    def generate_all_tendons(self, tt_model:mjcf.element.RootElement, string_df):
        for _, row in string_df.iterrows():
            data_dict = ast.literal_eval(row['Data'])
            label = data_dict['label']
            from_node_label = data_dict['from_node_label']
            stiffness_str = str(data_dict['stiffness'])
            to_node_label = data_dict['to_node_label']
            ori_length = data_dict['ori_length']
            color = data_dict['color']
            x_str = self.extract_node_number(from_node_label)
            y_str = self.extract_node_number(to_node_label)
            string_name = f'td{x_str}_{y_str}'

            spatial = tt_model.tendon.add('spatial', name=string_name, stiffness=stiffness_str,
                                       springlength=ori_length, rgba=color)
            
            to_node_label_with_class = next((elem for elem in self._node_label_with_class if elem.endswith(f'/{to_node_label}')), None)
            from_node_label_with_class = next((elem for elem in self._node_label_with_class if elem.endswith(f'/{from_node_label}')), None)

            spatial.add('site', site=from_node_label_with_class)
            spatial.add('site', site=to_node_label_with_class)

    def add_exclude_contacts(self, tt_model:mjcf.element.RootElement, bar_df):
        rod_name_list = []
        for _, row in bar_df.iterrows():
            data_dict = ast.literal_eval(row['Data'])
            from_node_label = data_dict['from_node_label']
            to_node_label = data_dict['to_node_label']
            x_str = self.extract_node_number(from_node_label)
            y_str = self.extract_node_number(to_node_label)
            body_name=f'rod{x_str}_{y_str}'
            rod_name_list.append(body_name)

        for i in range(len(rod_name_list)):
            for j in range(i+1, len(rod_name_list)):
                body1 = rod_name_list[i]
                body2 = rod_name_list[j]
                tt_model.contact.add('exclude', name=f'exclude_{body1}_{body2}', body1=f'{body1}/', body2=f'{body2}/')
    
    def add_actuator(self, tt_model:mjcf.element.RootElement):
        tt_model.actuator.add('motor',name="torque_x_ctrl", joint="middle_platform/x_control", gear="1", ctrlrange="-10 10", ctrllimited="true")
        tt_model.actuator.add('velocity',name="velocity_x_ctrl", joint="middle_platform/x_control", kv="10", ctrlrange="0 100")
    
    def add_sensor(self, tt_model:mjcf.element.RootElement):
        tt_model.sensor.add('framepos',name="pos_outside_ball", objtype="body", objname="middle_platform/", reftype="site", refname="world_site")
        tt_model.sensor.add('framepos',name="pos_inside_ball", objtype="body", objname="middle_platform/inside_ball", reftype="site", refname="world_site")
        
        tt_model.sensor.add('frameangvel',name="angvel_outside_ball", objtype="body", objname="middle_platform/", reftype="site", refname="world_site")
        tt_model.sensor.add('frameangvel',name="angvel_inside_ball", objtype="body", objname="middle_platform/inside_ball", reftype="site", refname="world_site")
        
        tt_model.sensor.add('frameangacc',name="angacc_outside_ball", objtype="body", objname="middle_platform/")
        tt_model.sensor.add('frameangacc',name="angacc_inside_ball", objtype="body", objname="middle_platform/inside_ball")
        
        tt_model.sensor.add('accelerometer',site='middle_platform/inside_site')
        
        # tt_model.sensor.add('framepos',name="pos_outside_ball", joint="middle_platform/x_control", kv="10", ctrlrange="0 100")

    def set_model_suspended_in_the_air(self):
        self.gravity = '0 0 0'
    
    def set_model_on_the_ground(self):
        self.gravity = '0 0 -10'
        self.floor_pos = '0 0 0.62'
    
    def set_bar_edge_shape(self, shape:str):
        self.bar_edge_shape = shape

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
        self.mjcf_model.option.timestep = self._time_step

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, gravity):
        self._gravity = gravity
        self.mjcf_model.option.gravity = self._gravity

    def test(self):
        print(self.mjcf_model.find(namespace='geom',identifier='floor'))

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
        # 把worldbody下所有name="tt_model的body删掉，该body下的东西不删掉，也就是说该body下的内容移到上一层
        
        # bodies = list(worldbody.findall('body'))
        # for body in bodies:
        #     worldbody.remove(body)
    
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

    def export_to_xml_file(self, export_xml_file_path:str, file_name:str):
        # xml_string = self.mjcf_model.to_xml_string()
        mjcf.export_with_assets(self.mjcf_model, 
                                export_xml_file_path, 
                                file_name)
        xml_path = export_xml_file_path + file_name
        self.delete_external_body(xml_path)
        self.correct_sensor_refname(xml_path)

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
        self.intside_x_pos_datas = []

        self.angvel_inside_ball_datas = []
        self.angvel_outside_ball_datas = []

        self.angacc_inside_ball_datas = []
        self.angacc_outside_ball_datas = []
        
        self.torque_datas = []
        # plt.figure()

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

    def plot_data(self):
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
    def __init__(self, mjcf_file_name:str) -> None:
        self.mjcf_file_name = mjcf_file_name
        self.physics = dmmj.Physics.from_xml_path(self.mjcf_file_name)

        self.Hz = 20
        self.model = mj.MjModel.from_xml_path(self.mjcf_file_name)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data

        self.recorded_data = record_data()
        self.init()

    def init(self):
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        # mj.mj_resetData(self.model, self.data)
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

    def controller(self, model, data):
        velocity_x_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "tt_model_0/velocity_x_ctrl")
        torque_x_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "tt_model_0/torque_x_ctrl")
        self.data.ctrl[velocity_x_actuator_id] = 100

    def plot_scene(self):
        pixels = self.physics.render()
        plt.imshow(pixels)
        plt.show()
    
    def reload(self):
        self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data

    def simulate_step(self):
        current_simstart = self.data.time   # 当前一帧的开始时间
        while (self.data.time - current_simstart < 1.0/self.Hz):
            mj.mj_step(self.model, self.data)

    def simulate(self):
        # mj.set_mjcb_control(self.controller)
        is_paused = False
        plot_counter = 0
        plot_frequency = 500  # 设置调用频率
        with mj_viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                if keyboard.is_pressed('esc'):  
                    print("Exiting simulation...")  
                    break
                # if self.data.time > 50:
                #     # Stop the simulation after 20 simulated seconds  
                #     print("Exiting simulation...")  
                #     break
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
                if keyboard.is_pressed('a'):
                    self.test()
                    while keyboard.is_pressed('a'):  
                        pass
                # Only step the simulation if not paused  
                if not is_paused:  
                    self.recorded_data.bind_data(time=self.data.time,
                                             # 位置                     
                                             outside_x_pos=self.data.sensor('tt_model_0/pos_outside_ball').data[0],
                                             intside_x_pos=self.data.sensor('tt_model_0/pos_inside_ball').data[0],
                                             # 速度
                                             angvel_inside_ball = self.data.sensor('tt_model_0/angvel_inside_ball').data[0],
                                             angvel_outside_ball = self.data.sensor('tt_model_0/angvel_outside_ball').data[0],
                                             # 加速度
                                             angacc_inside_ball = self.data.sensor('tt_model_0/angacc_inside_ball').data[0],
                                             angacc_outside_ball = self.data.sensor('tt_model_0/angacc_outside_ball').data[0],
                                             # 控制变量
                                             ctrl_v_x=self.data.ctrl[1])
                    self.simulate_step()
                # if plot_counter % plot_frequency == 0:
                #     self.recorded_data.plot_data_simultaneous()
                #     plot_counter -= plot_frequency
                # plot_counter += 1
                viewer.sync()

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


def main0610():
    detailed_csv_file_path = './data/csv/data_tt12bar.csv'
    export_xml_file_path = './data/xml/'
    export_xml_file_name = "TT12_0610.xml"
    xml_file_name = export_xml_file_path + export_xml_file_name

    case = 2
    # 生成模型
    if(case == 1):
        tt12_mjcf = TT12_MJCF()
        # set property
        tt12_mjcf.time_step = 0.001
        # tt12_mjcf.floor_friction = '0.1 0.005 0.0001'
        # tt12_mjcf.floor_solref = '-100000 -10'
        
        import math
        circle_radius = 2
        num_models = 10
        angle_increment = 2 * math.pi / num_models

        tt12_mjcf.generate_tt_model(detailed_csv_file_path,tt_model_name='tt_model_0', pos=(0,0,0))
        tt12_mjcf.export_to_xml_file(export_xml_file_path, export_xml_file_name)

        ## tt12_mjcf.test()

    # 控制模型
    if(case == 2):
        tt12_control = TT12_Control(xml_file_name)
        tt12_control.simulate()

    # 生成并查看模型
    if(case == 3):
        tt12_mjcf = TT12_MJCF(detailed_csv_file_path)
        # set property
        tt12_mjcf.time_step = 0.0001
        tt12_mjcf.floor_friction = '1 0.005 0.0001'
        # tt12_mjcf.floor_solref = '-100000 -10'
        
        tt12_mjcf.generate_all_bars()
        tt12_mjcf.generate_middle_platform()
        tt12_mjcf.generate_all_tendons()
        tt12_mjcf.add_exclude_contacts()
        tt12_mjcf.add_actuator()
        # tt12_mjcf.set_model_suspended_in_the_air()
        tt12_mjcf.set_model_on_the_ground()

        export_xml_file_path = './data/xml/'
        export_xml_file_name = "TT12_0608.xml"

        tt12_mjcf.export_to_xml_file(export_xml_file_path, export_xml_file_name)

        xml_file_name = './data/xml/TT12_0608.xml'
        tt12_control = TT12_Control(xml_file_name)
        # tt12_control.plot_scene()
        tt12_control.viewer_scene()

def main0823():
    """生成最新的模型，查看模型，并做实验验证MuJoCo的准确性
    """
    xlsx_path = './data/xlsx/topology_TSR_flexible_strut_ball_foot_v1_R.xlsx'
    csv_path = './data/csv/data_tt12bar_0823.csv'
    export_xml_file_path = './data/xml/'
    export_xml_file_name = "TT12_0823.xml"
    xml_path = export_xml_file_path + export_xml_file_name

    case = 3
    if(case == 1):
        tt12_with_middle = TT12WithMiddle(xlsx_path)
        tt12_with_middle.fill_all_data()
        tt12_with_middle.export_to_csv(csv_path)
    if(case == 2):
        tt12_mjcf = TT12_MJCF()
        tt12_mjcf.time_step = 0.001
        # tt12_mjcf.floor_friction = '0.1 0.005 0.0001'
        # tt12_mjcf.floor_solref = '-100000 -10'
        
        tt12_mjcf.generate_tt_model(csv_path,tt_model_name='tt_model_0', pos=(0,0,0))
        tt12_mjcf.export_to_xml_file(export_xml_file_path, export_xml_file_name)
    if(case == 3):
        tt12_control = TT12_Control(xml_path)
        tt12_control.simulate()
    
    if(case == 4):
        tt12_with_middle = TT12WithMiddle(xlsx_path)
        tt12_with_middle.fill_all_data()
        tt12_with_middle.export_to_csv(csv_path)
        tt12_mjcf = TT12_MJCF()
        tt12_mjcf.time_step = 0.001
        # tt12_mjcf.floor_friction = '0.1 0.005 0.0001'
        # tt12_mjcf.floor_solref = '-100000 -10'
        tt12_mjcf.generate_tt_model(csv_path,tt_model_name='tt_model_0', pos=(0,0,0))
        tt12_mjcf.export_to_xml_file(export_xml_file_path, export_xml_file_name)
        tt12_control = TT12_Control(xml_path)
        tt12_control.simulate()

def main0824():
    """生成最新的模型，查看模型，并做实验验证MuJoCo的准确性
    """
    xlsx_path = './data/xlsx/topology_TSR_flexible_strut_ball_foot_v1_R.xlsx'
    csv_path = './data/csv/data_tt12bar_0823.csv'
    export_xml_file_path = './data/xml/'
    export_xml_file_name = "TT12_0824.xml"
    xml_path = export_xml_file_path + export_xml_file_name

    case = 3
    if(case == 1):
        tt12_with_middle = TT12WithMiddle(xlsx_path)
        tt12_with_middle.fill_all_data()
        tt12_with_middle.export_to_csv(csv_path)
    if(case == 2):
        tt12_mjcf = TT12_MJCF()
        tt12_mjcf.time_step = 0.001
        # tt12_mjcf.floor_friction = '0.1 0.005 0.0001'
        # tt12_mjcf.floor_solref = '-100000 -10'
        
        tt12_mjcf.generate_tt_model(csv_path,tt_model_name='tt_model_0', pos=(0,0,0))
        tt12_mjcf.export_to_xml_file(export_xml_file_path, export_xml_file_name)
    if(case == 3):
        tt12_control = TT12_Control(xml_path)
        tt12_control.Hz = 5
        tt12_control.simulate()
        tt12_control.recorded_data.plot_data()
        tt12_control.recorded_data.save_data(filename='./data/csv/data_tt12bar_0824_1.csv')
    
    if(case == 4):
        tt12_with_middle = TT12WithMiddle(xlsx_path)
        tt12_with_middle.fill_all_data()
        tt12_with_middle.export_to_csv(csv_path)
        tt12_mjcf = TT12_MJCF()
        tt12_mjcf.time_step = 0.001
        # tt12_mjcf.floor_friction = '0.1 0.005 0.0001'
        # tt12_mjcf.floor_solref = '-100000 -10'
        tt12_mjcf.generate_tt_model(csv_path,tt_model_name='tt_model_0', pos=(0,0,0))
        tt12_mjcf.export_to_xml_file(export_xml_file_path, export_xml_file_name)
        tt12_control = TT12_Control(xml_path)
        tt12_control.simulate()

if __name__ == "__main__":
    main0824()


