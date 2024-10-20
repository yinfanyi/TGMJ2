import time
import os
import sys

from concurrent import futures
import nevergrad as ng
import matplotlib.pyplot as plt
import numpy as np
import mujoco as mj
import pandas as pd
import xml.etree.ElementTree as ET

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

import tt12_control_bar_without_spring as tt_ctrl

class ParameterRecognize:
    def __init__(self, xml_file_path='./data/xml/0919/TT12_0919.xml') -> None:
        self.xml_file_path = xml_file_path
        self.get_adams_data()

    def update_parameter_in_xml(self, xml_file, stiffness=5773, 
                                        damping=68, 
                                        d0=0.27, 
                                        d_width=0.07, 
                                        width=0.06, 
                                        rolling_friction=0.005, 
                                        torsional_friction=0.001,
                                        tongue=40):
        tree = ET.parse(xml_file)
        worldbody = tree.find('worldbody')
        geoms = worldbody.findall('geom')
        for geom in geoms:
            if geom.get('name') == 'floor':
                geom.set('solref', f'{-stiffness} {-damping}')
                geom.set('solimp', f'{d0} {d_width} {width} 0.5 2')
                geom.set('friction', f'0.5 {rolling_friction} {torsional_friction}')
        keyframe = tree.find('keyframe')
        for key in keyframe.findall('key'):
            name = key.attrib.get('name')
            if name.endswith('initial_state'):
                key.attrib['ctrl'] = f'{-tongue} 0'
        return ET.tostring(tree.getroot(), encoding='unicode')

    def simulate_with_specific_parameter(self,
                                         stiffness=4360.39, 
                                         damping=138, 
                                         d0=0.96988, 
                                         d_width=0.10222, 
                                         width=0.055103, 
                                         rolling_friction=0.539, 
                                         torsional_friction=0.001924,
                                         stop_time=0.5,
                                         is_render=False,
                                         tongue=40):
        xml_file_path = self.xml_file_path
        xml_string = self.update_parameter_in_xml(xml_file_path, stiffness, damping, d0, d_width, width, rolling_friction, torsional_friction, tongue)
        tt12_control = tt_ctrl.TT12_Control(keyframe_id=0, xml_string=xml_string)  
        tt12_control.Hz = 1010
        tt12_control.is_control = True  
        tt12_control.simulate(is_render=is_render, stop_time=stop_time)
        return tt12_control.recorded_data.time_datas, tt12_control.recorded_data.outside_z_pos_datas, tt12_control.recorded_data.outside_y_pos_datas, tt12_control.recorded_data.angvel_outside_ball_datas  

    def plot_simulated_data(self, 
                            stiffness=4360.39, 
                            damping=138, 
                            d0=0.96988, 
                            d_width=0.10222, 
                            width=0.055103, 
                            rolling_friction=0.539, 
                            torsional_friction=0.001924,
                            stop_time=0.5):
        _, axs = plt.subplots(3, 1, figsize=(10, 15))  
        # 绘制高度(mm)的变化  
        time_list, outside_z_pos_list, outside_y_pos_list, angvel_outside_ball_list = self.simulate_with_specific_parameter(stiffness, damping, d0, d_width, width, rolling_friction, torsional_friction, stop_time)
        axs[0].plot(time_list, outside_z_pos_list)  
        axs[0].set_title('height(mm) vs time')  
        axs[0].set_xlabel('time (s)')  
        axs[0].set_ylabel('height (mm)')  

        # 绘制水平位置(mm)的变化  
        axs[1].plot(time_list, outside_y_pos_list)
        axs[1].set_title('forward position(mm) vs time')  
        axs[1].set_xlabel('time (s)')  
        axs[1].set_ylabel('forward position (mm)')  

        # 绘制角速度(rad/s)的变化  
        axs[2].plot(time_list, angvel_outside_ball_list)  
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

    def plot_comparison(self, adams_df, simulated_time_list, simulated_outside_z_pos_list, simulated_outside_y_pos_list, simulated_angvel_outside_ball_list):
        _, axs = plt.subplots(3, 1, figsize=(10, 15))
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
        
    def get_adams_data(self, analysis_directory=f'./tests/0919/analysis', save_processed_data=False):
        df_dict = np.load(f'{analysis_directory}/ana_all_data.npy', allow_pickle=True).item()
        self.data_dict = {}
        for i in range(240):
            filtered_df = df_dict[f'{i+1}'][(df_dict[f'{i+1}']['Time'] <= 0.5)]
            filtered_df = filtered_df.interpolate(method='linear', limit_direction='both').iloc[::10]  
            filtered_df.reset_index(drop=True, inplace=True)
            filtered_df = filtered_df[['Time', 'height_center', 'disp_x_exter_sphere', 'ang_vel_exter_sphere']]
            filtered_df['height_center'] = filtered_df['height_center'] * 0.001
            filtered_df['disp_x_exter_sphere'] = filtered_df['disp_x_exter_sphere'] * 0.001 - 1
            self.data_dict[f'{i+1}'] = filtered_df
        # 保存self.data_dict
        if save_processed_data:
            np.save(f'{analysis_directory}/processed_data_dict.npy', self.data_dict)

    def get_error(self, index=0,
                        stiffness=4360.39, 
                        damping=138, 
                        d0=0.96988, 
                        d_width=0.10222, 
                        width=0.055103, 
                        rolling_friction=0.539, 
                        torsional_friction=0.001924,
                        tongue=40,
                        is_plot=False,
                        is_render=False):
        adams_df = self.data_dict[f'{index+1}']
        # 平均0.8s一个实验
        simulated_time_list, simulated_outside_z_pos_list, simulated_outside_y_pos_list, simulated_angvel_outside_ball_list = self.simulate_with_specific_parameter(stiffness, damping, d0, d_width, width, rolling_friction, torsional_friction, tongue=tongue, is_render=is_render, stop_time=0.8)
        interpolated_values_angvel = np.interp(adams_df['Time'], simulated_time_list, simulated_angvel_outside_ball_list)
        interpolated_values_height = np.interp(adams_df['Time'], simulated_time_list, simulated_outside_z_pos_list)
        interpolated_values_forward_pos = np.interp(adams_df['Time'], simulated_time_list, simulated_outside_y_pos_list)

        real_pose = np.array([adams_df['height_center'], 0.5*adams_df['disp_x_exter_sphere']])  # 乘以0.5使水平移动的影响变小
        mujoco_pose = np.array([interpolated_values_height, 0.5 * interpolated_values_forward_pos])
        # print(f'real_pose: {real_pose}, mujoco_pose: {mujoco_pose}')

        ROBOT_HALF_WIDTH = 0.35
        e_pos = np.sum(np.linalg.norm(real_pose - mujoco_pose, axis=0))/ROBOT_HALF_WIDTH/15  # 18和77是为了让这两个误差算出来差不多
        e_vel = np.linalg.norm(adams_df['ang_vel_exter_sphere'] - interpolated_values_angvel, axis=0)/77
        if is_plot:
            self.plot_comparison(adams_df, adams_df['Time'], interpolated_values_height, interpolated_values_forward_pos, interpolated_values_angvel)
        return e_pos, e_vel
    
    def loss_function(self,
                        d0=0.96988, 
                        d_width=0.10222, 
                        width=0.055103, 
                        stiffness=4360.39, 
                        damping=138, 
                        rolling_friction=0.539, 
                        torsional_friction=0.001924):
        # 一个loss function大概花费20s
        tongues = np.linspace(10, 200, 240)
        e_pos_all = 0
        e_vel_all = 0
        num = 0
        # start_time = time.time()
        indexes = [i for i in range(40, 240, 10)]   # 20个 作为训练组
        for index in indexes:
            # print(f'tongue: {tongues[index]}')
            e_pos, e_vel = self.get_error(index, stiffness, damping, d0, d_width, width, rolling_friction, torsional_friction, tongue=tongues[index])
            e_pos_all += e_pos
            e_vel_all += e_vel
            num += 1
            end_time = time.time()
            # print(f'第{num}个实验, e_pos: {e_pos}, e_vel: {e_vel}, 用时: {end_time - start_time} s')
            
        e_pos_all /= num
        e_vel_all /= num
        # end_time = time.time()
        # print(f'Time used: {end_time - start_time} s')
        # print(f'e_pos_all: {e_pos_all}, e_vel_all: {e_vel_all}')
        return e_pos_all + e_vel_all
        
    def recognize_parameter(self, logger_path='./tests/0919/optimizer_log'):
        instrum = ng.p.Instrumentation(
        d0=ng.p.Scalar(lower=0.5, upper=0.999),
        d_width=ng.p.Scalar(lower=0.01, upper=0.5),
        width=ng.p.Scalar(lower=0.00001, upper=0.1),
        stiffness=ng.p.Scalar(lower=3000, upper=10000),
        damping=ng.p.Scalar(lower=80, upper=500),
        rolling_friction=ng.p.Scalar(lower=0.0001, upper=0.6),
        torsional_friction=ng.p.Scalar(lower=0.0001, upper=0.2),
        )

        # child = instrum.spawn_child()
        # child.value = ((), {'d0': 0.9433598140425187, 'd_width': 0.1395079405492468, 'width': 0.05208016638200587, 'stiffness': 4724.630709636312, 'damping': 91.11742334723378, 'rolling_friction': 0.5985903152412875, 'torsional_friction': 0.02774301352299431})
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=1000, num_workers=28)

        logger = ng.callbacks.ParametersLogger(logger_path)
        optimizer.register_callback("tell",  logger)
        
        start_time = time.time()
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(self.loss_function, executor=executor, batch_mode=False, verbosity=1)

        print('loss:',recommendation.loss)
        print('value:',recommendation.value[1])
        print('total time: ', time.time() - start_time)

if __name__ == '__main__':

    logger_path='./tests/0920/optimizer_log'
    # plot_recognize_result(logger_path)

    # result = {'d0': 0.9698819142575082, 'd_width': 0.10222059345169457, 'width': 0.055102813760108824, 'stiffness': 4360.38894615363, 'damping': 138.02062726106232, 'rolling_friction': 0.5391332783891921, 'torsional_friction': 0.019243073269965646}
    # 共花费1510s,得到
    result = {'d0': 0.9947420777976741, 'd_width': 0.10118368978786463, 'width': 0.08651887500734659, 'stiffness': 3941.5469027108334, 'damping': 314.19129602918065, 'rolling_friction': 0.10012849815288849, 'torsional_friction': 0.0007644265148328824}
    # xlsx_path = './tests/data_to_fit_large_stiffness.xlsx'
    d0 = result['d0']
    d_width = result['d_width']
    width = result['width']
    stiffness = result['stiffness']
    damping = result['damping']
    rolling_friction = result['rolling_friction']
    torsional_friction = result['torsional_friction']

    tongues = np.linspace(10, 200, 240)
    index = 200
    print('tongue:', tongues[index])
    param_recognize = ParameterRecognize()
    # is_render = False
    # param_recognize.simulate_with_specific_parameter(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, rolling_friction=rolling_friction, torsional_friction=torsional_friction, stop_time=1.5, tongue=50, is_render=True)
    # error = param_recognize.get_error(index=index, tongue=tongues[index], stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, rolling_friction=rolling_friction, torsional_friction=torsional_friction, is_plot=True, is_render=is_render)
    # print(error)
    param_recognize.recognize_parameter(logger_path)
    
    # param_reconize.plot_adams_df(choice='adams_df_list_first_few_seconds')
    # param_reconize.plot_simulated_data(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, rolling_friction=rolling_friction, torsional_friction=torsional_friction, stop_time=1.5)
    # print(param_reconize.loss_function_for_all_z_and_all_parameter_in_all_time(d0=d0, d_width=d_width, width=width, stiffness=stiffness, damping=damping, rolling_friction=rolling_friction, torsional_friction=torsional_friction))
    # print(param_reconize.loss_function_for_all_z_and_all_parameter_in_all_time())
    # analysis_performance('loss_function_for_all_z_and_height_parameter_in_first_few_seconds')