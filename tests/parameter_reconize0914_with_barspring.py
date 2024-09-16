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

import tt12_control_bar_with_spring as tt_ctrl

def main_0828():
    xml_path = './data/xml/TT12_0828.xml'
    xlsx_path = './tests/drop_test_0826.xlsx'

    start_time = time.time()
    print(loss_function())

    loss_history = []
    param_history = []

    instrum = ng.p.Instrumentation(
    d0=ng.p.Scalar(lower=0.01, upper=0.99),
    d_width=ng.p.Scalar(lower=0.01, upper=0.99),
    width=ng.p.Scalar(lower=0.0001, upper=0.1),
    stiffness=ng.p.Scalar(lower=1000, upper=10000),
    damping=ng.p.Scalar(lower=0.01, upper=100),
    )

    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=1000, num_workers=30)
    # optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint (parametrization=instrum, budget=100, num_workers=20)

    logger_filepath = './data/test_logger.json'
    logger = ng.callbacks.ParametersLogger(logger_filepath)
    optimizer.register_callback("tell",  logger)
    
    with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(loss_function, executor=executor, batch_mode=True, verbosity=2)

    list_of_dict_of_data = logger.load()

    print('loss:',recommendation.loss)
    print('value:',recommendation.value[1])
    print('total time: ', time.time() - start_time)
    print('list_of_dict_of_data: ', list_of_dict_of_data)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_history, marker='o')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    plt.title('Loss History')

    # plt.subplot(1, 2, 2)
    # # params = list(zip(*param_history))
    # # print(param_history)
    # stiffness_values = [param[1]['stiffness'] for param in param_history]
    # damping_values = [param[1]['damping'] for param in param_history]
    # friction_values = [param[1]['friction'] for param in param_history]
    # plt.plot(stiffness_values, marker='o', label=f'param{i}')

    # plt.xlabel('Iteration')
    # plt.ylabel('Parameter Value')
    # plt.title('Parameter History')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

class ParameterReconize:
    def __init__(self, 
                 xml_folder_path='./data/xml/0914_z', 
                 adams_data_path='./tests/data_to_fit.xlsx', 
                 z_list=[165, 162, 160, 150],
                 first_few_seconds=0.2,
                 stop_time=1.5) -> None:
        self.xml_folder_path = xml_folder_path
        self.stop_time = stop_time
        self.z_list = z_list
        self.adams_df_list = []
        for z in self.z_list:
            adams_df = pd.read_excel(adams_data_path, sheet_name=f'z{z}')
            adams_df['高度(mm)'] = adams_df['高度(mm)'] * 0.001
            adams_df['水平位置(mm)'] = adams_df['水平位置(mm)'] * 0.001 - 1
            self.adams_df_list.append(adams_df)
        # 获取前0.2s的数据，储存在adams_df_list_first_few_seconds里
        self.adams_df_list_first_few_seconds = []
        for adams_df in self.adams_df_list:
            adams_df_first_few_seconds = adams_df[adams_df['Time'] < first_few_seconds]
            self.adams_df_list_first_few_seconds.append(adams_df_first_few_seconds)

    def create_xml_file():
        tt_ctrl.check_and_create_folder('./data/csv/output0914_z')
        tt_ctrl.check_and_create_folder('./data/csv/0914_z')
        tt_ctrl.check_and_create_folder('./data/xml/0914_z')
        z_list = [165, 162, 160, 150]
        for i, z in enumerate(z_list):
            tt_ctrl.process_z(z=z, index=i, folder_name='0914_z')

    def plot_adams_df(self, choice='adams_df_list'):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))  
        # 绘制高度(mm)的变化  
        if choice == 'adams_df_list':
            choice = self.adams_df_list
        elif choice == 'adams_df_list_first_few_seconds':
            choice = self.adams_df_list_first_few_seconds
        for i, df in enumerate(choice):
            axs[0].plot(df['Time'], df['高度(mm)'], label=f'z={self.z_list[i]}')  
            axs[0].set_title('height(mm) vs time')  
            axs[0].set_xlabel('time (s)')  
            axs[0].set_ylabel('height (mm)')  

            # 绘制水平位置(mm)的变化  
            axs[1].plot(df['Time'], df['水平位置(mm)'], label=f'z={self.z_list[i]}')  
            axs[1].set_title('forward position(mm) vs time')  
            axs[1].set_xlabel('time (s)')  
            axs[1].set_ylabel('forward position (mm)')  

            # 绘制角速度(rad/s)的变化  
            axs[2].plot(df['Time'], df['角速度(rad/s)'], label=f'z={self.z_list[i]}')  
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

    def plot_simulated_data(self, 
                            stiffness=5773, 
                            damping=68, 
                            d0=0.27, 
                            d_width=0.07, 
                            width=0.06, 
                            rolling_friction=0.005, 
                            torsional_friction=0.001,
                            stop_time=1.5):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))  
        # 绘制高度(mm)的变化  
        for i in range(len(self.z_list)):
            time_list, outside_z_pos_list, outside_y_pos_list, angvel_outside_ball_list = self.simulate_with_specific_parameter(i, stiffness, damping, d0, d_width, width, rolling_friction, torsional_friction, stop_time)
            axs[0].plot(time_list, outside_z_pos_list, label=f'z={self.z_list[i]}')  
            axs[0].set_title('height(mm) vs time')  
            axs[0].set_xlabel('time (s)')  
            axs[0].set_ylabel('height (mm)')  

            # 绘制水平位置(mm)的变化  
            axs[1].plot(time_list, outside_y_pos_list, label=f'z={self.z_list[i]}')  
            axs[1].set_title('forward position(mm) vs time')  
            axs[1].set_xlabel('time (s)')  
            axs[1].set_ylabel('forward position (mm)')  

            # 绘制角速度(rad/s)的变化  
            axs[2].plot(time_list, angvel_outside_ball_list, label=f'z={self.z_list[i]}')  
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

    def update_parameter_in_xml(self, xml_file, stiffness=5773, 
                                        damping=68, 
                                        d0=0.27, 
                                        d_width=0.07, 
                                        width=0.06, 
                                        rolling_friction=0.005, 
                                        torsional_friction=0.001):
        tree = ET.parse(xml_file)
        worldbody = tree.find('worldbody')
        geoms = worldbody.findall('geom')
        for geom in geoms:
            if geom.get('name') == 'floor':
                geom.set('solref', f'{-stiffness} {-damping}')
                geom.set('solimp', f'{d0} {d_width} {width} 0.5 2')
                geom.set('friction', f'0.5 {rolling_friction} {torsional_friction}')
        return ET.tostring(tree.getroot(), encoding='unicode')

    def simulate_with_specific_parameter(self, index=0, 
                                         stiffness=5773, 
                                         damping=68, 
                                         d0=0.27, 
                                         d_width=0.07, 
                                         width=0.06, 
                                         rolling_friction=0.005, 
                                         torsional_friction=0.001,
                                         stop_time=1.5,
                                         is_render=False,
                                         xml_file_path=None):
        if xml_file_path is None:
            xml_file_path = os.path.join(self.xml_folder_path, f'TT12_0911_{index}.xml')
        xml_string = self.update_parameter_in_xml(xml_file_path, stiffness, damping, d0, d_width, width, rolling_friction, torsional_friction)
        tt12_control = tt_ctrl.TT12_Control(keyframe_id=0, xml_string=xml_string)  
        tt12_control.Hz = 1010
        tt12_control.Hz = 20000
        tt12_control.is_control = True  
        tt12_control.simulate(is_render=is_render, stop_time=stop_time)
        return tt12_control.recorded_data.time_datas, tt12_control.recorded_data.outside_z_pos_datas, tt12_control.recorded_data.outside_y_pos_datas, tt12_control.recorded_data.angvel_outside_ball_datas  

    def loss_function_for_all_z_and_all_parameter_in_all_time(self, 
                                                              d0=0.27, d_width=0.07, width=0.06, 
                                                              stiffness=5773, 
                                                              damping=68, 
                                                              rolling_friction=0.005, 
                                                              torsional_friction=0.001):
        sum_difference = 0
        for i in range(len(self.z_list)):
            

            time_list, outside_z_pos_list, outside_y_pos_list, angvel_outside_ball_list = self.simulate_with_specific_parameter(i, stiffness, damping, d0, d_width, width, rolling_friction, torsional_friction, stop_time=self.stop_time)
            interpolated_values_angvel = np.interp(self.adams_df_list_first_few_seconds[i]['Time'], time_list, angvel_outside_ball_list)
            interpolated_values_height = np.interp(self.adams_df_list_first_few_seconds[i]['Time'], time_list, outside_z_pos_list)
            interpolated_values_forward_pos = np.interp(self.adams_df_list_first_few_seconds[i]['Time'], time_list, outside_y_pos_list)

            angvel_coefficient = 0.1
            height_coefficient = 0.5
            forward_pos_coefficient = 1
            # 设置滑动平均窗口大小  
            window_size = 3  # 根据需要调整窗口大小  

            # 计算差异并应用滑动平均  
            angvel_raw = self.adams_df_list_first_few_seconds[i]['角速度(rad/s)']  
            height_raw = self.adams_df_list_first_few_seconds[i]['高度(mm)']  
            forward_pos_raw = self.adams_df_list_first_few_seconds[i]['水平位置(mm)']  

            # 计算滑动平均  
            interpolated_values_angvel_ma = moving_average(interpolated_values_angvel, window_size)  
            interpolated_values_height_ma = moving_average(interpolated_values_height, window_size)  
            interpolated_values_forward_pos_ma = moving_average(interpolated_values_forward_pos, window_size)  

            # 计算差异并归一化  
            angvel_difference = np.abs(moving_average(angvel_raw, window_size) - interpolated_values_angvel_ma)  
            angvel_difference = angvel_difference / np.sum(angvel_difference, axis=0)  

            height_difference = np.abs(moving_average(height_raw, window_size) - interpolated_values_height_ma)  
            height_difference = height_difference / np.sum(height_difference, axis=0)  

            forward_pos_difference = np.abs(moving_average(forward_pos_raw, window_size) - interpolated_values_forward_pos_ma)  
            forward_pos_difference = forward_pos_difference / np.sum(forward_pos_difference, axis=0)  

            # 使用平方差  
            absolute_difference = (angvel_coefficient * angvel_difference ** 2 +  
                                height_coefficient * height_difference ** 2 +  
                                forward_pos_coefficient * forward_pos_difference ** 2)  

            # 加权平均  
            weights = np.linspace(1, 0, len(absolute_difference))
            weights = 1 - np.power(weights, 0.25)  
            absolute_difference = np.multiply(absolute_difference, weights)  

            # 引入正则化项（L2正则化示例）  
            regularization_strength = 0.01  
            regularization_term = regularization_strength * (angvel_coefficient**2 + height_coefficient**2 + forward_pos_coefficient**2)  

            # 计算总损失  
            sum_difference += np.sum(absolute_difference, axis=0) + regularization_term 
        return sum_difference

    def loss_function_for_all_z_and_height_parameter_in_first_few_seconds(self, d0=0.27, d_width=0.07, width=0.06, stiffness=5773, damping=68):
        sum_difference = 0
        for i in range(len(self.z_list)):
            time_list, outside_z_pos_list, _, _ = self.simulate_with_specific_parameter(i, stiffness, damping, d0, d_width, width, stop_time=self.stop_time)
            interpolated_values_height = np.interp(self.adams_df_list_first_few_seconds[i]['Time'], time_list, outside_z_pos_list)
            absolute_difference = np.abs(self.adams_df_list_first_few_seconds[i]['高度(mm)'] - interpolated_values_height)
            sum_difference += np.sum(absolute_difference, axis=0)
        return sum_difference

def moving_average(data, window_size):  
    """计算滑动平均"""  
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')  

def analysis_performance(function_name='loss_function_for_all_z_and_all_parameter_in_all_time'):
    import cProfile
    cProfile.run(f'print(ParameterReconize().{function_name}())')

def plot_recognize_result(result_file='./tests/0914_parameter_reconize'):
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
    with open(result_file, 'r') as file:  
        for line_number, line in enumerate(file):  
            data = json.loads(line.strip())  
            line_number_list.append(line_number)  
            loss_list.append(data.get("#loss"))  
            d0_list.append(data.get("d0"))  
            d_width_list.append(data.get("d_width"))  
            width_list.append(data.get("width"))  
            stiffness_list.append(data.get("stiffness"))  
            damping_list.append(data.get("damping"))  
            rolling_friction_list.append(data.get("rolling_friction"))  
            torsional_friction_list.append(data.get("torsional_friction"))  

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

def recognize_parameter_first_0_35_seconds_and_only_height():
    # 只参数辨识前0.35s的高度数据，得到结果不太正确。
    xlsx_path = './tests/data_to_fit_large_stiffness.xlsx'
    workplace_folder = './tests/0914_parameter_reconize/'
    param_reconize = ParameterReconize(adams_data_path=xlsx_path, z_list=[165, 162, 160, 150], first_few_seconds=0.35, stop_time=0.35)
    loss_history = []
    param_history = []

    instrum = ng.p.Instrumentation(
    d0=ng.p.Scalar(lower=0.01, upper=0.99),
    d_width=ng.p.Scalar(lower=0.01, upper=0.99),
    width=ng.p.Scalar(lower=0.0001, upper=0.1),
    stiffness=ng.p.Scalar(lower=1000, upper=10000),
    damping=ng.p.Scalar(lower=0.01, upper=100),
    )

    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=1000, num_workers=30)
    # optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint (parametrization=instrum, budget=100, num_workers=20)

    logger = ng.callbacks.ParametersLogger(workplace_folder)
    optimizer.register_callback("tell",  logger)
    
    start_time = time.time()
    with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(param_reconize.loss_function_for_all_z_and_height_parameter_in_first_few_seconds, executor=executor, batch_mode=True, verbosity=2)

    print('loss:',recommendation.loss)
    print('value:',recommendation.value[1])
    print('total time: ', time.time() - start_time)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_history, marker='o')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss History')

    # plt.subplot(1, 2, 2)
    # # params = list(zip(*param_history))
    # # print(param_history)
    # stiffness_values = [param[1]['stiffness'] for param in param_history]
    # damping_values = [param[1]['damping'] for param in param_history]
    # friction_values = [param[1]['friction'] for param in param_history]
    # plt.plot(stiffness_values, marker='o', label=f'param{i}')

    # plt.xlabel('Iteration')
    # plt.ylabel('Parameter Value')
    # plt.title('Parameter History')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

def recognize_parameter_first_1_seconds_and_all_parameter(logger_path='./tests/0915/recognize_parameter_first_1_seconds_and_all_parameter'):
    xlsx_path = './tests/data_to_fit_large_stiffness.xlsx'
    param_reconize = ParameterReconize(adams_data_path=xlsx_path, z_list=[165, 162, 160, 150], first_few_seconds=1, stop_time=1)

    instrum = ng.p.Instrumentation(
    d0=ng.p.Scalar(lower=0.5, upper=0.999),
    d_width=ng.p.Scalar(lower=0.01, upper=0.5),
    width=ng.p.Scalar(lower=0.00001, upper=0.1),
    stiffness=ng.p.Scalar(lower=3000, upper=10000),
    damping=ng.p.Scalar(lower=80, upper=500),
    rolling_friction=ng.p.Scalar(lower=0.0001, upper=0.6),
    torsional_friction=ng.p.Scalar(lower=0.0001, upper=0.2),
    )

    # create a new instance
    child = instrum.spawn_child()
    # update its value
    child.value = ((), {'d0': 0.9433598140425187, 'd_width': 0.1395079405492468, 'width': 0.05208016638200587, 'stiffness': 4724.630709636312, 'damping': 91.11742334723378, 'rolling_friction': 0.5985903152412875, 'torsional_friction': 0.02774301352299431})
    
    optimizer = ng.optimizers.NGOpt(parametrization=child, budget=1000, num_workers=28)

    # optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint (parametrization=instrum, budget=100, num_workers=20)

    logger = ng.callbacks.ParametersLogger(logger_path)
    optimizer.register_callback("tell",  logger)
    
    start_time = time.time()
    with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(param_reconize.loss_function_for_all_z_and_all_parameter_in_all_time, executor=executor, batch_mode=True, verbosity=1)

    print('loss:',recommendation.loss)
    print('value:',recommendation.value[1])
    print('total time: ', time.time() - start_time)

if __name__ == '__main__':
    # logger_path='./tests/0915/recognize_parameter_first_1_seconds_and_all_parameter_5'
    # recognize_parameter_first_0_35_seconds_and_only_height()
    # recognize_parameter_first_1_seconds_and_all_parameter(logger_path)
    # plot_recognize_result(logger_path)

    # 4时间递减权重，提高水平占比，降低角速度占比
    # result = {'d0': 0.9085700342518757, 'd_width': 0.0745428576730932, 'width': 0.03337242697776478, 'stiffness': 5145.305317271622, 'damping': 69.83962994484878, 'rolling_friction': 0.5709121535080875, 'torsional_friction': 0.02677454476816001}
    # 0.25时间递减权重
    # result = {'d0': 0.9433598140425187, 'd_width': 0.1395079405492468, 'width': 0.05208016638200587, 'stiffness': 4724.630709636312, 'damping': 91.11742334723378, 'rolling_friction': 0.5985903152412875, 'torsional_friction': 0.02774301352299431}
    # 损失函数大修
    # result =  {'d0': 0.3918458065047926, 'd_width': 0.06777539040324305, 'width': 0.004479947147087014, 'stiffness': 2045.9038525093313, 'damping': 100.19218200606545, 'rolling_friction': 0.5706360673627742, 'torsional_friction': 0.02266737747778054}
    result = {'d0': 0.9698819142575082, 'd_width': 0.10222059345169457, 'width': 0.055102813760108824, 'stiffness': 4360.38894615363, 'damping': 138.02062726106232, 'rolling_friction': 0.5391332783891921, 'torsional_friction': 0.019243073269965646}
    xlsx_path = './tests/data_to_fit_large_stiffness.xlsx'
    d0 = result['d0']
    d_width = result['d_width']
    width = result['width']
    stiffness = result['stiffness']
    damping = result['damping']
    rolling_friction = result['rolling_friction']
    torsional_friction = result['torsional_friction']
    param_reconize = ParameterReconize(adams_data_path=xlsx_path, z_list=[165, 162, 160, 150], first_few_seconds=1, stop_time=1)
    
    xml_file_path = './data/xml/0916/TT12_0911_0.xml'
    param_reconize.simulate_with_specific_parameter(index=0, stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, rolling_friction=rolling_friction, torsional_friction=torsional_friction, stop_time=10,is_render=True, xml_file_path=xml_file_path)
    # param_reconize.plot_adams_df(choice='adams_df_list_first_few_seconds')
    # param_reconize.plot_simulated_data(stiffness=stiffness, damping=damping, d0=d0, d_width=d_width, width=width, rolling_friction=rolling_friction, torsional_friction=torsional_friction, stop_time=1.5)
    # print(param_reconize.loss_function_for_all_z_and_all_parameter_in_all_time(d0=d0, d_width=d_width, width=width, stiffness=stiffness, damping=damping, rolling_friction=rolling_friction, torsional_friction=torsional_friction))
    # print(param_reconize.loss_function_for_all_z_and_all_parameter_in_all_time())
    # analysis_performance('loss_function_for_all_z_and_height_parameter_in_first_few_seconds')