from concurrent import futures
import nevergrad as ng
import matplotlib.pyplot as plt
import numpy as np
import mujoco as mj
import time
import pandas as pd
import xml.etree.ElementTree as ET


xml_path = './data/xml/TT12_0828.xml'
xlsx_path = './tests/tensegrity_jump0828.xlsx'

df = pd.read_excel(xlsx_path, sheet_name='Sheet1')
adams_df = df.copy()
adams_df['h'] = adams_df['h'] * 0.001
adams_np = adams_df.to_numpy()

def update_floor_solref_solimp(xml_file, solref, solimp):
    tree = ET.parse(xml_file)
    worldbody = tree.find('worldbody')
    geoms = worldbody.findall('geom')
    for geom in geoms:
        if geom.get('name') == 'floor':
            geom.set('solref', solref)
            geom.set('solimp', solimp)
    return ET.tostring(tree.getroot(), encoding='unicode')

def controller(model, data):
    torque_x_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "tt_model_0/torque_x_ctrl")

    if abs(data.sensor('tt_model_0/angvel_inside_ball').data[0] - data.sensor('tt_model_0/angvel_outside_ball').data[0]) < 0.1:
        data.ctrl[torque_x_actuator_id] = 0

def simulate_step(model, data, Hz):
    current_simstart = data.time   # 当前一帧的开始时间
    while (data.time - current_simstart < 1.0/Hz):
        mj.mj_step(model, data)

def loss_function(d0=0.27, d_width=0.07, width=0.06, stiffness=5773, damping=68):
    xml_string =update_floor_solref_solimp(xml_path, solref=f'{-stiffness} {-damping}', solimp=f'{d0} {d_width} {width} 0.5 2')

    m = mj.MjModel.from_xml_string(xml_string)
    d = mj.MjData(m)

    mj.mj_resetDataKeyframe(m, d, 0)
    mj.mj_forward(m, d)
    mj.set_mjcb_control(controller)
    Hz = 1000
    time_list = []
    outside_z_pos_list = []
    angvel_outside_ball_list = []

    while d.time < 1.5:
        simulate_step(m, d, Hz)
        time_list.append(d.time)
        outside_z_pos_list.append(d.sensor('tt_model_0/pos_outside_ball').data[2])
        angvel_outside_ball_list.append(d.sensor('tt_model_0/angvel_outside_ball').data[0])
    
    mj_np = np.column_stack((np.array(time_list), np.array(angvel_outside_ball_list), np.array(outside_z_pos_list)))
    interpolated_values_angvel = np.interp(adams_np[:,0], mj_np[:,0], mj_np[:,1])
    interpolated_values_height = np.interp(adams_np[:,0], mj_np[:,0], mj_np[:,2])
    absolute_difference = np.abs(adams_np[:, 1] - interpolated_values_angvel) + np.abs(adams_np[:, 2] - interpolated_values_height)
    sum_difference = np.sum(absolute_difference, axis=0)
    del m
    del d
    return sum_difference

if __name__ == '__main__':
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