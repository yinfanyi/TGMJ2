from concurrent import futures
import nevergrad as ng
import torch
import matplotlib.pyplot as plt
import numpy as np
import mujoco as mj
import time
import pandas as pd


xlsx_path = './tests/drop_test_0826.xlsx'
df = pd.read_excel(xlsx_path, sheet_name='Sheet1')
# 将df的名字为h的列全部乘以0.001
# 截取df的t>0.4的行
adams_df = df[(df['t'] > 0.4) & (df['t'] < 0.6)].copy()
adams_df['h'] = adams_df['h'] * 0.001
adams_np = adams_df.to_numpy()
time_a = adams_np[:, 0]  
values_a = adams_np[:, 1]  

def loss_function(d0=0.9, d_width=0.95, width=0.001, stiffness=3300, damping=40):
    friction1 = 0.6
    friction2 = 0.005
    friction3 = 0.0001

    a0 = 10
    Hz = 1000
    BLOCK_HALF_WIDTH = 0.05
    mass = 1
    time = 0.6

    pos = np.array([0, 0, 0.05])
    key1pos = np.array([0, 0, 1])
    quat = np.array([0, 0, 0, 1])
    vel = np.array([0, 0, 0])
    angvel = np.array([0, 0, 0])

    time_list = []
    xpos_list = []
    
    xml_string=f"""
        <mujoco>
        <option integrator="implicit" 
            o_solref="{-stiffness} {-damping}" 
            o_solimp="{d0} {d_width} {width} 0.5 1" 
            o_friction = '{friction1} {friction2} {friction3}'
            timestep="{1/Hz}" 
            gravity="0 0 -{a0}" >
        <flag override="disable" />
        </option>
        <worldbody>
            <light name="light_-2"  pos="-2 -1 3" dir="0.666667 0.333333 -0.666667"/>
            <light name="light_2"  pos="2 -1 3" dir="-0.666667 0.333333 -0.666667"/>
            <geom name='floor' 
                type='plane' 
                size='0 0 0.01' 
                priority='1' 
                solref="{-stiffness} {-damping}"
                solimp="{d0} {d_width} {width} 0.5 2" 
                friction='{friction1} {friction2} {friction3}'/>
            <body name='box' pos="{pos[0]} {pos[1]} {pos[2]}" quat='{quat[0]} {quat[1]} {quat[2]} {quat[3]}'>
                <freejoint/>
                <inertial pos='0 0 0' mass='0.37' diaginertia='0.00081 0.00081 0.00081'/>
                <geom name='box' 
                    type='box' 
                    size="{BLOCK_HALF_WIDTH} {BLOCK_HALF_WIDTH} {BLOCK_HALF_WIDTH}" 
                    rgba="0.8 0.1 0.1 1"
                    mass="{mass}"/>
            </body>
        </worldbody>
        <keyframe>
            <key name='initial_state' qvel='{vel[0]} {vel[1]} {vel[2]} {angvel[0]} {angvel[1]} {angvel[2]}'/>
            <key name='initial_state1' qpos='{key1pos[0]} {key1pos[1]} {key1pos[2]} 0 0 0 1' qvel='{vel[0]} {vel[1]} {vel[2]} {angvel[0]} {angvel[1]} {angvel[2]}'/>
        </keyframe>
        </mujoco>
        """
    m = mj.MjModel.from_xml_string(xml_string)
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 1)
    mj.mj_forward(m, d)
    while True:
        if d.time >= time:
            break
        mj.mj_step(m, d)
        time_list.append(d.time)
        xpos_list.append(d.body('box').xpos[2].copy())
    mj_np = np.column_stack((np.array(time_list), np.array(xpos_list)))  
    mj_np = mj_np[(mj_np[:, 0] > 0.4) & (mj_np[:, 0] < 0.6)] 
    time_b = mj_np[:, 0]  
    values_b = mj_np[:, 1]  

    # 使用 np.interp 进行插值，将 b 的值插值到 a 的时间点  
    interpolated_values_b = np.interp(time_a, time_b, values_b)  
    absolute_difference = np.abs(values_a - interpolated_values_b)  
    sum_difference = np.sum(absolute_difference, axis=0) 
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
    # print('list_of_dict_of_data: ', list_of_dict_of_data)

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