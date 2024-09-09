from concurrent import futures
import nevergrad as ng
import torch
import matplotlib.pyplot as plt
import numpy as np
import mujoco as mj
import time

BLOCK_HALF_WIDTH = 0.0524

class SingleTossExperiment():
    def __init__(self, stiffness, damping, friction, real_data:torch.Tensor, xml_save_name) -> None:
        self.model, self.data = self.init(stiffness, damping, friction, real_data, xml_save_name)
        mj.mj_resetDataKeyframe(self.model, self.data, 0)

    def init(self, stiffness, damping, friction, real_data:torch.Tensor, xml_save_name):
        pos = real_data[0,0,:3].numpy() * BLOCK_HALF_WIDTH
        quat = real_data[0,0,3:7].numpy()
        vel = real_data[0,0,7:10].numpy() * BLOCK_HALF_WIDTH
        angvel = real_data[0,0,10:13].numpy()
        Hz=148
        d0 = 0.9
        d_width = 0.95
        width = 0.001
        a0 = 9.8

        xml_string=f"""
            <mujoco>
            <option integrator="implicit" 
                o_solref="{-stiffness} {-damping}" 
                o_solimp="{d0} {d_width} {width} 0.5 1" 
                o_friction = '{friction} {friction} {friction}'
                timestep="{0.1/Hz}" 
                gravity="0 0 -{a0}" >
            <flag override="enable" />
            </option>
            <worldbody>
                <geom name='floor' type='plane' size='0 0 0.01'/>
                <body name='box' pos="{pos[0]} {pos[1]} {pos[2]}" quat='{quat[0]} {quat[1]} {quat[2]} {quat[3]}'>
                    <freejoint/>
                    <inertial pos='0 0 0' mass='0.37' diaginertia='0.00081 0.00081 0.00081'/>
                    <geom name='box' type='box' size="{BLOCK_HALF_WIDTH} {BLOCK_HALF_WIDTH} {BLOCK_HALF_WIDTH}"/>
                </body>
            </worldbody>
            <keyframe>
                <key name='initial_state' qvel='{vel[0]} {vel[1]} {vel[2]} {angvel[0]} {angvel[1]} {angvel[2]}'/>
            </keyframe>
            </mujoco>
            """
        
        with open(xml_save_name, "w") as f:
            f.write(xml_string)
        
        model = mj.MjModel.from_xml_string(xml_string)
        data = mj.MjData(model) 
        return model, data

    def step(self, frame_skip=10):
        for _ in range(frame_skip):
            mj.mj_step(self.model, self.data)

class Agent():
    def __init__(self, real_data:torch.Tensor) -> None:
        self.total_time = real_data.shape[1]/148

        self.time_list = []
        self.xpos_list = []
        self.xquat_list = []
        
        self.real_poss = real_data[0,:,:3].numpy() * BLOCK_HALF_WIDTH
        self.real_quats = real_data[0,:,3:7].numpy()
        self.real_vels = real_data[0,:,7:10].numpy() * BLOCK_HALF_WIDTH
        self.real_angvels = real_data[0,:,10:13].numpy()

        self.is_post_process = False
    
    def bind_data(self, data):
        self.time_list.append(data.time)
        self.xpos_list.append(data.body('box').xpos.copy())
        # print(data.body('box').xpos)
        # print(self.xpos_list)
        self.xquat_list.append(data.body('box').xquat.copy())
            
    def post_process(self):
        if not self.is_post_process:
            self.time_array = np.array(self.time_list)
            self.xpos_array = np.array(self.xpos_list)
            self.xquat_array = np.array(self.xquat_list)
        self.is_post_process = True
    
    def plot_data(self):
        if not self.is_post_process:
            self.post_process()
        _, ax = plt.subplots(1, 1)
        ax.plot(self.time_array, self.xpos_array[:, 2])
        incremental_array = np.arange(0, self.real_poss.shape[0]) * (1 / 148)
        ax.plot(incremental_array, self.real_poss[:, 2])
        ax.set_xlabel('time')
        ax.set_ylabel('height')
        ax.grid(True)
        ax.legend(['sim', 'real'])
    
    def get_error(self):
        if not self.is_post_process:
            self.post_process()
        error = 0
        for i in range(self.real_poss.shape[0]):
            e_pos = np.linalg.norm(self.real_poss[i] - self.xpos_array[i])/BLOCK_HALF_WIDTH
            # 对应起来，因为mujoco的频率是1480，数据量比实测数据大10倍
            e_angle = self.quaternion_angle(self.real_quats[i], self.xquat_array[i])

            error += e_pos + e_angle
            # print(f'step{i}, error:{e_pos}, {e_angle}')
        error = error/self.real_poss.shape[0]
        return error
    
    def quaternion_angle(self, q1, q2):
        dot_product = np.dot(q1, q2)
        abs_q1 = np.linalg.norm(q1)
        abs_q2 = np.linalg.norm(q2)
        tmp = abs(dot_product) / (abs_q1 * abs_q2)
        tmp = min(tmp, 1)
        angle = 2 * np.arccos(tmp)
        return angle

class TossExperiment():
    def __init__(self, 
                 stiffness:float, 
                 damping:float, 
                 friction:float,
                 ) -> None:
        self.stiffness = stiffness
        self.damping = damping
        self.friction = friction
        
    def get_single_experiment_error(self, i):
        loaded_tensor = torch.load(f'./data/tosses_processed/{i}.pt')
        agent = Agent(real_data=loaded_tensor)
        single_toss_experiment = SingleTossExperiment(self.stiffness, self.damping, self.friction, loaded_tensor, xml_save_name = f'./data/xml/0622/{i}.xml')
        for _ in range(loaded_tensor.shape[1]):
            single_toss_experiment.step()
            agent.bind_data(single_toss_experiment.data)
        # agent.plot_data()
        # print('error: ', agent.get_error())
        return agent.get_error()
    
    def get_loss_function(self):
        # n = 570
        n = 570
        loss_function = 0
        for i in range(n):
            loss_function += self.get_single_experiment_error(i)
        return loss_function/n
      
def loss_function(stiffness=3300, damping=45, friction=0.22):
    toss_experiment = TossExperiment(stiffness, damping, friction)
    return toss_experiment.get_loss_function()

# optimization on x as an array of shape (2,)

if __name__ == '__main__':
    start_time = time.time()
    print(loss_function())
    

    loss_history = []
    param_history = []

    instrum = ng.p.Instrumentation(
        stiffness=ng.p.Scalar(lower=1000, upper=10000),
        damping=ng.p.Scalar(lower=0.001, upper=1000),
        friction=ng.p.Scalar(lower=0.001, upper=1),
    )

    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100, num_workers=30)
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