# 滑块摩擦测试

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np

stiffness = 3300
damping = 40
d0 = 0.01
d_width = 0.01
width = 0.001

friction1 = 0.6
friction2 = 0.005
friction3 = 0.0001

a0 = 10
Hz = 1000
BLOCK_HALF_WIDTH = 0.05
mass = 5
time = 0.6

pos = np.array([0, 0, 0.05])
key1pos = np.array([0, 0, 1])
quat = np.array([0, 0, 0, 1])
vel = np.array([0, 0, 0])
angvel = np.array([0, 0, 0])

time_list = []
xpos_list = []
xvel_list = []

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

def viewer_scene():
    def load_callback(m=None, d=None):
        m = mj.MjModel.from_xml_string(xml_string)
        d = mj.MjData(m)
        return m, d
    mj_viewer.launch(loader=load_callback)

def sim(keyframe_id=0):
    m = mj.MjModel.from_xml_string(xml_string)
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, keyframe_id)
    mj.mj_forward(m, d)
    while True:
        if d.time >= time:
            break
        mj.mj_step(m, d)
        time_list.append(d.time)
        xpos_list.append(d.body('box').xpos[2].copy())
        xvel_list.append(d.body('box').cvel[5].copy())

def plot_result(): 
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.title('Position')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.grid()
    plt.plot(time_list, xpos_list)
    plt.subplot(2, 1, 2)
    plt.title('Velocity')
    plt.plot(time_list, xvel_list)
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__': 
    viewer_scene()
    sim(keyframe_id=1)
    plot_result()
    # print('time_list:')
    # print(time_list)
    # print('xpos_list:')
    # print(xpos_list)
    # print('xvel_list:')
    # print(xvel_list)

    