import argparse
from gymnasium.envs.registration import register
import gymnasium as gym



def parse_arguments(*args_list):
    parser = argparse.ArgumentParser(description='Process some integers.')

    for arg in args_list:
        parser.add_argument(arg['name'], type=arg['type'], default=arg['default'], help=arg['help'])

    args = parser.parse_args()
    return args

def make_mujoco_env(model_path, 
                    env_name='TT12_0608-v1', 
                    entry_point="my_envs.tt12_v1:TT12Env", 
                    **kwargs):
    """生成基于mujoco物理引擎的Gym环境

    Args:
        model_path (_type_): 待输入的xml文件路径+文件名
        env_name (str, optional): 生成的环境名称. Defaults to 'TT12_0608-v1'.
        render_mode:default:None,可以填写"human", 则环境运行时会渲染视频，默认关闭
        record_video:default:False,如果true,则记录视频
        video_folder:default:'./data/videos',
        name_prefix:default:'rl-video-{env_name}',

    Returns:
        Gymnasium.Env: Gym环境
    """
    print('make_mujoco_env',env_name)
    register(
        id=env_name,
        entry_point=entry_point,
    )
    render_mode = kwargs.get('render_mode', None)
    record_video = kwargs.get('record_video', False)
    video_folder = kwargs.get('video_folder', './data/videos')
    name_prefix = kwargs.get('name_prefix', f"rl-video-{env_name}")
    # print('hhh i am used make mujoco env')
    if record_video:
        render_mode = 'rgb_array'
        return gym.wrappers.RecordVideo(gym.make(env_name, render_mode=render_mode, 
                                                 model_path=model_path), 
                                                 video_folder=video_folder, 
                                                 name_prefix=name_prefix)
    else:
        return gym.make(env_name, render_mode=render_mode, model_path=model_path)