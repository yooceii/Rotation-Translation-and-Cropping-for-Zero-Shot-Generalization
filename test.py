import gym
import gym_gvgai
import ZeldaEnv
from a2c_zelda.a2c import Model
from a2c_zelda.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

env_args = {'episode_life': False, 'clip_rewards': False, 'crop': True}
env = VecFrameStack(make_vec_env("gvgai-zelda-lvl0-v0", 3, 43, level_path="/home/chang/gail/levels/", wrapper_kwargs=env_args), 4)
model = Model(policy='cnn', env=env, nsteps=5)
model.load('logs/test/checkpoints/965000')
env.close()
 




