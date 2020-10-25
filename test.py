import timeit
import pickle
import numpy as np
import gym
import gym_gvgai
# import gvgai
import ZeldaEnv
from a2c_zelda.a2c import Model
from a2c_zelda.cmd_util import make_vec_env
from baselines.common.policies import build_policy
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

def main():
    numOfTests = 40
    env_args = {'episode_life': False, 'clip_rewards': False, 'crop': True, 'rotate': True}
    env = VecFrameStack(make_vec_env("gvgai-zelda-lvl0-v0", numOfTests, 43, wrapper_kwargs=env_args), 4)
    policy  = build_policy(env, "cnn")
    model = Model(policy=policy, env=env, nsteps=5)
    model.load('logs/test_4*5_r1_right/checkpoints/260000')
    nh, nw, nc = env.observation_space.shape
    result = dict()
    for j in range(201, 601):
        # obs = np.zeros((numOfTests, nh, nw, nc), dtype=np.uint8)
        done = np.array([False] * numOfTests)
        env.venv.set_level("GVGAI_GYM/gym_gvgai/envs/games/zelda_v0/zelda_lvl{}.txt".format(j))
        obs = env.reset()
        infos = [False] * numOfTests
        # dones = [False] * numOfTests

        while not all(done):
            actions, values, state, _ = model.step(obs)
            obs, rewards, dones, info = env.step(actions)
            done[np.where(dones!=False)] = True
            for i in np.where(dones!=False)[0].tolist():
                if not infos[i]:
                    # print(info)
                    del info[i]["grid"]
                    del info[i]["ascii"]
                    infos[i] = info[i]
            # print(np.where(dones!=False)[0])
            # print(done)
            # print(infos)

        # print(dones)
        win = [1 if (i['winner'] == 'PLAYER_WINS') else 0 for i in infos]
        # score = [i['episode']['r'] for i in infos]
        # steps = [i['episode']['l'] for i in infos]
        # time = [i['episode']['t'] for i in infos]
        print("level {}".format(j), win)
        result[j] = infos
  
    env.close()


    with open("result_4*5_r1_right_200~600", "wb") as f:
        pickle.dump(result, f)
        # pass

if __name__ == '__main__':
    # import time
    # now = time.time()
    main()
    # print("running time {}".format(time.time()-now))
