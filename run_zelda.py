#!/usr/bin/env python3

from baselines import logger
from a2c_zelda.cmd_util import make_vec_env, common_arg_parser, atari_arg_parser, parse_unknown_args
from a2c_zelda.a2c import learn
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
# from baselines.a2c.a2c import learn
# from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, save_path, load_path, wrapper_kwargs):
    env_args = {'episode_life': False, 'clip_rewards': False}
    env_args.update(wrapper_kwargs)
    env = VecFrameStack(make_vec_env(env_id, num_env, seed, wrapper_kwargs=env_args), 4)
    # env = make_vec_env(env_id, num_env, seed, wrapper_kwargs=env_args)
    model = learn(policy, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, load_path=load_path)
    model.save(save_path)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--crop', help='trasnlate observation', action='store_true')
    parser.add_argument('--rotate', help='rotate observation', action="store_true")
    parser.add_argument('--full', help='full observation', action="store_true")
    parser.add_argument('--repava', help='replace avatar', action="store_true")
    # args = parser.parse_args()
    args, unknow_args = parser.parse_known_args()
    extra_args = parse_unknown_args(unknow_args)
    logger.configure(args.log_path)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16, save_path=args.save_path, load_path=extra_args.get('load_path', None), wrapper_kwargs={'crop':args.crop, 'rotate':args.rotate, 'full':args.full, 'repava':args.repava})

if __name__ == '__main__':
    main()
