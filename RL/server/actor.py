import os
import pickle
import subprocess
import time
from argparse import ArgumentParser
from collections import deque
from itertools import count
from multiprocessing import Process, Array
from pathlib import Path
from typing import Tuple, Any, List, Dict

import numpy as np
import zmq
from pyarrow import serialize

from common import load_yaml_config, create_experiment_dir
from utils import logger
from utils.cmdline import parse_cmdline_kwargs

parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='The game environment')
parser.add_argument('--num_steps', type=int, default=10000000, help='The number of total training steps')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--num_replicas', type=int, default=1, help='The number of actors')
parser.add_argument('--model', type=str, default='accnn', help='Training model')
parser.add_argument('--max_steps_per_update', type=int, default=128,
                    help='The maximum number of steps between each update')
parser.add_argument('--exp_path', type=str, default=None,
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=10, help='Number of recent checkpoint files to be saved')
parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum length of trajectory')
parser.add_argument('--config', type=str, default=None, help='The YAML configuration file')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU to sample every action')
parser.add_argument('--num_envs', type=int, default=10, help='The number of environment copies')


def run_one_agent(index, args, unknown_args, actor_status):
    from tensorflow.keras.backend import set_session
    import tensorflow.compat.v1 as tf

    # Set 'allow_growth'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Connect to learner
    context = zmq.Context()
    context.linger = 0  # For removing linger behavior
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.data_port}')

    # Initialize environment and agent instance
    # env, agent = init_components(args, unknown_args)

    # Configure logging only in one process
    if index == 0:
        logger.configure(str(args.log_path))
        # save_yaml_config(args.exp_path / 'config.yaml', args, 'actor', agent)
    else:
        logger.configure(str(args.log_path), format_strs=[])

    # Initialize values
    model_id = -1
    episode_infos = deque(maxlen=100)
    num_episode = 0
    # state = env.reset()

    while True:
        # Do some updates
        # agent.update_sampling(update, nupdates)

        mb_states, mb_actions, mb_rewards, mb_dones, mb_extras = [], [], [], [], []
            # Sample action
            # action, extra_data = agent.sample(state)
            # state, reward, done, info = env.step(action)


        '''读文件在这里'''
        while not os.listdir("/root/data"):
            # print(os.listdir("/root/data"))
            time.sleep(1)
        temp = os.listdir("/root/data")
        filename = "/root/data/" + temp[0]
        time.sleep(5)
        num_episode += 1
        df = open(filename,'rb')
        data = pickle.load(df)
        df.close()
        os.system("rm " + filename)
        for x in data:
            mb_states.append([x[0]])
            mb_actions.append([x[1]])
            mb_rewards.append([x[2]])
            temp = {}
            temp['neglogp'] = [x[3]]
            temp['value'] = [x[4]]
            mb_extras.append(temp)
            mb_dones.append([x[5]])
            next_state = x[0]
            # for info_i in info:
            #     maybeepinfo = info_i.get('episode')
            #     if maybeepinfo:
            #         episode_infos.append(maybeepinfo)
            #         num_episode += 1

        mb_states = np.asarray(mb_states, dtype=next_state.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # print(mb_states.shape, mb_actions.shape, mb_rewards.shape, mb_dones.shape,mb_extras.shape)
        data = prepare_training_data([mb_states, mb_actions, mb_rewards, mb_dones, next_state, mb_extras])
        # data = agent.prepare_training_data([mb_states, mb_actions, mb_rewards, mb_dones, state, mb_extras])
        # post_processed_data = agent.post_process_training_data(data)
        socket.send(serialize(data).to_buffer())
        socket.recv()

        # Log information
        logger.record_tabular("episodes", num_episode)
        logger.dump_tabular()

        # Update weights
        latest_file, new_weights, model_id = find_new_weights(model_id, args.ckpt_path)
        if new_weights is not None:
            # agent.set_weights(new_weights)
            '''在这里改模型存储路径'''
            os.system(f'cp {args.ckpt_path}/{latest_file} /root/model/model.pt')

    actor_status[index] = 1


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def prepare_training_data(trajectory: List[Tuple[Any, Any, Any, Any, Any, dict]]) -> Dict[str, np.ndarray]:
    mb_states, mb_actions, mb_rewards, mb_dones, next_state, mb_extras = trajectory
    mb_values = np.asarray([extra_data['value'] for extra_data in mb_extras])
    mb_neglogp = np.asarray([extra_data['neglogp'] for extra_data in mb_extras])
    print(mb_values.shape)

    last_values = [[0]]
    mb_values = np.concatenate([mb_values, last_values])
    # mb_values.resize((mb_values.shape[0], 1))
    # print(mb_values.shape)

    mb_deltas = mb_rewards + 0.99 * mb_values[1:] * (1.0 - mb_dones) - mb_values[:-1]

    nsteps = len(mb_states)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(nsteps)):
        nextnonterminal = 1.0 - mb_dones[t]
        mb_advs[t] = lastgaelam = mb_deltas[t] + 0.99 * 0.95 * nextnonterminal * lastgaelam

    mb_returns = mb_advs + mb_values[:-1]
    print(mb_states.shape, mb_returns.shape, mb_actions.shape, mb_values[:-1].shape, mb_neglogp.shape)
    data = [sf01(arr) for arr in [mb_states, mb_returns, mb_actions, mb_values[:-1], mb_neglogp]]
    name = ['state', 'return', 'action', 'value', 'neglogp']
    return dict(zip(name, data))


def run_weights_subscriber(args, actor_status):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):  # Starts from 1
        while True:
            try:
                weights = socket.recv(flags=zmq.NOBLOCK)

                # Weights received
                with open(args.ckpt_path / f'{model_id}.{args.alg}.{args.env}.ckpt', 'wb') as f:
                    f.write(weights)

                if model_id > args.num_saved_ckpt:
                    os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.{args.alg}.{args.env}.ckpt')
                break
            except zmq.Again:
                pass

            if all(actor_status):
                # All actors finished works
                return

            # For not cpu-intensive
            time.sleep(1)


def find_new_weights(current_model_id: int, ckpt_path: Path) -> Tuple[Any, int]:
    try:
        ckpt_files = sorted(os.listdir(ckpt_path), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    new_weights = pickle.load(f)
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return latest_file, new_weights, new_model_id
    else:
        return latest_file, None, current_model_id


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'actor')

    # Create experiment directory
    create_experiment_dir(args, 'ACTOR-')

    args.ckpt_path = args.exp_path / 'ckpt'
    args.log_path = args.exp_path / 'log'
    args.ckpt_path.mkdir()
    args.log_path.mkdir()

    # Record commit hash
    with open(args.exp_path / 'hash', 'w') as f:
        f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    # Disable GPU
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Running status of actors
    actor_status = Array('i', [0] * args.num_replicas)

    # Run weights subscriber
    subscriber = Process(target=run_weights_subscriber, args=(args, actor_status))
    subscriber.start()

    def exit_wrapper(index, *x, **kw):
        """Exit all agents on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_agent(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(agents):
                    if _i != index:
                        _p.terminate()
                    actor_status[_i] = 1

    agents = []
    for i in range(args.num_replicas):
        p = Process(target=exit_wrapper, args=(i, args, unknown_args, actor_status))
        p.start()
        os.system(f'taskset -p -c {i % os.cpu_count()} {p.pid}')  # For CPU affinity

        agents.append(p)

    for agent in agents:
        agent.join()

    subscriber.join()


if __name__ == '__main__':
    main()
