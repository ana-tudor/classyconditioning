import tensorflow as tf
from baselines.ddpg import ddpg
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common import set_global_seeds, tf_util
from baselines.her.experiment import config
from baselines.her import her
from baselines.her.rollout import RolloutWorker
import os
import json
import numpy as np

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

# LOG_DIR = '/tmp/procgen'
LOG_DIR = './train_procgen/models/'

def main():
    num_envs = 64
    learning_rate = 5e-4

    nsteps = 256
    nminibatches = 8


    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    #These are arguments for venv
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--start_level', type=int, default=0)

    #These are model arguments
    # parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_total', type=int, default=10_000_000)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--n_test_rollouts', type=int, default=10)

    parser.add_argument('--policy_save_interval', type=int, default=0, help='the interval with which policy pickles are saved. \
                        If set to 0, only the best and latest policy will be pickled.')
    # parser.add_argument('--save_interval', type=int, default=0)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=LOG_DIR+"default")
    parser.add_argument('--demo_file', type=str, default=None)

    args = parser.parse_args()

    test_worker_interval = args.test_worker_interval
    timesteps_per_proc = args.timesteps_total
    policy_save_interval = args.policy_save_interval
    load_path = args.load_path
    save_path = args.save_path
    demo_file = args.demo_file
    run_dir = args.run_dir

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_cpu = comm.Get_size()
    print(rank)
    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)
    print("is_test_worker", is_test_worker)
    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=run_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config_)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    print("num_levels", num_levels)

    if args.test:
        logger.info("testing")
    else:
        logger.info("training")

    '''
    Setup for HER parameters
    '''

    rank_seed = 1000000*rank
    set_global_seeds(rank_seed)

    # print([attr for attr in config])
    params = config.DEFAULT_PARAMS
    env = venv
    env_name = args.env_name
    params['env_name'] = env_name
    params['replay_strategy'] = 'future'
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    # params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    if demo_file is not None:
        params['bc_loss'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=True)
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = env

    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    if args.test:
        n_cycles = 0
        n_epochs = total_timesteps//n_test_rollouts
    else:
        n_test_rollouts = 0
        n_cycles = params['n_cycles']
        n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size




    her.train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=n_test_rollouts,
        n_cycles=n_cycles, n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file)

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main()
