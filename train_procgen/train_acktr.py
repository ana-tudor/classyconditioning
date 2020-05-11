import tensorflow as tf
import gym
from baselines.acktr import acktr
from baselines.common.cmd_util import make_vec_env
from baselines.common.models import nature_cnn
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from stable_baselines.common.policies import CnnLnLstmPolicy, MlpLnLstmPolicy, MlpPolicy
from baselines.common.cmd_util import make atari_env
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
    num_envs = 1

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--start_level', type=int, default=0)

    parser.add_argument('--timesteps_total', type=int, default=50_000_000)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=LOG_DIR+"default")

    args = parser.parse_args()

    test_worker_interval = 0#args.test_worker_interval
    timesteps_per_proc = args.timesteps_total
    save_interval = args.save_interval
    load_path = args.load_path
    run_dir = args.run_dir
    seed = args.seed
    log_interval = args.log_interval

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
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
    config = tf.ConfigProto() #device_count = {'GPU':0})
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    # conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    network = nature_cnn#'cnn'

    print("num_levels", num_levels)
    logger.info("training")

    #venv = make_vec_env("BreakoutNoFrameskip-v4", 'atari', 2, seed=10)

    acktr.learn(
        network=network,
        env=venv,
        seed=seed,
        total_timesteps = timesteps_per_proc,
        log_interval=log_interval,
        save_interval=save_interval,
        load_path=load_path
    )

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main()
