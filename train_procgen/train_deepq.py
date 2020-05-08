import tensorflow as tf
from baselines.deepq import deepq
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
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
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_total', type=int, default=50_000_000)
    parser.add_argument('--save_interval', type=int, default=0)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=LOG_DIR+"default")

    args = parser.parse_args()

    test_worker_interval = args.test_worker_interval
    timesteps_per_proc = args.timesteps_total
    save_interval = args.save_interval
    load_path = args.load_path
    run_dir = args.run_dir

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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    if is_test_worker:
        train_freq = timesteps_per_proc + 1
    else:
        train_freq = 1

    if save_interval == 0:
        checkpoint_freq = None
    else: 
        checkpoint_freq = 

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    print("num_levels", num_levels)
    logger.info("training")
    deepq.learn(
        env = venv,
        network = conv_fn,
        seed=None,
        lr=5e-4,
        total_timesteps=timesteps_per_proc,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=train_freq,
        batch_size=32,
        print_freq=1000,
        checkpoint_freq=checkpoint_freq,
        checkpoint_path=None,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        callback=None,
        load_path=load_path
    )

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main()
