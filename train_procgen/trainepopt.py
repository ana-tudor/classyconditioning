import tensorflow.compat.v1 as tf
# from baselines.ppo2 import ppo2
import epopt
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
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import argparse

# LOG_DIR = '/tmp/procgen'
LOG_DIR = './train_procgen/models/'
tf.disable_v2_behavior()

def main():
    num_envs = 2
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 50
    nminibatches = 10
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
    parser.add_argument('--test_mode', type=bool, default=False)

    args = parser.parse_args()

    test_worker_interval = args.test_worker_interval
    timesteps_per_proc = args.timesteps_total
    save_interval = args.save_interval
    load_path = args.load_path
    run_dir = args.run_dir
    test_mode = args.test_mode

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # print(rank)
    # is_test_worker = False

    # if test_worker_interval > 0:
    #     is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)
    # print("is_test_worker", is_test_worker)
    mpi_rank_weight = 0 #if is_test_worker else 1
    num_levels = args.num_levels

    # log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] #if log_comm.Get_rank() == 0 else []
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

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    print("num_levels", num_levels)
    logger.info("training")
    epopt.learn(
        env=venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        # test_mode=test_mode,
        save_interval=save_interval,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=None,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        load_path=load_path,
        paths=10, epsilon=1.0
    )

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()