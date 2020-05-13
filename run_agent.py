import tensorflow.compat.v1 as tf
# from baselines.ppo2 import ppo2
from fruitbot_ppo import ppo_agent
from fruitbot_ppo.reg_impala_cnn import build_reg_impala_cnn

# Default CNN
#from baselines.common.models import build_impala_cnn

build_impala_cnn = build_reg_impala_cnn


from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
# from mpi4py import MPI
import argparse

# LOG_DIR = '/tmp/procgen'
LOG_DIR = 'models/'
tf.disable_v2_behavior()


'''
Global variables defaults, values can be changed via parser
All variables of interest which are desired to be tuned must be listed here
'''

#Hyperparameters
num_envs = 32
learning_rate = 1e-3
ent_coef = .01
gamma = .999
lam = .95
nsteps = 256
nminibatches = 8
ppo_epochs = 3
clip_range = .2
use_vf_clipping = True

#Important variables of interest
rew_scale = 1
rew_baseline = False
conv_fn = lambda x: build_impala_cnn(x, depths=[16,64,64], emb_size=256)
conv_fn_vals = [lambda x: build_impala_cnn(x, depths=[16, 32, 64], emb_size=256),
                lambda x: build_impala_cnn(x, depths=[32, 32], emb_size=256),
                lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)]

seeds = 1543


def main():
    #Create argument parser
    parser = argparse.ArgumentParser(
        description='Process fruitbot_ppo agent training arguments.')

    parser.add_argument('--env_name', type=str, default='fruitbot',
        help='Provide an environment name available in procgen')
    parser.add_argument('--distribution_mode', type=str, default='easy',
                choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50,
        help='Number of levels to run in the environment')
    parser.add_argument('--start_level', type=int, default=0,
        help='The point in the list of levels available to the environment at \
                which to index into, eg. --num_levels 50 --start_level 50 makes \
                levels 50-99 available to this environment')
    parser.add_argument('--timesteps_total', type=int, default=1_000_000,
        help='The desired number of total timesteps spent training or testing')
    parser.add_argument('--save_interval', type=int, default=0,
        help='The interval spent in between checkpoints saved, 0 will save none,\
                and 1 will save checkpoints after every model update.')
    parser.add_argument('--load_path', type=str, default=None,
        help='The relative or absolute path to a model checkpoint if an initial \
                load from this checkpoint is desired')
    parser.add_argument('--run_dir', type=str, default=LOG_DIR+"default",
        help='The relative or absolute path to the directory where results should be logged')
    parser.add_argument('--test_mode', type=bool, default=False,
        help='True if the model should run as a testing agent, and should not be updated')
    parser.add_argument('--variable_oi', type=str, default=None,
        help='A global variable name of interest for hyperparameter searching')
    parser.add_argument('--values_oi', type=float, nargs='+', default=None,
        help='Values of interest for hyperparameter searching')
    parser.add_argument('--num_envs', type=int, default=32,
        help='The number of environments across which the agent should be run in parallel')
    parser.add_argument('--epopt_timestep', type=int, default=0,
        help='The number of timesteps to burn-in the model before it begins implementing EPO-pt')
    parser.add_argument('--paths', type=int, default=10,
        help='The number of trajectories to explore in EPO-pt')

    args = parser.parse_args()


    if args.variable_oi is not None and args.variable_oi not in globals().keys():
        raise Exception("Invalid variable of interest - var must be in list:",
                            globals().keys())

    if ((args.values_oi is None) and (args.variable_oi is None)):
        learn_helper(args)
        return
    elif ((args.values_oi is None) and (args.variable_oi is not None)):
        if args.variable_oi == 'conv_fn':
            valois = conv_fn_vals
        else:
            raise Exception('Invalid variable of interest and values pairing')
    elif ((args.values_oi is not None) and (args.variable_oi is None)):
        raise Exception('Invalid variable of interest and values pairing')
    elif ((args.values_oi is not None) and (args.variable_oi is not None)):
        valois = args.values_oi

    for valoi in valois:
        # with tf.get_default_graph().as_default():
        learn_helper(args, args.variable_oi, valoi,
            run_dir=args.run_dir+"_"+str(args.variable_oi)+"_"+str(valoi),
            seed=seeds, save_once=True)


def learn_helper(args, voi=None, valoi=None, run_dir=None, seed=None, save_once=False):
    #num_envs = args.num_envs
    if (voi is not None) and (valoi is not None):
        if isinstance(globals()[voi], int ):
            globals()[voi] = int(valoi)
        else:
            globals()[voi] = valoi

    timesteps_per_proc = args.timesteps_total
    save_interval = args.save_interval
    epopt_timestep = args.epopt_timestep
    paths = args.paths
    if save_once:
        save_interval = timesteps_per_proc//(nsteps*num_envs)
        print(save_interval)

    load_path = args.load_path
    if run_dir is None:
        run_dir = args.run_dir
    test_mode = args.test_mode

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    num_levels = args.num_levels

    # log_comm = comm.Split(0, 0)
    format_strs = ['csv', 'stdout'] #if log_comm.Get_rank() == 0 else []
    logger.configure(dir=run_dir, format_strs=format_strs)

    print("num_envs" + str(num_envs))
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs,
                      env_name=args.env_name,
                      num_levels=num_levels,
                      start_level=args.start_level,
                      distribution_mode=args.distribution_mode)
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

    logger.info("training")
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        ppo_agent.learn(
            env=venv,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            test_mode=test_mode,
            save_interval=save_interval,
            seed=seed,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            clip_vf=use_vf_clipping,
            comm=None,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            rew_scale=rew_scale,
            epopt_timestep=epopt_timestep,
            paths = paths,
            load_path=load_path
        )
    sess.close()
    tf.get_variable_scope().reuse_variables()
    # tf.reset_default_graph()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()
