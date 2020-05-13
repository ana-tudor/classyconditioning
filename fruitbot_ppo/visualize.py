import numpy as np
import imageio

import tensorflow.compat.v1 as tf
# from baselines.ppo2 import ppo2
# from fruitbot_ppo import ppo_agent
# from fruitbot_ppo.reg_impala_cnn import build_reg_impala_cnn

# Default CNN
from baselines.common.models import build_impala_cnn

# build_impala_cnn = build_reg_impala_cnn


from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.policies import build_policy
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
LOG_DIR = 'images/'
tf.disable_v2_behavior()

MAXIMUM_GIF_FRAMES = 7200

def visualize_model(env, model, timesteps = None, render = False, save_path = None, end_frames = 10, fps = 60, render_every = 1):
    
    num_envs = env.num_envs
    
    states = model.initial_state
    obs = np.zeros((num_envs,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
    obs[:] = env.reset()
    dones = [False for _ in range(num_envs)]
    
    t = 0
    
    if save_path:
        img_arr = [env.render(mode='rgb_array')]
        
    if render:
        env.render()
    
    
    while(True):
        actions, values, states, neglogpacs = model.step(obs, S=states, M=dones)
        
        obs[:], rewards, dones, infos = env.step(actions)
        
        if (timesteps is None and (np.all(dones) or t >= MAXIMUM_GIF_FRAMES)) or (timesteps is not None and t >= timesteps):
            if save_path:
                for _ in range(end_frames):
                    img_arr.append(img_arr[-1])
            break
            
        if t % render_every == 0:
            if save_path:
                img_arr.append(env.render(mode='rgb_array'))
        
            if render:
                env.render()
            
        
            
        t += 1
        
    if save_path:
        imageio.mimsave(save_path, img_arr, duration = render_every / fps)
    
    






def main():
    #Create argument parser
    parser = argparse.ArgumentParser(
        description='Process fruitbot visualization arguments.')

    parser.add_argument('--env_name', type=str, default='fruitbot',
        help='Provide an environment name available in procgen')
    parser.add_argument('--distribution_mode', type=str, default='easy',
                choices=["easy", "hard", "exploration", "memory", "extreme"])
    
    parser.add_argument('--start_level', type=int, default=0,
        help='The point in the list of levels available to the environment at \
                which to index into, eg. --num_levels 50 --start_level 50 makes \
                levels 50-99 available to this environment')
    parser.add_argument('--num_levels', type=int, default=50,
        help='Number of levels to run in the environment')
    parser.add_argument('--timesteps', type=int, default=None,
        help='The desired number of total timesteps visualizing')
    #parser.add_argument('--save_interval', type=int, default=0,
    #    help='The interval spent in between checkpoints saved, 0 will save none,\
    #            and 1 will save checkpoints after every model update.')
    parser.add_argument('--load_path', type=str,
        help='The relative or absolute path to a model checkpoint to load')
    parser.add_argument('--save_path', type=str, default=None,
        help='The relative or absolute path of a GIF file to be saved')
    parser.add_argument('--render', type=bool, default=False,
        help='True if the environment is to be rendered')
    parser.add_argument('--num_envs', type=int, default=1,
        help='The number of environments across which the agent should be run in parallel')
    parser.add_argument('--fps', type=int, default=60, help='FPS of resulting GIF')
    parser.add_argument('--compress', type=int, default=1, help='output every _ frames for compression')
    
    

    args = parser.parse_args()
    
    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()
    
    venv = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, 
                  num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)

    venv = VecNormalize(venv=venv, ob=False)
    
    
    
    
    #==============================LOAD_MODEL====================================
    '''
    Global variables defaults, values can be changed via parser
    All variables of interest which are desired to be tuned must be listed here
    '''

    #Hyperparameters

    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    update_fn=None,
    init_fn=None,
    vf_coef=0.5,
    max_grad_norm=0.5
    comm = None


    #Important variables of interest
    rew_scale = 1
    rew_baseline = False
    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    conv_fn_vals = [lambda x: build_impala_cnn(x, depths=[16, 32, 64], emb_size=256),
                lambda x: build_impala_cnn(x, depths=[32, 32], emb_size=256),
                lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)]

    seeds = [1543, 90023]

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    

    policy = build_policy(venv, conv_fn)

    # Get the nb of env
    nenvs = venv.num_envs

    # Get state_space and action_space
    ob_space = venv.observation_space
    ac_space = venv.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches


    # Instantiate the model object (that creates act_model and train_model)
    from baselines.ppo2.model import Model
    model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=0)

    model.load(args.load_path)
    
    #==============================================================================
    
    
    
    visualize_model(venv, model, render = args.render, save_path = args.save_path, render_every = args.compress, 
                    timesteps = args.timesteps, fps = args.fps)



    




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()


