import numpy as np
import imageio

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
        imageio.mimsave(save_path, img_arr, duration = 1 / fps)
    
    