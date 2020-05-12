import numpy as np
from .runner import Runner

class EPOptRunner(Runner):
    def run(self, *, paths):
        nenvs = self.env.num_envs
        obs_shape = self.obs.shape
        n_mb_obs = np.zeros(shape=(paths, self.nsteps, obs_shape[0], obs_shape[1], obs_shape[2], obs_shape[3]))
        n_mb_rewards = np.zeros(shape=(paths, self.nsteps, nenvs))
        n_mb_actions = np.zeros(shape=(paths, self.nsteps, nenvs))
        n_mb_values = np.zeros(shape=(paths, self.nsteps, nenvs))
        n_mb_dones = np.zeros(shape=(paths, self.nsteps, nenvs))
        n_mb_neglogpacs = np.zeros(shape=(paths, self.nsteps, nenvs))
        n_epinfos = [[[] for _2 in range(self.nsteps)] for _ in range(paths)]
        mb_states = self.states
        obs_start = self.obs.copy()
        dones_start = self.dones.copy()

        for N in range(paths):
            # mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, epinfos = \
            #     n_mb_obs[N], n_mb_rewards[N], n_mb_actions[N], n_mb_values[N], n_mb_dones[N], n_mb_neglogpacs[N], n_epinfos[N]
            
            epinfos = n_epinfos[N]
            self.obs[:] = obs_start
            self.dones[:] = dones_start

            for t in range(self.nsteps):
                epinfos_step = epinfos[t]

                actions, values, self.states, neglogpacs = self.model.step(self.obs, S = self.states, M = self.dones)
                n_mb_obs[N, t] = self.obs.copy()
                n_mb_actions[N, t] = actions
                n_mb_values[N, t] = values
                n_mb_neglogpacs[N, t] = neglogpacs
                n_mb_dones[N, t] = self.dones.copy()
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                rewards *= self.rew_scale

                # print(infos)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos_step.append(maybeepinfo)
                    else: epinfos_step.append("None")
                n_mb_rewards[N, t] = rewards
                # Stop once single thread has finished an episode
                # if self.dones:  # ie [True]
                #     break
            
        # Compute the worst epsilon paths and concatenate them
        # print(n_mb_rewards.shape)
        episode_returns = np.sum(n_mb_rewards, axis = 1)
        # print(episode_returns.shape)
        # cutoff = np.percentile(episode_returns, 100*epsilon, axis = 0)
        # print(len(cutoff))
        
        #Use k smallest episode returns
        idxs = np.argmin(episode_returns, axis = 0)
        # print("idxs", idxs)
        #idxs = np.argsort(episode_returns,axis=0)[:3,:,:]

        # indexes = [i for i, r in enumerate(mb_rewards) if r <= cutoff]

        #index into all arrays 
        # print(n_mb_actions.shape)
        # print(np.transpose(n_mb_actions, axes=[2,0,1]).shape)
        # # n_mb_actions = np.squeeze(np.take_along_axis(np.transpose(n_mb_actions, axes=[2,0,1]), np.expand_dims(idxs, axis=(1, 2)), axis=1))
        # print(n_mb_actions.shape) # number taken, num steps, num envs

        #Set self.obs to correct thing

        next_obs = np.squeeze(np.take_along_axis(np.transpose(n_mb_obs, axes=[2, 0, 1, 3, 4, 5]), np.expand_dims(idxs, axis=(1, 2, 3, 4, 5)), axis=1))
        mb_obs = np.transpose(next_obs, axes=[1, 0, 2, 3, 4]).astype(self.obs.dtype)
        mb_rewards = np.squeeze(np.take_along_axis(np.transpose(n_mb_rewards, axes=[2, 0, 1]), np.expand_dims(idxs, axis=(1, 2)), axis=1)).T.astype(np.float32)
        mb_actions = np.squeeze(np.take_along_axis(np.transpose(n_mb_actions, axes=[2, 0, 1]), np.expand_dims(idxs, axis=(1, 2)), axis=1)).T
        mb_values = np.squeeze(np.take_along_axis(np.transpose(n_mb_values, axes=[2, 0, 1]), np.expand_dims(idxs, axis=(1, 2)), axis=1)).T.astype(np.float32)
        mb_dones = np.squeeze(np.take_along_axis(np.transpose(n_mb_dones, axes=[2, 0, 1]), np.expand_dims(idxs, axis=(1, 2)), axis=1)).T.astype(np.bool)
        mb_neglogpacs = np.squeeze(np.take_along_axis(np.transpose(n_mb_neglogpacs, axes=[2, 0, 1]), np.expand_dims(idxs, axis=(1, 2)), axis=1)).T.astype(np.float32)

        # mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        epinfos = []

        for i in range(nenvs):
            for j in range(self.nsteps):
                epinfo = n_epinfos[idxs[i]][j][i]
                if epinfo != "None":
                    epinfos.append(epinfo)
        # print(len(epinfos))
        # mb_epinfos.extend(next_epinfos)
        
        # mb_obs.extend(next_obs)
        # mb_rewards.extend(next_rewards)
        # mb_actions.extend(next_actions)
        # mb_values.extend(next_values)
        # mb_dones.extend(next_dones)
        # mb_neglogpacs.extend(next_neglogpacs)

        # if ([2,2] < 1):
        #     print("stop")

        # mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        # epinfos = []
        # taken = 0
        # for N in range(paths):
        #     #if n_mb_rewards[N] <= cutoff:
        #     if np.all(episode_returns[N] <= cutoff):
        #         taken += 1 
        #         # only count the episodes that are returned
        #         # num_episodes += 1
        #         # "cache" values to keep track of final ones
        #         next_obs = n_mb_obs[N]
        #         next_rewards = n_mb_rewards[N]
        #         next_actions = n_mb_actions[N]
        #         next_values = n_mb_values[N]
        #         next_dones = n_mb_dones[N]
        #         next_neglogpacs = n_mb_neglogpacs[N]
        #         next_epinfos = n_epinfos[N]
        #         # concatenate
        #         mb_obs.extend(next_obs)
        #         mb_rewards.extend(next_rewards)
        #         mb_actions.extend(next_actions)
        #         mb_values.extend(next_values)
        #         mb_dones.extend(next_dones)
        #         mb_neglogpacs.extend(next_neglogpacs)
        #         epinfos.extend(next_epinfos)
        # print(taken)
        total_steps = len(mb_rewards)

        #  batch of steps to batch of rollouts
        # mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        # mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        # mb_actions = np.asarray(mb_actions)
        # mb_values = np.asarray(mb_values, dtype=np.float32)
        # mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        # mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # We can't just use self.obs etc, because the last of the N paths
        # may not be included in the update
        self.obs = mb_obs[-1]
        self.dones = mb_dones[-1]
        last_values = self.model.value(self.obs, S = self.states, M = self.dones)

        #  discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        # Instead using nsteps, use the total number of steps in all kept trajectories
        #for t in reversed(range(self.nsteps)):
        for t in reversed(range(total_steps)):
            #if t == self.nsteps - 1:
            if t == total_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])