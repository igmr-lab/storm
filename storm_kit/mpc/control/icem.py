
import copy

import numpy as np
import scipy.special
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import normalize as f_norm
from .control_utils import generate_noise, scale_ctrl, generate_gaussian_halton_samples, generate_gaussian_sobol_samples, gaussian_entropy, matrix_cholesky, batch_cholesky, get_stomp_cov

from .control_utils import cost_to_go, matrix_cholesky, batch_cholesky
# from .olgaussian_mpc import OLGaussianMPC
from .control_base import Controller
import colorednoise
import torch.autograd.profiler as profiler

class iCEM(Controller):
    """
    .. inheritance-diagram:: MPPI
       :parts: 1

    Class that implements Model Predictive Path Integral Controller
    
    Implementation is based on 
    Williams et. al, Information Theoretic MPC for Model-Based Reinforcement Learning
    with additional functions for updating the covariance matrix
    and calculating the soft-value function.

    """

    def __init__(self,
                 d_action,
                 horizon,
                 beta,
                 gamma,
                 action_lows,
                 action_highs,
                 squash_fn,
                 sigma = None,
                 num_particles = 100,
                 num_elites = 10,
                 elites_keep_fraction = 0.5,
                 alpha = 0.05,
                 noise_beta = 3,
                 warmup_iters = 100,
                 online_iters = 100,
                 includes_x0 = False,
                 rollout_fn=None,
                 sample_mode='mean',
                 hotstart=True,
                 seed=0,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 visual_traj='state_seq'):
        
        super(iCEM, self).__init__( d_action,
                                    action_lows,
                                    action_highs,
                                    horizon,
                                    gamma,
                                    n_iters =1,
                                    rollout_fn = rollout_fn,
                                    sample_mode = sample_mode,
                                    hotstart = hotstart,
                                    seed = seed,
                                    tensor_args = tensor_args)
        
        self.beta = beta
        self.visual_traj = visual_traj


        self.squash_fn = squash_fn
        self.best_traj = None
        self.num_elites = num_elites
        self.num_particles = num_particles

        if sigma is None:
            sigma = torch.ones(self.d_action, **self.tensor_args).float()
        elif isinstance(sigma, float):
            sigma = torch.ones(self.d_action, **self.tensor_args).float() * sigma
        self.sigma = sigma

        self.warmup_iters = warmup_iters
        self.online_iters = online_iters
        self.includes_x0 = includes_x0
        self.noise_beta = noise_beta
        self.alpha = alpha
        self.keep_fraction = elites_keep_fraction

        # initialise mean and std of actions
        self.mean_action = torch.zeros(self.horizon, self.d_action, **self.tensor_args)
        self.sigma = torch.tensor(self.sigma).to(**self.tensor_args)
        self.std = self.sigma.clone()

        self.kept_elites = None
        self.warmed_up = False
        self.num_kept_elites = int(self.num_particles*self.keep_fraction) 
        self.reset_distribution()
        
    def generate_rollouts(self, state):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, costs,  
            actions

            Parameters
            ----------
            state : dict or np.ndarray
                Initial state to set the simulation env to
         """
        
        act_seq = self.sample_actions(state=state) # sample noise from covariance of current control distribution

        trajectories = self._rollout_fn(state, act_seq)
        return trajectories
    
    def sample_actions(self, state = None):
        if self.kept_elites is None:
            U = self._sample_actions(self.num_particles)
        else:
            U = self._sample_actions(self.num_particles - len(self.kept_elites))
            U = torch.cat((U, self.kept_elites), dim = 0)

        return U

    def _sample_actions(self, N):
        sample_shape = (N, self.horizon, self.d_action)
        # colored noise
        if self.noise_beta > 0:
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
            samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(N, self.d_action, self.horizon)).transpose(
                [0, 2, 1])
            samples = torch.from_numpy(samples).to(**self.tensor_args)
        else:
            samples = torch.randn(size = sample_shape, **self.tensor_args)

        U = self.mean_action + self.std * samples
        return U


    def _shift(self, shift_steps):
        """
            Predict mean for the next time step by
            shifting the current mean forward by one step
        """
        if(shift_steps == 0):
            return
        self.mean_action = self.mean_action.roll(-shift_steps,0)
        self.mean_action[-1] = torch.zeros(self.d_action, **self.tensor_args)
        self.std = self.sigma.clone()
        self.best_traj = self.best_traj.roll(-shift_steps,0)        
        if self.kept_elites is not None:
            self.kept_elites = self.kept_elites.roll(-shift_steps, dims=1)
            self.kept_elites[:, -1] = self.sigma * torch.randn(len(self.kept_elites), self.d_action, **self.tensor_args)


    def _exp_util(self, costs, actions):
        """
            Calculate weights using exponential utility
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)
        # if not self.time_based_weights: traj_costs = traj_costs[:,0]
        traj_costs = traj_costs[:,0]
        #control_costs = self._control_costs(actions)

        total_costs = traj_costs #+ self.beta * control_costs
        
        # #calculate soft-max
        w = torch.softmax((-1.0/self.beta) * total_costs, dim=0)
        self.total_costs = total_costs
        return w


    def _update_distribution(self, trajectories):
        # Unpack trajecotires
        actions = trajectories['actions']
        costs = trajectories['costs']
        vis_seq = trajectories[self.visual_traj].to(**self.tensor_args)
        
        w = self._exp_util(costs, actions) #This updates self.total_costs

        # Parse trajectories for top performers
        # top_values, top_idx = torch.topk(self.total_costs, 10)
        self.best_idx = torch.argmax(w)
        self.best_traj = torch.index_select(actions, 0, self.best_idx).squeeze(0)

        self.top_values, self.top_idx = torch.topk(-self.total_costs, self.num_kept_elites)

        self.top_trajs = torch.index_select(vis_seq, 0, self.top_idx[:self.num_elites])
        elites = actions[self.top_idx[:self.num_elites]]
        # fit around mean of elites
        new_mean = elites.mean(dim=0)
        new_std = elites.std(dim=0)

        self.mean_action = (1 - self.alpha) * new_mean + self.alpha * self.mean_action  # [h,d]
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std        
        self.kept_elites = actions[self.top_idx]


    
    def _get_action_seq(self, mode='mean'):
        # if mode == 'mean':
        #     act_seq = self.mean_action.clone()
        # elif mode == 'sample':
        #     delta = self.generate_noise(shape=torch.Size((1, self.horizon)),
        #                                 base_seed=self.seed_val + 123 * self.num_steps)
        #     act_seq = self.mean_action + torch.matmul(delta, self.full_scale_tril)
        # else:
        #     raise ValueError('Unidentified sampling mode in get_next_action')
        act_seq = scale_ctrl(self.best_traj, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        return act_seq
    
    def _control_costs(self, actions):

        delta = actions - self.mean_action.unsqueeze(0)
        u_normalized = self.mean_action.matmul(self.full_inv_cov).unsqueeze(0)
        control_costs = 0.5 * u_normalized * (self.mean_action.unsqueeze(0) + 2.0 * delta)
        control_costs = torch.sum(control_costs, dim=-1)
        control_costs = cost_to_go(control_costs, self.gamma_seq)
        control_costs = control_costs[:,0]
        return control_costs

    def _calc_val(self, trajectories):
        costs = trajectories["costs"].to(**self.tensor_args)
        actions = trajectories["actions"].to(**self.tensor_args)
        delta = actions - self.mean_action.unsqueeze(0)
        
        traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs + self.beta * control_costs
        val = -self.beta * torch.logsumexp((-1.0/self.beta) * total_costs)
        return val
        

    def reset_mean(self):
        self.mean_action = torch.zeros((self.horizon, self.d_action), **self.tensor_args)
        self.best_traj = self.mean_action.clone()

    def reset_covariance(self):
        self.std = self.sigma.clone()

    def reset_distribution(self):
        """
            Reset control distribution
        """
        self.reset_mean()
        self.reset_covariance()
        self.kept_elites = None
        self.warmed_up = False

    def optimize(self, state, calc_val=False, shift_steps=1, n_iters=None):
        """
        Optimize for best action at current state

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal action from
        
        calc_val : bool
            If true, calculate the optimal value estimate
            of the state along with action
                
        Returns
        -------
        action : torch.Tensor
            next action to execute
        value: float
            optimal value estimate (default: 0.)
        info: dict
            dictionary with side-information
        """

        # n_iters = n_iters if n_iters is not None else self.n_iters
        # get input device:
        inp_device = state.device
        inp_dtype = state.dtype
        state.to(**self.tensor_args)

        info = dict(rollout_time=0.0, entropy=[])
        # shift distribution to hotstart from previous timestep
        # if self.hotstart:
        #     self._shift(shift_steps)
        # else:
        #     self.reset_distribution()
        if self.warmed_up:
            n_iters = self.online_iters
        else:
            self.reset_distribution()
            n_iters = self.warmup_iters

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                for _ in range(n_iters):
                    # generate random simulated trajectories
                    trajectory = self.generate_rollouts(state)

                    # update distribution parameters
                    with profiler.record_function("controller_update"):
                        self._update_distribution(trajectory)
                    info['rollout_time'] += trajectory['rollout_time']

                    # check if converged
                    if self.check_convergence():
                        break
        self.trajectories = trajectory
        self._shift(shift_steps)
        #calculate best action
        # curr_action = self._get_next_action(state, mode=self.sample_mode)
        curr_action_seq = self._get_action_seq(mode=self.sample_mode)
        #calculate optimal value estimate if required
        value = 0.0
        if calc_val:
            trajectories = self.generate_rollouts(state)
            value = self._calc_val(trajectories)

        self.num_steps += 1
        self.warmed_up = True

        return curr_action_seq.to(inp_device, dtype=inp_dtype), value, info