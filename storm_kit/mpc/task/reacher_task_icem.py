#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import yaml
import numpy as np

from ...util_file import get_mpc_configs_path as mpc_configs_path
# from ...mpc.rollout.arm_reacher import ArmBase
from ...mpc.rollout.arm_reacher import ArmReacher

from ...mpc.control import iCEM
from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .task_base import BaseTask

# from pytorch_icem import iCEM


class ReacherTaskiCEM(BaseTask):
    def __init__(self, task_file='ur10.yml', robot_file='ur10_reacher.yml', world_file='collision_env.yml', tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super().__init__(tensor_args=tensor_args)
        
        

        self.controller = self.init_controller(task_file, robot_file, world_file)
        # self.controller = self.init_mppi(task_file, robot_file, world_file)
        self.init_aux()
        
    def get_rollout_fn(self, **kwargs):
        rollout_fn = ArmReacher(**kwargs)
        return rollout_fn


    def init_mppi(self, task_file, robot_file, world_file):
        '''
        this implementation does not use mppi, but we can alias our controller initialization function still
        '''
        self.init_controller(self, task_file, robot_file, world_file)

    def init_controller(self, task_file, robot_file, collision_file):
        robot_yml = join_path(get_gym_configs_path(), robot_file)
        
        with open(robot_yml) as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)

        world_yml = join_path(get_gym_configs_path(), collision_file)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)

        mpc_yml_file = join_path(mpc_configs_path(), task_file)

        with open(mpc_yml_file) as file:
            exp_params = yaml.load(file, Loader=yaml.FullLoader)
        exp_params['robot_params'] = exp_params['model'] #robot_params
        
        
        rollout_fn = self.get_rollout_fn(exp_params=exp_params, tensor_args=self.tensor_args, world_params=world_params)
        
    
        icem_params = exp_params['icem']
        icem_params['num_particles'] = exp_params['common_controller_params']['num_particles']
        icem_params['horizon'] = exp_params['common_controller_params']['horizon']

        ### TODO: fit icem where it makes sense
        dynamics_model = rollout_fn.dynamics_model
        icem_params['d_action'] = dynamics_model.d_action
        icem_params['action_lows'] = -exp_params['model']['max_acc'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        icem_params['action_highs'] = exp_params['model']['max_acc'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        init_q = torch.tensor(exp_params['model']['init_state'], **self.tensor_args)
        init_action = torch.zeros((icem_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        init_action[:,:] += init_q
        icem_params['rollout_fn'] = rollout_fn
        icem_params['tensor_args'] = self.tensor_args
        controller = iCEM(**icem_params)
        self.exp_params = exp_params
        return controller
