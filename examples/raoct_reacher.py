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
""" Example spawning a robot in gym 

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#



import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path, get_mpc_configs_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
from storm_kit.mpc.task.reacher_task_icem import ReacherTaskiCEM
from storm_kit.mpc.control.icem import iCEM

np.set_printoptions(precision=2)


import logging

from datetime import datetime
from pytz import timezone

# Create logger
log = logging.getLogger('raoct_reacher.py')

# Prevent adding multiple handlers to the logger
if not log.handlers:
    log.setLevel(logging.INFO)

    # Create file handler which logs messages to a file 
    now = datetime.now(timezone("EST")).strftime("%b-%d-%Y_%H-%M-%S")
    fh = logging.FileHandler('/home/gymuser/raoct/storm/data/mpc_log_' + now + '.txt')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)


class reacherInterface:
    def __init__(self, gym_instance, 
                robot = 'raoct',
                mode = 'mppi',
                cuda = True, 
                viz_collision_spheres = True,
                viz_ee_target = True,
                init_state = None, 
                world_file = None):
        # Flags from argparse
        self.viz_collision_spheres = viz_collision_spheres
        self.viz_ee_target = viz_ee_target
        
        # Files
        robot_file = robot + '.yml'
        task_file = robot + '_reacher.yml'
        # world_file = 'collision_primitives_3d.yml'
        if world_file is None:
            world_file = 'wall.yml'

        # Get gym and sim
        self.gym_instance = gym_instance
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim

        # Get params
        world_yml = join_path(get_gym_configs_path(), world_file)
        world_params = self.get_params(world_yml)

        robot_yml = join_path(get_gym_configs_path(), robot_file)
        robot_params = self.get_params(robot_yml)

        sim_params = robot_params['sim_params']
        sim_params['asset_root'] = get_assets_path()
        sim_params['collision_model'] = None

        task_yml = join_path(get_mpc_configs_path(), task_file)
        if init_state is not None:
            sim_params['init_state'] = init_state
            
            #Overwrite contents of file (yaml.dump changes formatting)
            with open(task_yml, 'r') as file:
                lines = file.readlines()
            with open(task_yml, 'w') as file:
                for line in lines:
                    if line.lstrip().startswith("init_state"):
                        # Replace the old line with the new one
                        indent = len(line) - len(line.lstrip())
                        file.write(' ' * indent + f"init_state: {init_state}\n")
                    else:
                        file.write(line)
        else:
            task_params = self.get_params(task_yml)
            sim_params['init_state'] = task_params['model']['init_state']

                    
        if(cuda):
            device = torch.device('cuda', 0)
        else:
            device = torch.device('cpu') 
        self.tensor_args = {'device':device, 'dtype':torch.float32}

        # create robot simulation:
        self.robot_sim = RobotSim(gym_instance=self.gym, sim_instance=self.sim, **sim_params, device=device)
        
        # create gym environment:
        robot_pose = sim_params['robot_pose']
        self.env_ptr = gym_instance.env_list[0]
        self.robot_ptr = self.robot_sim.spawn_robot(self.env_ptr, robot_pose, coll_id=2)


        # spawn camera:
        robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
        q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
        robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])
        self.robot_sim.spawn_camera(self.env_ptr, 60, 640, 480, robot_camera_pose)

        # Get world to robot traensform
        self.w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)
        self.w_T_r_SE3 = torch.eye(4)
        quat = torch.tensor([self.w_T_r.r.w, self.w_T_r.r.x, self.w_T_r.r.y, self.w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        self.w_T_r_SE3[0,3] = self.w_T_r.p.x
        self.w_T_r_SE3[1,3] =self. w_T_r.p.y
        self.w_T_r_SE3[2,3] = self.w_T_r.p.z
        self.w_T_r_SE3[:3,:3] = rot[0]

        # Instantiate world
        self.world = World(self.gym, self.sim, self.env_ptr, world_params, w_T_r=self.w_T_r)

        # Insantiate controller
        if mode == 'mppi':
            self.mpc_control = ReacherTask(task_file, robot_file, world_file, self.tensor_args)
        elif mode == 'icem':
            self.mpc_control = ReacherTaskiCEM(task_file, robot_file, world_file, self.tensor_args)


        # Can't use self.set_goal here because object handle hasn't been instantiated yet
        # goal_as_robot_state = np.array([0.1933,  -0.9666,  1.2566, -1.8688,  -1.6111,  0.1289,
        #                                 0.0   ,  0.0    ,  0.0   ,  0.0   ,  0.0    ,  0.0])
        # self.mpc_control.update_params(goal_state=goal_as_robot_state)

        goal_pose_robot_frame = gymapi.Transform()
        goal_pose_robot_frame.p = gymapi.Vec3(-0.5, 1.25 + 0.2, 0.04) #x, z, y
        goal_pose_robot_frame.r = gymapi.Quat(-0.028, -0.693, -0.720, 0.018)

        goal_pose_world_frame = self.w_T_r.inverse()*goal_pose_robot_frame
        self.goal_pos, self.goal_q = self.transform_to_array(goal_pose_world_frame)
        self.mpc_control.update_params(goal_ee_pos = self.goal_pos, goal_ee_quat = self.goal_q)



        # Add goal relevant objects to world
        tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
        goal_pose = self.get_goal_pose()
        ee_pose = self.get_ee_pose()
        target_asset_file = "urdf/head/movable_head.urdf"
        ee_as_target_asset_file = "urdf/head/head.urdf"
        obj_asset_root = get_assets_path()

        target_object = self.world.spawn_object(target_asset_file, obj_asset_root, goal_pose, color=tray_color, name='ee_target_object')
        self.target_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 0)
        self.target_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 6)
        self.gym.set_rigid_body_color(self.env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        self.gym.set_rigid_body_color(self.env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)


        # Head that follows ee
        ee_handle = self.world.spawn_object(ee_as_target_asset_file, obj_asset_root, ee_pose, color=tray_color, name='ee_current_as_head')
        self.ee_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, ee_handle, 0)
        tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        self.gym.set_rigid_body_color(self.env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)


        # Params used while running
        self.sim_dt = self.mpc_control.exp_params['control_dt']
        self.n_dof = self.mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        self.w_robot_coord = CoordinateTransform(trans=self.w_T_r_SE3[0:3,3].unsqueeze(0),
                                                   rot=self.w_T_r_SE3[0:3,0:3].unsqueeze(0))
        self.q_des = None
        self.t_step = gym_instance.get_sim_time()

        self.goal_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        self.goal_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

        self.ee_pose = gymapi.Transform()
        self.ee_error = None
    def is_warmed_up(self):
        if isinstance(self.mpc_control, iCEM):
            return self.mpc_control.warmed_up
        else:
            return True

    def get_params(self, yaml_file: str) -> dict:
        '''
        :param yaml_file: filename
        '''        
        with open(yaml_file) as file:
            params = yaml.load(file, Loader = yaml.FullLoader)
        return params

    def transform_to_array(self, pose: gymapi.Transform) -> tuple:
        pos = np.array([pose.p.x, pose.p.y, pose.p.z])
        quat = np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z])

        return pos, quat

    def set_goal(self, goal_state: np.ndarray = None, goal_ee_pos = None, goal_ee_rot = None, goal_ee_quat = None):
        '''
        :param goal_state: array of shape (2*ndof,)
        :param goal_ee_pos: array of shape (3,)
        :param goal_ee_rot: array of shape (3,3)
        :param goal_ee_quat: array of shape (4,)
        
        '''
        self.mpc_control.update_params(goal_state = goal_state, goal_ee_pos = goal_ee_pos, goal_ee_rot = goal_ee_rot, goal_ee_quat = goal_ee_quat)
        goal_pose = self.get_goal_pose()
        # This being movable is a problem, we ideally want to just set the base handle, but then we would need to change the joint values
        # TODO: if moving base handle then set joint values to 0
        self.gym.set_rigid_transform(self.env_ptr, self.target_base_handle, goal_pose)

    def get_goal_pose(self) -> gymapi.Transform:
        object_pose = gymapi.Transform()

        g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])
        object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
        # print(object_pose.p)

        object_pose = self.w_T_r * object_pose
        return object_pose
    
    def get_ee_pose(self) -> gymapi.Transform:
        ee_pose = gymapi.Transform()

        filtered_state_mpc = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))
        curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)

        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
        ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
        ee_pose = copy.deepcopy(self.w_T_r) * copy.deepcopy(ee_pose)

        return ee_pose
    
    def update_goal_from_world(self):
        '''
        updates mpc goal pose if world goal pose has changed

        this discrepancy occurs primarily form pose override with the GUI
        '''
        pose = copy.deepcopy(self.world.get_pose(self.target_body_handle))
        pose = copy.deepcopy(self.w_T_r.inverse() * pose)

        if(np.linalg.norm(self.goal_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(self.goal_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
            self.goal_pos, self.goal_q = self.transform_to_array(pose)
            self.set_goal(goal_ee_pos=self.goal_pos,
                            goal_ee_quat=self.goal_q)
        
    def step(self):
        self.gym_instance.step()
        self.update_goal_from_world()
        # self.get_goal_pose()

        self.t_step += self.sim_dt        
        current_robot_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))
        
        command = self.mpc_control.get_command(self.t_step, current_robot_state, control_dt=self.sim_dt, WAIT=True)

        filtered_state_mpc = current_robot_state #mpc_control.current_state
        curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
        
        # get position command:
        self.q_des = copy.deepcopy(command['position'])
        
        # Critical! This updates self.mpc_control.rollout_fn.link_pos_seq and 
        #                        self.mpc_control.rollout_fn.link_rot_seq
        self.ee_error = self.mpc_control.get_current_error(filtered_state_mpc)
        # print(ee_error)
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
        # self.mpc_control.controller.rollout_fn.update_link_poses()

        # get current pose:
        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        self.ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
        self.ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
        
        log.info(f"time: {self.t_step:0.4f} s")
        log.info(f"position: {e_pos[0]:0.4f}, {e_pos[1]:0.4f}, {e_pos[2]:0.4f}")

        self.ee_pose = copy.deepcopy(self.w_T_r) * copy.deepcopy(self.ee_pose)
        
        if(self.viz_ee_target):
            self.gym.set_rigid_transform(self.env_ptr, self.ee_body_handle, copy.deepcopy(self.ee_pose))

        # print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(self.mpc_control.opt_dt),
        #       "{:.3f}".format(self.mpc_control.mpc_dt))
    
        
        self.gym_instance.clear_lines()
        top_trajs = self.mpc_control.top_trajs.cpu().float()#.numpy()
        n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
        w_pts = self.w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

        top_trajs = w_pts.cpu().numpy()
        color = np.array([0.0, 1.0, 0.0])
        for k in range(top_trajs.shape[0]):
            pts = top_trajs[k,:,:]
            color[0] = float(k) / float(top_trajs.shape[0])
            color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            self.gym_instance.draw_lines(pts, color=color)
        if self.viz_collision_spheres:

            ### From PR https://github.com/NVlabs/storm/pull/10/commits/a03c4e0057290bd6796a549623461b932fc0a979
            link_pos_seq = copy.deepcopy(self.mpc_control.controller.rollout_fn.link_pos_seq)
            link_rot_seq = copy.deepcopy(self.mpc_control.controller.rollout_fn.link_rot_seq)
            batch_size = link_pos_seq.shape[0]
            horizon = link_pos_seq.shape[1]
            n_links = link_pos_seq.shape[2]
            link_pos = link_pos_seq.view(batch_size * horizon, n_links, 3)
            link_rot = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)

            self.mpc_control.controller.rollout_fn.robot_self_collision_cost.coll.update_batch_robot_collision_objs(link_pos, link_rot)

            spheres = self.mpc_control.controller.rollout_fn.get_spheres()
            arr = None
            for sphere in spheres:
                if arr is None:
                    arr = np.array(sphere[1:,:,:4].cpu().numpy().squeeze())
                else:
                    arr = np.vstack((arr,sphere[1:,:,:4].cpu().numpy().squeeze()))
            [self.gym_instance.draw_collision_spheres(sphere,self.w_T_r) for sphere in arr]
            ####################################################################################################

        if self.is_warmed_up():
            self.robot_sim.command_robot_position(self.q_des, self.env_ptr, self.robot_ptr)


def mpc_robot_interactive(args, gym_instance):
    vis_ee_target = True
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'

    
    gym = gym_instance.gym
    sim = gym_instance.sim
    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    if(args.cuda):
        device = 'cuda'
    else:
        device = 'cpu'
    
    sim_params['collision_model'] = None
    # create robot simulation:
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)
    
    # create gym environment:
    robot_pose = sim_params['robot_pose']
    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    device = torch.device('cuda', 0) 

    
    tensor_args = {'device':device, 'dtype':torch.float32}
    

    # spawn camera:
    robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
    q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
    robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])

    
    robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0,3] = w_T_r.p.x
    w_T_robot[1,3] = w_T_r.p.y
    w_T_robot[2,3] = w_T_r.p.z
    w_T_robot[:3,:3] = rot[0]

    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)
    
    
    # table_dims = np.ravel([1.5,2.5,0.7])
    # cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])
    # cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    # table_dims = np.ravel([0.35,0.1,0.8])
    # cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    # table_dims = np.ravel([0.3,0.1,0.8])
    
    # get camera data:
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    
    start_qdd = torch.zeros(n_dof, **tensor_args)
    # update goal:
    exp_params = mpc_control.exp_params
    current_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
    ee_list = []
    

    mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

    # goal_as_robot_state = np.array([-0.3,  0.3,  0.2, -2.0,  0.0,  2.4,
    #                                 0.0,  0.0,  0.0,  0.0,  0.0,  0.0])
    goal_as_robot_state = np.array([0.1933,  -0.9666,  1.2566, -1.8688,  -1.6111,  0.1289,
                                    0.0,  0.0,  0.0,  0.0,  0.0,  0.0])


    #bl_state[:n_dofs] are joint values
    #bl_state[n_dofs:] are joint velocities (this gets passed through compute_forward_kinematics)
    
    x_des_list = [goal_as_robot_state]
    
    ee_error = 10.0
    j = 0
    t_step = 0
    i = 0
    x_des = x_des_list[0]
    

    mpc_control.update_params(goal_state=x_des)

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    # spawn object (this pose doesnt matter):
    x,y,z = 0.0, 0.0, 0.0

    tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
    # asset_options = gymapi.AssetOptions()
    # asset_options.armature = 0.001
    # asset_options.fix_base_link = True
    # asset_options.thickness = 0.002


    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(x, y, z)
    object_pose.r = gymapi.Quat(0,0,0, 1)

    obj_asset_file = "urdf/head/movable_head.urdf"
    # obj_asset_file = "urdf/mug/movable_mug.urdf" 
    obj_asset_root = get_assets_path()
    
    if(vis_ee_target):        
        target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_target_object')
        obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
        obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
        gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)


        # Head that follows ee
        obj_asset_file = "urdf/head/head.urdf"
        obj_asset_root = get_assets_path()
        ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_current_as_head')
        ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
        tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)



    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])

    object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
    object_pose = w_T_r * object_pose
    if(vis_ee_target):
        gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    prev_acc = np.zeros(n_dof)
    ee_pose = gymapi.Transform()
    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                        rot=w_T_robot[0:3,0:3].unsqueeze(0))

    rollout = mpc_control.controller.rollout_fn
    tensor_args = mpc_tensor_dtype
    sim_dt = mpc_control.exp_params['control_dt']
    
    log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                'qddd_des':[]}

    q_des = None
    qd_des = None
    t_step = gym_instance.get_sim_time()

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())



    while(i > -100):
        try:
            gym_instance.step()
            if(vis_ee_target):
                pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))

                pose = copy.deepcopy(w_T_r.inverse() * pose)
                if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                    g_pos[0] = pose.p.x
                    g_pos[1] = pose.p.y
                    g_pos[2] = pose.p.z
                    g_q[1] = pose.r.x
                    g_q[2] = pose.r.y
                    g_q[3] = pose.r.z
                    g_q[0] = pose.r.w

                    mpc_control.update_params(goal_ee_pos=g_pos,
                                              goal_ee_quat=g_q)
            t_step += sim_dt
            
            current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
            

            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)

            filtered_state_mpc = current_robot_state #mpc_control.current_state
            curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity']) #* 0.5
            qdd_des = copy.deepcopy(command['acceleration'])
            
            ee_error = mpc_control.get_current_error(filtered_state_mpc)
             
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            
            # get current pose:
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
            
            ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
            
            if(vis_ee_target):
                gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))

            # print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt),
            #       "{:.3f}".format(mpc_control.mpc_dt))
        
            
            gym_instance.clear_lines()
            top_trajs = mpc_control.top_trajs.cpu().float()#.numpy()
            n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


            top_trajs = w_pts.cpu().numpy()
            color = np.array([0.0, 1.0, 0.0])
            for k in range(top_trajs.shape[0]):
                pts = top_trajs[k,:,:]
                color[0] = float(k) / float(top_trajs.shape[0])
                color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                gym_instance.draw_lines(pts, color=color)
            

            ### From PR https://github.com/NVlabs/storm/pull/10/commits/a03c4e0057290bd6796a549623461b932fc0a979
            link_pos_seq = copy.deepcopy(mpc_control.controller.rollout_fn.link_pos_seq)
            link_rot_seq = copy.deepcopy(mpc_control.controller.rollout_fn.link_rot_seq)
            batch_size = link_pos_seq.shape[0]
            horizon = link_pos_seq.shape[1]
            n_links = link_pos_seq.shape[2]
            link_pos = link_pos_seq.view(batch_size * horizon, n_links, 3)
            link_rot = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
            mpc_control.controller.rollout_fn.robot_self_collision_cost.coll.update_batch_robot_collision_objs(link_pos, link_rot)

            spheres = mpc_control.controller.rollout_fn.get_spheres()
            arr = None
            for sphere in spheres:
                if arr is None:
                    arr = np.array(sphere[1:,:,:4].cpu().numpy().squeeze())
                else:
                    arr = np.vstack((arr,sphere[1:,:,:4].cpu().numpy().squeeze()))

            [gym_instance.draw_collision_spheres(sphere,w_T_r) for sphere in arr]
            ####################################################################################################

            robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
            #robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            current_state = command
            
            i += 1

        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
    mpc_control.close()
    return 1 
    
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('-m', '--mode', type=str, default='mppi', help='Choose TrajOpt algo: \n' + \
                                                                        'mppi, icem')
    parser.add_argument('--robot', type=str, default='raoct', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    parser.add_argument('-vcs', '--viz_collision_spheres', action='store_true', default=True, help='visualize collision spheres')
    parser.add_argument('-vet', '--viz_ee_target', action='store_true', default=True, help='visualize ee as target')



    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)
    
    init1 = [np.pi/180*angle for angle in [90, -90, 150, -150, -90, 0]]
    init2 = [np.pi/180*angle for angle in [90, -60, 120, -150, -90, 0]]
    if args.mode not in ["mppi", "icem"]:
        raise ValueError("Invalid TrajOpt mode provided")


    # mpc_robot_interactive(args, gym_instance)
    mpc_robot_interactive_class = reacherInterface(gym_instance,
                                                   robot = args.robot,
                                                   mode = args.mode,
                                                   cuda = args.cuda,
                                                   viz_collision_spheres= args.viz_collision_spheres,
                                                   viz_ee_target= args.viz_ee_target,
                                                   init_state=init2
                                                   )
    mpc_robot_interactive_class.set_goal(goal_state = [np.pi/180*angle for angle in [-90, -140, -80, -50, 90, 90]] + 6*[0])
    import time
    while True:
        # start = time.time()
        try:
            mpc_robot_interactive_class.step()
        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
        finally:
            pass
            # print(f"Loop time: {time.time() - start}")
    mpc_robot_interactive_class.mpc_control.close()
