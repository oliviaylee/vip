# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import namedtuple
import numpy as np
import gym
from gym.spaces.box import Box
import glob
import omegaconf
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from dm_control.mujoco import engine

from PIL import Image
from pathlib import Path
import pickle
from torchvision.utils import save_image
import hydra
import os
import sys 
from trajopt.envs.gym_env import GymEnv

sys.path.append('/iris/u/oliviayl/repos/affordance-learning/d5rl/benchmark/domains/relay-policy-learning/adept_envs/')
# import jaxrl2
# from examples.train_offline_pixels_kitchen import make_env
import adept_envs
from adept_envs import franka


CAMERAS = {
    0: dict(distance=2.1, lookat=[-0.4, 0.5, 2.0], azimuth=70,
            elevation=-37.5),
    1: dict(distance=2.2,
            lookat=[-0.2, 0.75, 2.0],
            azimuth=150,
            elevation=-30.0),
    2: dict(distance=4.5, lookat=[-0.2, 0.75, 2.0], azimuth=-66, elevation=-65),
    3: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            ),  # original, as in https://relay-policy-learning.github.io/
    4: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70,
            elevation=-50),  # angled up to get a more top-down view
    5: dict(distance=2.65, lookat=[0, 0, 2.0], azimuth=90, elevation=-60
            ),  # similar to appendix D of https://arxiv.org/pdf/1910.11956.pdf
    6: dict(distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-60
            ),  # 3-6 are first person views at different angles and distances
    7: dict(
        distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-45
    ),  # problem w/ POV is that the knobs can be hidden by the hinge drawer and arm
    8: dict(distance=2.9, lookat=[-0.05, 0.5, 2.0], azimuth=90, elevation=-50),
    9: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90,
            elevation=-50),  # move back so less of cabinets
    10: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-35),
    11: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-10),

    12: dict(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60) # LEXA view
}


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def _get_embedding(embedding_name='resnet50', load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == 'resnet34':
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet50':
        model = models.resnet50(pretrained=prt, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim

def env_constructor(env_name, device='cuda', image_width=256, image_height=256,
                    camera_name=None, embedding_name='resnet50', pixel_based=True,
                    embedding_reward=True,
                    render_gpu_id=0, load_path="", proprio=False, goal_timestep=49, init_timestep=0):
    # print("Constructing environment with GPU", render_gpu_id)
    if not pixel_based and not embedding_reward: 
            env = GymEnv(env_name)
    else:
        env = gym.make('kitchen_relax_rpl-v1') # gym.make(env_name)
        ## Wrap in pixel observation wrapper
        # env = MuJoCoPixelObs(env, width=image_width, height=image_height, camera_name=camera_name, device_id=render_gpu_id)
        ## Wrapper which encodes state in pretrained model (additionally compute reward)
        # env = StateEmbedding(env, embedding_name=embedding_name, device=device, load_path=load_path, proprio=proprio, camera_name=camera_name, env_name=env_name, pixel_based=pixel_based, embedding_reward=embedding_reward, goal_timestep=goal_timestep, init_timestep=init_timestep)
        env = CustomEmbedding(env, device='cuda', demo_path="/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_kettle_bottomknob_switch_slide/20230528T010656-1be74c034d6940f1a2d9e63d24fc7f83-218.npz")
        env = GymEnv(env)
    return env

class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class CustomEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env, device='cuda', demo_path="/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_kettle_bottomknob_switch_slide/20230528T010656-1be74c034d6940f1a2d9e63d24fc7f83-218.npz"):
        super().__init__(env)

        # Load demo, extract end effector positions / proprio data
        self.data = np.load(demo_path)
        # self.domain = ... # RANDOMIZED ENV (D5RL) 
        # self.unwrapped.reset(domain ...) # RANDOMIZED ENV RESET

        self.first_frame = self.data['image'][0]
        self.first_frame_camera12 = self.data['extra_image_camera_12_rgb'][0]
        self.first_frame_gripper = self.data['extra_image_camera_gripper_rgb'][0]

        imgs = [Image.fromarray(img) for img in self.data['image']]
        imgs[0].save(f"./demo.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)

        # Image.fromarray(self.first_frame).save('image0.png')
        # Image.fromarray(self.first_frame_camera12).save('camera12_img0.png')
        # Image.fromarray(self.first_frame_gripper).save('gripper_img0.png')

        # DO A REACHING TASK AND START FROM INITIAL FRAME, JUST USE ROBOT POS AS REWARD
        # Later on, set qpos/qvel to the first frame of truncated demo
        self.data_keys = self.data.files
        self.robot_pos = self.data["qpos"][:, :3] # self.data["proprio"]? check ee_pos and proprio
        self.kettle_pos = self.data["qpos"][:, -7:-4] # init_qpos: last 7 are kettle xyz + quat
        self.kettle_rew = self.data["reward kettle"]
        self.first_contact = list(self.kettle_rew).index(1)

        # REACHING TRAJECTORY
        self.robot_traj = self.robot_pos[:self.first_contact]
        if len(self.robot_traj) > 50:
            self.robot_traj = self.robot_traj[:50]
        else:
            padding = np.stack([self.robot_traj[-1] for _ in range(50 - len(self.robot_traj))], axis=0)
            self.robot_traj = np.concatenate((self.robot_traj, padding))

        # KETTLE INTERACTION TRAJECTORY
        # self.first_contact = list(self.kettle_rew).index(1)
        # self.robot_traj = self.robot_pos[max(0, self.first_contact - 25):min(len(self.robot_pos), self.first_contact + 25)]
        # self.kettle_traj = self.kettle_pos[max(0, self.first_contact - 25):min(len(self.kettle_pos), self.first_contact + 25)]
        # Padding
        # if self.first_contact - 25 < 0:
        #     self.robot_traj = [self.robot_traj[0] * abs(self.first_contact)] + self.robot_traj
        #     self.kettle_traj = [self.kettle_traj[0] * abs(self.first_contact)] + self.kettle_traj
        # if self.first_contact + 25 > len(self.kettle_pos):
        #     self.robot_traj = [self.robot_traj[-1] * len(self.robot_traj) - (self.first_contact + 25)] + self.robot_traj
        #     self.kettle_traj = [self.kettle_traj[-1] * len(self.kettle_pos) - (self.first_contact + 25)] + self.kettle_traj
        self.traj_len = 50 # len(self.data["proprio"])
        assert len(self.robot_traj) == self.traj_len
        # assert len(self.kettle_traj) == self.traj_len
        
        self.init_qpos = self.data["init_qpos"]
        self.init_qvel = self.data["init_qvel"]
        self.step_num = 0
        self.real_step = True
        self.add_cameras(camera_ids=range(13))
        self.add_cameras(camera_ids=range(13))

        if device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

    def get_obs(self):
        return self.unwrapped.get_obs()
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Uncomment after testing base version
        obs_ee = info['obs_dict']['ee_qp'][:3]
        obs_kettle = info['obs_dict']['obj_qp'][-6:-3]
        gt_obs_ee = self.robot_pos[self.step_num]
        gt_obs_kettle = self.kettle_pos[self.step_num]
        reward_ee = -np.linalg.norm(obs_ee - gt_obs_ee)
        reward_kettle = -np.linalg.norm(obs_kettle - gt_obs_kettle)
        reward = reward_ee # + reward_kettle
        # KIV: Investigating different parameters, magnitude across timesteps
        # Robosuite: r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult
        if self.real_step:
            self.step_num += 1

        info['rwd_sparse'] = bool(reward >= -0.1)
        info['rwd_dense'] = reward
        info['rwd_ee'] = reward_ee
        info['rwd_kettle'] = reward_kettle
        info['solved'] = bool(reward >= -0.1)
        return state, reward, done, info
    
    def reset(self):
        if self.real_step:
            self.env.reset()
            self.step_num = 0
            reset_state = dict(qpos=self.init_qpos, qvel=self.init_qvel)
            self.set_env_state(reset_state)
            state = self.get_env_state()
            assert np.allclose(state['qpos'], self.init_qpos)
            assert np.allclose(state['qvel'], self.init_qvel)
            # print('saving reset img', os.getcwd())
            # img = self.render_extra_views()['camera_12_rgb']
            # Image.fromarray(img).save('env_reset_img0.png')
            # input()
        observation = self.get_obs()

        return observation
    
    def get_env_state(self):
        # https://github.com/oliviaylee/vip/blob/0aadeab2736324eda45de0e8df96dbb59e608145/evaluation/mj_envs/mj_envs/envs/env_base.py#L373
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        act = self.sim.data.act.ravel().copy() if self.sim.model.na>0 else None
        mocap_pos = self.sim.data.mocap_pos.copy() if self.sim.model.nmocap>0 else None
        mocap_quat = self.sim.data.mocap_quat.copy() if self.sim.model.nmocap>0 else None
        site_pos = self.sim.model.site_pos[:].copy() if self.sim.model.nsite>0 else None
        site_quat = self.sim.model.site_quat[:].copy() if self.sim.model.nsite>0 else None
        body_pos = self.sim.model.body_pos[:].copy()
        body_quat = self.sim.model.body_quat[:].copy()
        return dict(qpos=qp,
                    qvel=qv,
                    act=act,
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                    site_pos=site_pos,
                    site_quat=site_quat,
                    body_pos=body_pos,
                    body_quat=body_quat)
    
    def set_env_state(self, state_dict): # set initial qpos
        qp = state_dict['qpos'] # robot + object
        qv = state_dict['qvel']
        # act = state_dict['act']
        self.set_state(qp, qv, None)
        # if self.sim.model.nmocap>0:
        #     self.sim.data.mocap_pos[:] = state_dict['mocap_pos']
        #     self.sim.data.mocap_quat[:] = state_dict['mocap_quat']
        # if self.sim.model.nsite>0:
        #     self.sim.model.site_pos[:] = state_dict['site_pos']
        #     self.sim.model.site_quat[:] = state_dict['site_quat']
        # self.sim.model.body_pos[:] = state_dict['body_pos']
        # self.sim.model.body_quat[:] = state_dict['body_quat']
        self.sim.forward()

    def set_state(self, qpos=None, qvel=None, act=None):
        """
        Set MuJoCo sim state
        """
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        state = self.sim.get_state()
        for i in range(self.sim.model.nq):
            state[i] = qpos[i]
        for i in range(self.sim.model.nv):
            state[self.sim.model.nq + i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()

    def add_cameras(self, camera_ids=None):
        self.cameras = dict()
        for camera_id in camera_ids:
            camera = engine.MovableCamera(self.sim,
                                          height=64,
                                          width=64)
            camera.set_pose(**CAMERAS[camera_id])
            self.cameras['camera_{}'.format(camera_id)] = camera

        self.cameras['camera_gripper'] = engine.Camera(
            self.sim,
            height=64,
            width=64,
            camera_id='gripper_camera_rgb')

    def render_extra_views(self, mode='rgb', depth=False, segmentation=False):
        imgs = {}
        if 'rgb' in mode:
            # http://www.mujoco.org/book/APIreference.html#mjvOption
            # https://github.com/deepmind/dm_control/blob/9e0fe0f0f9713a2a993ca78776529011d6c5fbeb/dm_control/mujoco/engine.py#L200
            # mjtRndFlag(mjRND_SHADOW=0, mjRND_WIREFRAME=1, mjRND_REFLECTION=2, mjRND_ADDITIVE=3, mjRND_SKYBOX=4, mjRND_FOG=5, mjRND_HAZE=6, mjRND_SEGMENT=7, mjRND_IDCOLOR=8, mjNRNDFLAG=9)

            for camera_id, camera in self.cameras.items():
                img_rgb = camera.render(render_flag_overrides=dict(
                    skybox=False, fog=False, haze=False))
                imgs[camera_id + "_rgb"] = img_rgb

        if 'depth' in mode:
            for camera_id, camera in self.cameras.items():
                img_depth = camera.render(depth=True, segmentation=False)
                imgs[camera_id + "_depth"] = np.clip(img_depth, 0.0, 4.0)

        if 'human' in mode:
            self.renderer.render_to_window()  # adept_envs.mujoco_env.MujocoEnv.render

        return imgs


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env, embedding_name=None, device='cuda', load_path="", checkpoint="",
    proprio=0,camera_name=None, env_name=None, pixel_based=True, embedding_reward=False,
      goal_timestep=49, init_timestep=0):
        gym.ObservationWrapper.__init__(self, env)

        self.env_name = env_name 
        self.cameras = [camera_name]
        self.camera_name = self.cameras[0]

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False

        if "vip" in load_path:
            print(f"Loading pre-trained {load_path} model!")
            from vip import load_vip 
            rep = load_vip()
            rep.eval()
            embedding_dim = rep.module.hidden_dim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255
        elif "r3m" in load_path:
            print(f"Loading pre-trained {load_path} model!")
            from r3m import load_r3m_reproduce
            rep = load_r3m_reproduce(load_path)
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255        
        elif load_path == "clip":
            import clip
            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif (load_path == "random") or (load_path == "resnet"):
                embedding, embedding_dim = _get_embedding(load_path=load_path)
                self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            raise NameError("Invalid Model")
        embedding.eval()

        if device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.pixel_based = pixel_based
        self.embedding_reward = embedding_reward 
        self.init_state = None
        if self.pixel_based:
            self.observation_space = Box(
                        low=-np.inf, high=np.inf, shape=(self.embedding_dim+self.proprio,))
        else:
            self.observation_space = self.env.unwrapped.observation_space

        if self.embedding_reward: 
            self.init_timestep = init_timestep
            self.goal_timestep = goal_timestep
            
            # evaluation information
            from trajopt import DEMO_PATHS
            demopath = DEMO_PATHS[self.env_name] 
            demo_id = demopath[-1] 
            traj_path = demopath[:-1] + f'traj_{demo_id}.pickle'
            demo = pickle.load(open(traj_path, 'rb'))
            self.goal_robot_pose = demo.sol_info[self.goal_timestep]['obs_dict']['robot_jnt']
            self.goal_object_pose = demo.sol_info[self.goal_timestep]['obs_dict']['objs_jnt']
            self.goal_end_effector = demo.sol_info[self.goal_timestep]['obs_dict']['end_effector']
 
            self.goal_embedding = {} 
            for camera in self.cameras:
                
                # mj_envs MPPI demo for goal embedding 
                if init_timestep != 0:
                    self.init_state = {}
                    for key in demo.sol_state[init_timestep]:
                        self.init_state[key] = demo.sol_state[init_timestep][key]
                    self.init_state['env_timestep'] = init_timestep + 1 

                video_paths = [demopath + f'/{camera}']
                num_vid = len(video_paths)
                end_goals = [] 
                for i in range(num_vid):
                    vid = f"{video_paths[i]}"
                    img = Image.open(f"{vid}/{self.goal_timestep}.png")
                    cur_dir = os.getcwd() 
                    img.save(f"{cur_dir}/goal_image_{camera}.png") # save goal image
                    end_goals.append(img)
                
                # hack to get when there is only one goal image working
                if len(end_goals) == 1:
                    end_goals.append(end_goals[-1])
                
                goal_embedding = self.encode_batch(end_goals)
                self.goal_embedding[camera] = goal_embedding.mean(axis=0) 

    def observation(self, observation):
        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None and len(observation.shape) > 1:
            if isinstance(observation, np.ndarray):
                o = Image.fromarray(observation.astype(np.uint8))

            inp = self.transforms(o).reshape(-1, 3, 224, 224)
            if  "vip" in self.load_path or "r3m" in self.load_path:
                inp *= 255.0
            inp = inp.to(self.device)

            with torch.no_grad():                
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()

            ## IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[:self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[:self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def encode_batch(self, obs, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp = []
        for o in obs:
            if isinstance(o, np.ndarray):
                o = Image.fromarray(o.astype(np.uint8))
            o = self.transforms(o).reshape(-1, 3, 224, 224)
            if "vip" in self.load_path or "r3m" in self.load_path:
                o *= 255.0
            inp.append(o)

        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None and self.pixel_based:
            return self.observation(self.env.observation(None))
        else:
            return self.env.unwrapped.get_obs()
    def get_views(self, embedding=False):
        views = {}
        embeddings = {}
        for camera in self.cameras:
            view = self.env.get_image(camera_name=camera)
            views[camera] = view
            if embedding:
                embeddings[camera] = self.observation(view)
        if embedding:
            return embeddings 
        return views  

    def start_finetuning(self):
        self.start_finetune = True
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action) 
        obs_embedding = self.observation(observation)
        info['obs_embedding'] = obs_embedding 
        if self.embedding_reward:
            rewards = []
            # Note: only single camera evaluation is supported 
            for camera in self.cameras:
                img_camera = self.env.get_image(camera_name=camera)
                obs_embedding_camera = self.observation(img_camera)
                obs_embedding_camera = obs_embedding_camera if self.proprio == 0 else obs_embedding_camera[:-self.proprio]
                reward_camera = -np.linalg.norm(obs_embedding_camera-self.goal_embedding[camera])
                rewards.append(reward_camera) 
            # some state-based info for evaluating learned reward func.
            if 'end_effector' in info['obs_dict']:
                info['obs_dict']['ee_error'] = np.linalg.norm(self.goal_end_effector-info['obs_dict']['end_effector'])
            if 'hand_jnt' in info['obs_dict']:
                info['obs_dict']['robot_error'] = np.linalg.norm(self.goal_robot_pose-info['obs_dict']['hand_jnt'])
            elif 'robot_jnt' in info['obs_dict']:
                info['obs_dict']['robot_error'] = np.linalg.norm(self.goal_robot_pose-info['obs_dict']['robot_jnt'])
            if 'objs_jnt' in info['obs_dict']:
                info['obs_dict']['objs_error'] = np.linalg.norm(self.goal_object_pose-info['obs_dict']['objs_jnt'])
            
            reward = min(rewards)
        if not self.pixel_based:
            state = self.env.unwrapped.get_obs()
        else: 
            state = obs_embedding 

        return state, reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        try:
            if self.init_state is not None:
                self.env.set_env_state(self.init_state)
        except Exception as e:
            print("Resetting Initial State Error")
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
        if not self.pixel_based:
            observation = self.env.unwrapped.get_obs()
        else:
            observation = self.observation(observation) # This is needed for IL, but would it break other evaluations?
        return observation

class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

    def get_image(self, camera_name=None):
        if camera_name is None:
            camera_name = self.camera_name
        if camera_name == "default" or camera_name == "all":
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
             device_id=self.device_id)
        else:
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                            camera_name=camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return img

    def observation(self, observation=None):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()
