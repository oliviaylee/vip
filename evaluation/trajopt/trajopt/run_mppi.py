"""
This is a launcher script for launching mjrl training using hydra
"""
import numpy as np
import os
import time as timer
import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle 
from tqdm import tqdm 
import multiprocessing as mp
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip
import skvideo.io
import os 
import wandb
from PIL import Image 
 
from mj_envs.envs.env_variants import register_env_variant
from trajopt.envs.obs_wrappers import env_constructor
from trajopt.algos.mppi import MPPI
from trajopt import DEMO_PATHS


@hydra.main(config_name="mppi_config", config_path="config")
def configure_jobs(job_data):
    OUT_DIR = '.'
    PICKLE_FILE = OUT_DIR + '/trajectories.pickle'

    assert job_data.embedding_reward ==  True
    assert job_data.env_kwargs.embedding_reward == True 

    job_data.env_kwargs.load_path = job_data.embedding
    job_data.job_name = job_data.embedding
    reward_type = f"{job_data.embedding}" if job_data.embedding_reward else "true"
    job_data.job_name = f"{job_data.env}-{reward_type}-{job_data.camera}-{job_data.env_kwargs.init_timestep}-{job_data.env_kwargs.goal_timestep}-seed{job_data.seed}"
    
    with open('job_config.yaml', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)
    
    if 'env_hyper_params' in job_data.keys():
        job_data.env = register_env_variant(job_data.env, job_data.env_hyper_params)

    # Construct environment 
    demo0 = '/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_kettle_bottomknob_switch_slide/20230528T010656-1be74c034d6940f1a2d9e63d24fc7f83-218.npz'
    demo1 = '/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_kettle_switch_hinge_slide/20230528T011924-3c028631c2ed4843ab817c0e011443d7-218.npz'
    demo2 = '/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_microwave_bottomknob_hinge_slide/20230528T020316-0f04a151a19248efa3548473221e7553-243.npz'
    demo3 = '/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_microwave_bottomknob_switch_slide/20230528T021912-0b87c52551594f0b93f615d1d83d31a7-212.npz'
    demo4 = '/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_topknob_bottomknob_hinge_slide/20230528T022647-0bc6fd9c1fad4716bf81a39c5de9f566-240.npz'
    demo5 = '/iris/u/oliviayl/repos/affordance-learning/d5rl/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz/kitchen_demos_multitask_npz/friday_topknob_bottomknob_switch_slide/20230528T024135-1e3f7a3a772b4f128eccb9a6d91bc7e5-182.npz'
    env_kwargs = job_data['env_kwargs']
    env = env_constructor(job_data['H_total'], job_data['plan_horizon'], **env_kwargs)

    # envs = []
    # for _ in range(job_data['num_cpu']):
    #     env_kwargs = job_data['env_kwargs']
    #     env = env_constructor(**env_kwargs)
    #     envs.append(env)

    mean = np.zeros(env.action_dim)
    sigma = 1.0*np.ones(env.action_dim)
    filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]
    trajectories = []  # TrajOpt format (list of trajectory classes)

    actual_trajectory = []
    # Generate trajectories and plot embedding distances
    for i in range(job_data['num_traj']):
        os.makedirs(f"./{i}", exist_ok=True)
        os.mkdir(f"./{i}/logs")
        start_time = timer.time()
        print("Currently optimizing trajectory : %i" % i)
        seed = job_data['seed'] + i*12345
        env.reset(seed=seed) # env reset for every env in envs

        agent = MPPI(env,
                    H=job_data['plan_horizon'],
                    paths_per_cpu=job_data['paths_per_cpu'],
                    num_cpu=job_data['num_cpu'],
                    kappa=job_data['kappa'],
                    gamma=job_data['gamma'],
                    mean=mean,
                    filter_coefs=filter_coefs,
                    default_act=job_data['default_act'],
                    seed=seed,
                    env_kwargs=env_kwargs) # pass in envs

        # trajectory optimization
        # distances = {}
        # for camera in agent.env.env.cameras:
        #     distances[camera] = []
        #     goal_embedding = agent.env.env.goal_embedding[camera]
        #     distance = np.linalg.norm(agent.sol_embedding[-1][camera]-goal_embedding)
        #     distances[camera].append(distance)

        for i in tqdm(range(job_data['H_total'])):
            # take one-step with trajectory optimization
            agent.train_step(job_data['num_iter'])
            step_info = agent.sol_info[-1]
            step_log = {'t':step_info['obs_dict']['t'],
            'rwd_sparse': step_info['rwd_sparse'],
            'rwd_dense': step_info['rwd_dense'],
            'rwd_ee': step_info['rwd_ee'],
            'rwd_kettle': step_info['rwd_kettle'],
            'solved': step_info['solved'] * 1.0}
            # 'ee_error':step_info['obs_dict']['ee_error'],
            # 'robot_error':step_info['obs_dict']['robot_error'],
            # 'objs_error':step_info['obs_dict']['objs_error']}
            
            # Save embedding distance curve
            # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
            # for camera_id, camera in enumerate(agent.env.env.cameras):
            #     goal_embedding = agent.env.env.goal_embedding[camera]
            #     goal_distance = np.linalg.norm(agent.sol_embedding[-1][camera]-goal_embedding)
            #     distances[camera].append(goal_distance)
            #     ax[camera_id].plot(np.arange(len(distances[camera])), distances[camera])
            #     ax[camera_id].set_title(camera)
            #     step_log[camera] = goal_distance
            
            # for key in step_log:
            #     agent.logger.log_kv(key, step_log[key])

            # agent.logger.save_log(f'./{i}/logs')
            # plt.suptitle(f"{job_data.env} Video MPPI {job_data.embedding} Distance")
            # plt.savefig(f"{i}_{job_data.embedding}_embedding_distance.png")
            # plt.close() 

            # Save trajectory video: step through each action in agent.act_sequence, save out that image, write out the video
            env.real_env_step(False)
            env.reset()
            env.set_env_state(agent.sol_state[-2])
            imgs, cumulative_rew = [], 0
            # TO-DO: Calculate metrics
            for act in agent.act_sequence:
                img = env.env.render_extra_views()['camera_12_rgb'].copy()
                imgs.append(img)
                state, reward, done, info = env.step(act)
                cumulative_rew += reward
            Image.fromarray(imgs[0]).save(f"./{i}_init.png")
            imgs = [Image.fromarray(img) for img in imgs]
            imgs[0].save(f"./{i}_plan_{cumulative_rew}.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)

            actual_trajectory.append(imgs[0].copy())

            # for camera in agent.env.env.cameras:
            #     os.makedirs(f"./{i}/{camera}", exist_ok=True)
            #     frames = agent.animate_result_offscreen(camera_name=camera)
            #     VID_FILE = OUT_DIR + f'/{i}/{i}_{job_data.embedding}_{camera}' + '.gif'
            #     cl = ImageSequenceClip(frames, fps=20)
            #     cl.write_gif(VID_FILE, fps=20)
            #     frames = np.array(frames)
            #     for t2 in range(frames.shape[0]):
            #         img = frames[t2]
            #         result = Image.fromarray((img).astype(np.uint8))
            #         result.save(f"./{i}/{camera}/{t2}.png")
        
        actual_rew = np.sum(agent.sol_reward)
        actual_trajectory[0].save(
            f"./actual_traj_{actual_rew}.gif",
            save_all=True,
            append_images=actual_trajectory[1:],
            duration=100,
            loop=0)

        # Save trajectory
        SAVE_FILE = OUT_DIR + '/traj_%i.pickle' % i
        pickle.dump(agent, open(SAVE_FILE, 'wb'))
        
        end_time = timer.time()
        print("Trajectory reward = %f" % actual_rew)
        print("Optimization time for this trajectory = %f" % (end_time - start_time))

        last_state_distance, robot_distance_per_timestep_loss, kettle_distance_per_timestep_loss = env.env.calculate_metrics(agent.sol_state, agent.sol_info)
        print("Last state distance = %f" % last_state_distance)
        print("Sum of L2 losses at each timestep (robot end effector) = %f" % robot_distance_per_timestep_loss)
        print("Sum of L2 losses at each timestep (kettle) = %f" % kettle_distance_per_timestep_loss)

        trajectories.append(agent)
        pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))
    
if __name__ == "__main__":
    mp.set_start_method('fork') # 'spawn'
    configure_jobs()