import argparse
import os
import pickle
import shutil

from ..go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import torch
import numpy as np
import genesis as gs

from genesis.utils.terrain import parse_terrain
from genesis.utils.geom import transform_quat_by_quat

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        # "termination_if_roll_greater_than": 10,  # degree
        # "termination_if_pitch_greater_than": 10,
        "termination_if_roll_greater_than": 30,  # degree
        "termination_if_pitch_greater_than": 30,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            # "tracking_lin_vel": 1.0,
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 0.2,
            # "lin_vel_z": -1.0,
            # "base_height": -50.0,
            "base_height": -10.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

class Go2RoughEnv(Go2Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, edit_terrain=True, device="cuda"):
        self.dt = 0.02  # control frequence on real robot is 50hz
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        # add plane
        # n_subterrains=(2, 2)
        # subterrain_types=[
        #     ["pyramid_sloped_terrain", "wave_terrain"],
        #     ["stairs_terrain","pyramid_stairs_terrain"]
        #     ]
        n_subterrains=(1, 1)
        subterrain_types=[
            ["wave_terrain"]
            ]
        self.terrain = gs.morphs.Terrain(
            n_subterrains=n_subterrains,
            subterrain_types=subterrain_types,
        )
        _, _, self.terrain.height_field = parse_terrain(morph=self.terrain, surface=gs.surfaces.Default())
        self.scene.add_entity(self.terrain)
        print("height field is\n", self.terrain.height_field)

        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, edit_terrain, device)
        self.envs_origins = torch.zeros((self.num_envs, 7), device=self.device)

        # build
        # self.scene.build(n_envs=num_envs)

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        for index in envs_idx:
            x, y, z, q = self._ramdom_robot_position()
            self.envs_origins[index, 0] = x
            self.envs_origins[index, 1] = y
            self.envs_origins[index, 2] = z
            # self.envs_origins[index, 3:7] = q
            # self.base_pos[index, 0] = x # maybe not use
            # self.base_pos[index, 1] = y
            # self.base_pos[index, 2] = z
            self.base_quat[index] = transform_quat_by_quat(q, self.base_quat[index])
        # debug
        # print(f"envs_idx device: {envs_idx.device}")
        # print(f"self.base_pos device: {self.base_pos.device}")
        # print(f"self.envs_origins device: {self.envs_origins.device}")
        # self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_pos(self.base_pos[envs_idx]+self.envs_origins[envs_idx, :3], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        # self.robot.set_quat(self.envs_origins[envs_idx, 3:7], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def _ramdom_robot_position(self):
        # 1. Sample random row, col(a subterrain)
        # 0.775 ~ l2_norm(0.7, 0.31)
        go2_size_xy = 0.775
        row = np.random.randint(int((self.terrain.n_subterrains[0]*self.terrain.subterrain_size[0]-go2_size_xy)/self.terrain.horizontal_scale))
        col = np.random.randint(int((self.terrain.n_subterrains[1]*self.terrain.subterrain_size[1]-go2_size_xy)/self.terrain.horizontal_scale))
        # 2. Convert (row, col) -> (x, y) in world coords
        # Each cell is horizontal_scale in size
        x = row*self.terrain.horizontal_scale + go2_size_xy/2
        y = col*self.terrain.horizontal_scale + go2_size_xy/2
        # 3. Get terrain height in meters
        z = self.terrain.height_field[row, col]*self.terrain.vertical_scale
        # z = 0.5

        # 4. Add a small offset so the robot spawns above the ground
        # z += 0.1  # for example

        # 5. rotation quaternion
        angle = np.random.uniform(2*np.pi)
        q = torch.tensor([np.cos(angle), 0, 0, np.sin(angle)], device=self.device)
        
        return x, y, z, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Go2RoughEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
