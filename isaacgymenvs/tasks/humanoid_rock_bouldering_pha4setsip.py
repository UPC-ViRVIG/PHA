# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..

from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgymenvs.tasks.amp.humanoid_pmp_base import HumanoidPMPBase, dof_to_obs_mpl
from isaacgymenvs.tasks.amp.utils_amp.motion_lib_mpl import MotionLibMPL
from isaacgymenvs.tasks.amp.utils_amp.motion_lib_ip import MotionLibIP

from isaacgymenvs.utils.torch_jit_utils import (
    quat_mul,
    to_torch,
    calc_heading_quat_inv,
    quat_to_tan_norm,
    my_quat_rotate,
    quat_rotate,
    normalize,
    exp_map_to_quat,
    quat_conjugate,
)

# [root_h,  root_rot,   root_vel,   root_ang_vel,   dof_pos,    dof_vel,    key_body_pos,   key_body_sensors_global_space]
# [1,       6,          3,          3,              102,        72,         3 * 4,          0]
BASE_OBS_COUNT = 105
RIGHT_HAND_IP_OBS_COUNT = 87
LEFT_HAND_IP_OBS_COUNT = 87
RIGHT_HAND_MOREF_OBS_COUNT = 29
LEFT_HAND_MOREF_OBS_COUNT = 29
NUM_OBS_PER_STEP = BASE_OBS_COUNT + RIGHT_HAND_IP_OBS_COUNT - 13 + LEFT_HAND_IP_OBS_COUNT - 13 + 3 + 3 # Adding foot distances to target. Removing actions for each hand
NUM_PMP4SETSIP_OBS_PER_STEP = BASE_OBS_COUNT + RIGHT_HAND_IP_OBS_COUNT + LEFT_HAND_IP_OBS_COUNT + RIGHT_HAND_MOREF_OBS_COUNT + LEFT_HAND_MOREF_OBS_COUNT


class HumanoidRockBoulderingPHA4SetsIP(HumanoidPMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(
            self,
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
    ):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidRockBoulderingPHA4SetsIP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_pmp4setsip_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_pmp4setsip_obs_steps >= 2

        self._reset_ref_env_ids = []

        self.NUM_OBS = NUM_OBS_PER_STEP
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self._right_hand_0_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 1, :]
        self._left_hand_0_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 2, :]
        self._right_foot_0_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 3, :]
        self._left_foot_0_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 4, :]
        self._hand_1_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 5, :]
        self._foot_1_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 6, :]
        self._hand_2_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 7, :]
        self._foot_2_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 8, :]
        self._hand_3_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 9, :]
        self._foot_3_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 10, :]
        self._hand_4_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 11, :]
        self._foot_4_rock_root_states = self._root_states.view(
            self.num_envs, self.num_actors_per_env, self._root_states.shape[-1]
        )[..., 12, :]

        self._right_hand_0_rock_actor_ids = self._humanoid_actor_ids + 1
        self._left_hand_0_rock_actor_ids = self._humanoid_actor_ids + 2
        self._right_foot_0_rock_actor_ids = self._humanoid_actor_ids + 3
        self._left_foot_0_rock_actor_ids = self._humanoid_actor_ids + 4
        self._hand_1_rock_actor_ids = self._humanoid_actor_ids + 5
        self._foot_1_rock_actor_ids = self._humanoid_actor_ids + 6
        self._hand_2_rock_actor_ids = self._humanoid_actor_ids + 7
        self._foot_2_rock_actor_ids = self._humanoid_actor_ids + 8
        self._hand_3_rock_actor_ids = self._humanoid_actor_ids + 9
        self._foot_3_rock_actor_ids = self._humanoid_actor_ids + 10
        self._hand_4_rock_actor_ids = self._humanoid_actor_ids + 11
        self._foot_4_rock_actor_ids = self._humanoid_actor_ids + 12

        motion_file = cfg["env"].get("motion_file", "Running_AMP_MPL.npy")
        motion_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../assets/amp/motions/" + motion_file,
        )
        self._load_motion(motion_file_path)

        hands_ip = cfg["env"].get("hands_ip", "hand_grasping_demonstrations.npy")
        hands_ip_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../assets/amp/motions/" + hands_ip,
        )
        self._load_hands_interactive_prior(hands_ip_path)

        self.num_pmp4setsip_obs = (
                self._num_pmp4setsip_obs_steps * NUM_PMP4SETSIP_OBS_PER_STEP
        )

        self._pmp4setsip_obs_space = spaces.Box(
            np.ones(self.num_pmp4setsip_obs) * -np.Inf,
            np.ones(self.num_pmp4setsip_obs) * np.Inf,
        )

        self._pmp4setsip_obs_buf = torch.zeros(
            (
                self.num_envs,
                self._num_pmp4setsip_obs_steps,
                NUM_PMP4SETSIP_OBS_PER_STEP,
            ),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_pmp4setsip_obs_buf = self._pmp4setsip_obs_buf[:, 0]
        self._hist_pmp4setsip_obs_buf = self._pmp4setsip_obs_buf[:, 1:]

        self._pmp4setsip_obs_demo_buf = None
        self.actions = torch.zeros(
            [self.num_envs, self.humanoid_num_actuators], device=self.device
        )
        self.motion_ids = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )

        self.init_rfingers_rot = (
            torch.FloatTensor(self.cfg["misc"]["init_rfingers_rot"])[None].to(self.device).repeat(self.num_envs, 1, 1)
        )
        self.init_lfingers_rot = (
            torch.FloatTensor(self.cfg["misc"]["init_lfingers_rot"])[None].to(self.device).repeat(self.num_envs, 1, 1)
        )
        self.unit_x = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.unit_x[:, 0] = 1

        self.unit_y = torch.zeros_like(self.unit_x)
        self.unit_y[:, 1] = 1

        self.unit_z = torch.zeros_like(self.unit_x)
        self.unit_z[:, 2] = 1

        self.init_rpalm_rot = (
            torch.FloatTensor([[0.7071068, 0, 0, 0.7071068]]).to(self.device).repeat(self.num_envs, 1)
        )
        self.init_lpalm_rot = (
            torch.FloatTensor([[0, 0.7071068, 0.7071068, 0]]).to(self.device).repeat(self.num_envs, 1)
        )

        self.default_rfingers_facing_dir = torch.stack([-self.unit_z] * 4 + [self.unit_y] * 16, dim=1)
        self.default_lfingers_facing_dir = torch.stack([-self.unit_z] * 4 + [-self.unit_y] * 16, dim=1)

        self.right_hand_goal_pos = torch.zeros_like(self._right_hand_0_rock_root_states[:, 0:3])
        self.right_hand_goal_rot = torch.zeros_like(self._right_hand_0_rock_root_states[:, 3:7])
        self.left_hand_goal_pos = torch.zeros_like(self._left_hand_0_rock_root_states[:, 0:3])
        self.left_hand_goal_rot = torch.zeros_like(self._left_hand_0_rock_root_states[:, 3:7])
        self.right_foot_goal_pos = torch.zeros_like(self._right_foot_0_rock_root_states[:, 0:3])
        self.right_foot_goal_rot = torch.zeros_like(self._right_foot_0_rock_root_states[:, 3:7])
        self.left_foot_goal_pos = torch.zeros_like(self._left_foot_0_rock_root_states[:, 0:3])
        self.left_foot_goal_rot = torch.zeros_like(self._left_foot_0_rock_root_states[:, 3:7])

        return

    def update_rock_goal_pos(self):
        right_hand_is_higher = self._right_hand_0_rock_root_states[:, 2] >= self._left_hand_0_rock_root_states[:, 2]
        right_hand_is_lower = right_hand_is_higher == False

        mask_update_0 = (self.progress_buf < 30)
        mask_update_1_0 = (self.progress_buf >= 30) & (self.progress_buf < 60) & right_hand_is_higher
        mask_update_1_1 = (self.progress_buf >= 30) & (self.progress_buf < 60) & right_hand_is_lower
        mask_update_2_0 = (self.progress_buf >= 60) & (self.progress_buf < 90) & right_hand_is_higher
        mask_update_2_1 = (self.progress_buf >= 60) & (self.progress_buf < 90) & right_hand_is_lower
        mask_update_3_0 = (self.progress_buf >= 90) & (self.progress_buf < 120) & right_hand_is_higher
        mask_update_3_1 = (self.progress_buf >= 90) & (self.progress_buf < 120) & right_hand_is_lower
        mask_update_4_0 = (self.progress_buf >= 120) & right_hand_is_higher
        mask_update_4_1 = (self.progress_buf >= 120) & right_hand_is_lower

        self.right_hand_goal_pos[mask_update_1_0] = self._right_hand_0_rock_root_states[mask_update_1_0, 0:3].clone()
        self.left_hand_goal_pos[mask_update_1_0] = self._hand_1_rock_root_states[mask_update_1_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_1_0] = self._foot_1_rock_root_states[mask_update_1_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_1_0, 2] += 0.05
        self.left_foot_goal_pos[mask_update_1_0] = self._left_foot_0_rock_root_states[mask_update_1_0, 0:3].clone()
        self.left_foot_goal_pos[mask_update_1_0, 2] += 0.05
        self.right_hand_goal_pos[mask_update_1_1] = self._hand_1_rock_root_states[mask_update_1_1, 0:3].clone()
        self.left_hand_goal_pos[mask_update_1_1] = self._left_hand_0_rock_root_states[mask_update_1_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_1_1] = self._right_foot_0_rock_root_states[mask_update_1_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_1_1, 2] += 0.05
        self.left_foot_goal_pos[mask_update_1_1] = self._foot_1_rock_root_states[mask_update_1_1, 0:3].clone()
        self.left_foot_goal_pos[mask_update_1_1, 2] += 0.05

        self.right_hand_goal_pos[mask_update_2_0] = self._hand_2_rock_root_states[mask_update_2_0, 0:3].clone()
        self.left_hand_goal_pos[mask_update_2_0] = self._hand_1_rock_root_states[mask_update_2_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_2_0] = self._foot_1_rock_root_states[mask_update_2_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_2_0, 2] += 0.05
        self.left_foot_goal_pos[mask_update_2_0] = self._foot_2_rock_root_states[mask_update_2_0, 0:3].clone()
        self.left_foot_goal_pos[mask_update_2_0, 2] += 0.05
        self.right_hand_goal_pos[mask_update_2_1] = self._hand_1_rock_root_states[mask_update_2_1, 0:3].clone()
        self.left_hand_goal_pos[mask_update_2_1] = self._hand_2_rock_root_states[mask_update_2_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_2_1] = self._foot_2_rock_root_states[mask_update_2_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_2_1, 2] += 0.05
        self.left_foot_goal_pos[mask_update_2_1] = self._foot_1_rock_root_states[mask_update_2_1, 0:3].clone()
        self.left_foot_goal_pos[mask_update_2_1, 2] += 0.05

        self.right_hand_goal_pos[mask_update_3_0] = self._hand_2_rock_root_states[mask_update_3_0, 0:3].clone()
        self.left_hand_goal_pos[mask_update_3_0] = self._hand_3_rock_root_states[mask_update_3_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_3_0] = self._foot_3_rock_root_states[mask_update_3_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_3_0, 2] += 0.05
        self.left_foot_goal_pos[mask_update_3_0] = self._foot_2_rock_root_states[mask_update_3_0, 0:3].clone()
        self.left_foot_goal_pos[mask_update_3_0, 2] += 0.05
        self.right_hand_goal_pos[mask_update_3_1] = self._hand_3_rock_root_states[mask_update_3_1, 0:3].clone()
        self.left_hand_goal_pos[mask_update_3_1] = self._hand_2_rock_root_states[mask_update_3_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_3_1] = self._foot_2_rock_root_states[mask_update_3_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_3_1, 2] += 0.05
        self.left_foot_goal_pos[mask_update_3_1] = self._foot_3_rock_root_states[mask_update_3_1, 0:3].clone()
        self.left_foot_goal_pos[mask_update_3_1, 2] += 0.05

        self.right_hand_goal_pos[mask_update_4_0] = self._hand_4_rock_root_states[mask_update_4_0, 0:3].clone()
        self.left_hand_goal_pos[mask_update_4_0] = self._hand_3_rock_root_states[mask_update_4_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_4_0] = self._foot_3_rock_root_states[mask_update_4_0, 0:3].clone()
        self.right_foot_goal_pos[mask_update_4_0, 2] += 0.05
        self.left_foot_goal_pos[mask_update_4_0] = self._foot_4_rock_root_states[mask_update_4_0, 0:3].clone()
        self.left_foot_goal_pos[mask_update_4_0, 2] += 0.05
        self.right_hand_goal_pos[mask_update_4_1] = self._hand_3_rock_root_states[mask_update_4_1, 0:3].clone()
        self.left_hand_goal_pos[mask_update_4_1] = self._hand_4_rock_root_states[mask_update_4_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_4_1] = self._foot_4_rock_root_states[mask_update_4_1, 0:3].clone()
        self.right_foot_goal_pos[mask_update_4_1, 2] += 0.05
        self.left_foot_goal_pos[mask_update_4_1] = self._foot_3_rock_root_states[mask_update_4_1, 0:3].clone()
        self.left_foot_goal_pos[mask_update_4_1, 2] += 0.05

    def pre_physics_step(self, actions):
        # self.actions[:] = actions.to(self.device).clone()
        self.actions[:,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 36, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52]] = actions.to(self.device).clone()
        # Right MCP
        self.actions[:, [37, 38, 40]] = self.actions[:, 36, None].clone()
        # Left MCP
        self.actions[:, [50, 51, 53]] = self.actions[:, 49, None].clone()

        if self._pd_control:
            if not self.kinematics_mode:
                self.cur_targets[:, self.actuated_dof_indices] = self._action_to_pd_targets(
                    self.actions
                )
                self.gym.set_dof_position_target_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.cur_targets)
                )
            else:
                motion_ids = np.zeros(self.num_envs).astype(np.int64)
                longest_time = self._motion_lib.get_motion_length(0)
                motion_times = (self.progress_buf.detach().cpu().numpy() / 30.0) % longest_time

                (
                    root_pos,
                    root_rot,
                    dof_pos,
                    root_vel,
                    root_ang_vel,
                    dof_vel,
                    key_pos,
                ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

                root_pos, root_rot, root_vel, root_ang_vel = (
                    root_pos[0],
                    root_rot[0],
                    root_vel[0],
                    root_ang_vel[0],
                )
                # scale dof_pos
                dof_pos_norm = (dof_pos - self._pd_action_offset[None]) / self._pd_action_scale[None]
                dof_pos_norm = torch.clamp(dof_pos_norm, -0.8, 0.8)
                dof_pos = dof_pos_norm * self._pd_action_scale[None] + self._pd_action_offset[None]

                num_actors = int(self._root_states.shape[0] / self.num_envs)

                kinematic_dof_state = torch.stack([dof_pos.flatten(), dof_vel.flatten()], dim=-1)
                self._dof_pos[:] = dof_pos
                self._dof_vel[:] = dof_vel
                global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32, device=self.device)
                multi_env_ids_int32 = global_indices.view(self.num_envs, -1)[:, :1].flatten().contiguous()
                self.gym.set_dof_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(kinematic_dof_state),
                    gymtorch.unwrap_tensor(multi_env_ids_int32),
                    len(multi_env_ids_int32),
                )

                kinematic_root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)

                if num_actors > 1:
                    temp = self._root_states.detach().clone().view(self.num_envs, -1, 13)
                    temp[:, 0] = kinematic_root_states
                    kinematic_root_states = temp.view(-1, 13).contiguous()
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(kinematic_root_states),
                    gymtorch.unwrap_tensor(multi_env_ids_int32),
                    len(multi_env_ids_int32),
                )
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_pmp4setsip_obs()
        self._compute_pmp4setsip_observations()

        pmp4setsip_obs_flat = self._pmp4setsip_obs_buf.view(
            -1, self.num_pmp4setsip_obs
        )
        self.extras["pmp4setsip_obs"] = pmp4setsip_obs_flat
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            right_wrist_pos = self._rigid_body_pos[:, self.hand_parts_right_handles[0], :]  # 0 is right_palm
            right_wrist_rot = self._rigid_body_rot[:, self.hand_parts_right_handles[0], :]
            right_palm_pos = right_wrist_pos + my_quat_rotate(
                right_wrist_rot, self.rpalm_rigid_body_com
            )
            left_wrist_pos = self._rigid_body_pos[:, self.hand_parts_left_handles[0], :]  # 0 is left_palm
            left_wrist_rot = self._rigid_body_rot[:, self.hand_parts_left_handles[0], :]
            left_palm_pos = left_wrist_pos + my_quat_rotate(
                left_wrist_rot, self.lpalm_rigid_body_com
            )
            right_foot_rot = self._rigid_body_rot[:, self.foot_right_handles[0], :]
            right_foot_pos = self._rigid_body_pos[:, self.foot_right_handles[0], :] + my_quat_rotate(
                right_foot_rot, self.rfoot_rigid_body_com
            )
            left_foot_rot = self._rigid_body_rot[:, self.foot_left_handles[0], :]
            left_foot_pos = self._rigid_body_pos[:, self.foot_left_handles[0], :] + my_quat_rotate(
                left_foot_rot, self.lfoot_rigid_body_com
            )

            right_hand_goal_pos = self.right_hand_goal_pos[0]
            left_hand_goal_pos = self.left_hand_goal_pos[0]
            right_foot_goal_pos = self.right_foot_goal_pos[0]
            left_foot_goal_pos = self.left_foot_goal_pos[0]

            right_hand_start_line = right_palm_pos[0].cpu().numpy()
            right_hand_end_line = right_hand_goal_pos.cpu().numpy()
            right_hand_lines = [right_hand_start_line, right_hand_end_line]

            left_hand_start_line = left_palm_pos[0].cpu().numpy()
            left_hand_end_line = left_hand_goal_pos.cpu().numpy()
            left_hand_lines = [left_hand_start_line, left_hand_end_line]

            right_foot_start_line = right_foot_pos[0].cpu().numpy()
            right_foot_end_line = right_foot_goal_pos.cpu().numpy()
            right_foot_lines = [right_foot_start_line, right_foot_end_line]

            left_foot_start_line = left_foot_pos[0].cpu().numpy()
            left_foot_end_line = left_foot_goal_pos.cpu().numpy()
            left_foot_lines = [left_foot_start_line, left_foot_end_line]

            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                1,  # 1
                right_hand_lines,
                [0.0, 1.0, 1.0] * 1,
            )

            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                1,  # 1
                left_hand_lines,
                [1.0, 1.0, 0.0] * 1,
            )

            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                1,  # 1
                right_foot_lines,
                [0.0, 1.0, 1.0] * 1,
            )

            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                1,  # 1
                left_foot_lines,
                [1.0, 1.0, 0.0] * 1,
            )

        return

    def _reset_rocks(self, env_ids):
        if env_ids is None:
            return
        # Update rock wall positions
        rock_right_wrist_pos = torch.zeros_like(self._rigid_body_pos[env_ids, self.hand_parts_right_handles[0], :])
        rock_left_wrist_pos = torch.zeros_like(self._rigid_body_pos[env_ids, self.hand_parts_right_handles[0], :])
        rock_right_foot_pos = torch.zeros_like(self._rigid_body_pos[env_ids, self.hand_parts_right_handles[0], :])
        rock_left_foot_pos = torch.zeros_like(self._rigid_body_pos[env_ids, self.hand_parts_right_handles[0], :])

        rock_right_wrist_pos[:, 0] = 0.3
        rock_left_wrist_pos[:, 0] = 0.3
        rock_right_foot_pos[:, 0] = 0.3
        rock_left_foot_pos[:, 0] = 0.3
        rock_right_wrist_pos[:, 1] = -0.40
        rock_left_wrist_pos[:, 1] = 0.0
        rock_right_foot_pos[:, 1] = -0.4
        rock_left_foot_pos[:, 1] = 0.0

        higher_randomness = torch.rand(rock_right_wrist_pos.shape[0], device=self.device)
        right_hand_is_higher = higher_randomness > 0.5
        right_hand_is_lower = higher_randomness <= 0.5

        rock_right_wrist_pos[right_hand_is_higher, 2] = 2.60
        rock_right_wrist_pos[right_hand_is_lower, 2] = 2.30
        rock_left_wrist_pos[right_hand_is_lower, 2] = 2.60
        rock_left_wrist_pos[right_hand_is_higher, 2] = 2.30
        rock_right_foot_pos[right_hand_is_lower, 2] = 1.70
        rock_right_foot_pos[right_hand_is_higher, 2] = 1.40
        rock_left_foot_pos[right_hand_is_higher, 2] = 1.70
        rock_left_foot_pos[right_hand_is_lower, 2] = 1.40

        self._right_hand_0_rock_root_states[env_ids, 0:3] = rock_right_wrist_pos
        self._left_hand_0_rock_root_states[env_ids, 0:3] = rock_left_wrist_pos
        self._right_foot_0_rock_root_states[env_ids, 0:3] = rock_right_foot_pos
        self._left_foot_0_rock_root_states[env_ids, 0:3] = rock_left_foot_pos

        rock_wrist_1_pos = rock_right_wrist_pos.clone()
        rock_wrist_1_pos[right_hand_is_higher] = rock_left_wrist_pos[right_hand_is_higher].clone()
        rock_wrist_1_pos[:, 2] += 0.60
        rock_foot_1_pos = rock_left_foot_pos.clone()
        rock_foot_1_pos[right_hand_is_higher] = rock_right_foot_pos[right_hand_is_higher].clone()
        rock_foot_1_pos[:, 2] += 0.60
        self._hand_1_rock_root_states[env_ids, 0:3] = rock_wrist_1_pos
        self._foot_1_rock_root_states[env_ids, 0:3] = rock_foot_1_pos

        rock_wrist_2_pos = rock_left_wrist_pos.clone()
        rock_wrist_2_pos[right_hand_is_higher] = rock_right_wrist_pos[right_hand_is_higher].clone()
        rock_wrist_2_pos[:, 2] += 0.60
        rock_foot_2_pos = rock_right_foot_pos.clone()
        rock_foot_2_pos[right_hand_is_higher] = rock_left_foot_pos[right_hand_is_higher].clone()
        rock_foot_2_pos[:, 2] += 0.60
        self._hand_2_rock_root_states[env_ids, 0:3] = rock_wrist_2_pos
        self._foot_2_rock_root_states[env_ids, 0:3] = rock_foot_2_pos

        rock_wrist_3_pos = rock_wrist_1_pos.clone()
        rock_wrist_3_pos[:, 2] += 0.60
        rock_foot_3_pos = rock_foot_1_pos.clone()
        rock_foot_3_pos[:, 2] += 0.60
        self._hand_3_rock_root_states[env_ids, 0:3] = rock_wrist_3_pos
        self._foot_3_rock_root_states[env_ids, 0:3] = rock_foot_3_pos

        rock_wrist_4_pos = rock_wrist_2_pos.clone()
        rock_wrist_4_pos[:, 2] += 0.60
        rock_foot_4_pos = rock_foot_2_pos.clone()
        rock_foot_4_pos[:, 2] += 0.60
        self._hand_4_rock_root_states[env_ids, 0:3] = rock_wrist_4_pos
        self._foot_4_rock_root_states[env_ids, 0:3] = rock_foot_4_pos

        self.right_hand_goal_pos[env_ids] = self._right_hand_0_rock_root_states[env_ids, 0:3].clone()
        self.left_hand_goal_pos[env_ids] = self._left_hand_0_rock_root_states[env_ids, 0:3].clone()
        self.right_hand_goal_rot[env_ids] = self._right_hand_0_rock_root_states[env_ids,
                                            3:7].clone()  # Good for now since rocks don't rotate
        self.left_hand_goal_rot[env_ids] = self._left_hand_0_rock_root_states[env_ids,
                                           3:7].clone()  # Good for now since rocks don't rotate
        self.right_foot_goal_pos[env_ids] = self._right_foot_0_rock_root_states[env_ids, 0:3].clone()
        self.right_foot_goal_pos[env_ids, 2] += 0.05
        self.right_foot_goal_rot[env_ids] = self._right_foot_0_rock_root_states[env_ids, 3:7].clone() # Good for now since rocks don't rotate
        self.left_foot_goal_pos[env_ids] = self._left_foot_0_rock_root_states[env_ids, 0:3].clone()
        self.left_foot_goal_pos[env_ids, 2] += 0.05
        self.left_foot_goal_rot[env_ids] = self._left_foot_0_rock_root_states[env_ids, 3:7].clone() # Good for now since rocks don't rotate

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        # compute aggregate size
        max_agg_bodies = self.humanoid_num_bodies + 12
        max_agg_shapes = self.humanoid_num_shapes + 12
        self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_rock(env_id, env_ptr, "right_hand_rock_0")
        self._build_rock(env_id, env_ptr, "left_hand_rock_0")
        self._build_rock(env_id, env_ptr, "right_foot_rock_0")
        self._build_rock(env_id, env_ptr, "left_foot_rock_0")
        self._build_rock(env_id, env_ptr, "hand_rock_1")
        self._build_rock(env_id, env_ptr, "foot_rock_1")
        self._build_rock(env_id, env_ptr, "hand_rock_2")
        self._build_rock(env_id, env_ptr, "foot_rock_2")
        self._build_rock(env_id, env_ptr, "hand_rock_3")
        self._build_rock(env_id, env_ptr, "foot_rock_3")
        self._build_rock(env_id, env_ptr, "hand_rock_4")
        self._build_rock(env_id, env_ptr, "foot_rock_4")
        self.gym.end_aggregate(env_ptr)

        return

    def _build_rock(self, env_id, env_ptr, rock_name):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        rock_handle = self.gym.create_actor(
            env_ptr,
            self._rock_asset,
            default_pose,
            rock_name,
            col_group,
            col_filter,
            segmentation_id,
        )
        dof_prop = self.gym.get_asset_dof_properties(self._rock_asset)
        dof_prop["driveMode"] = gymapi.DOF_MODE_POS
        dof_prop["effort"][:] = 0
        dof_prop["friction"][:] = 0.1
        dof_prop["damping"][:] = 0.05
        dof_prop["stiffness"][:] = 5.0
        self.gym.set_actor_dof_properties(env_ptr, rock_handle, dof_prop)
        self.gym.set_rigid_body_color(
            env_ptr,
            rock_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(133 / 255, 94 / 255, 66 / 255),
        )
        self._rock_handles.append(rock_handle)

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._rock_handles = []
        self._load_rock_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

        self.fingertip_right_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.fingertips_right
        ]
        self.hand_parts_right_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.hand_parts_right
        ]
        self.hand_dof_right_handles = [
            self.gym.find_asset_dof_index(self.humanoid_asset, name)
            for name in self.hand_dof_right
        ]
        self.hand_actions_right_handles = [
            self.gym.find_asset_actuator_index(self.humanoid_asset, name)
            for name in self.hand_actions_right
        ]
        self.fingertip_left_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.fingertips_left
        ]
        self.hand_parts_left_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.hand_parts_left
        ]
        self.hand_dof_left_handles = [
            self.gym.find_asset_dof_index(self.humanoid_asset, name)
            for name in self.hand_dof_left
        ]
        self.hand_actions_left_handles = [
            self.gym.find_asset_actuator_index(self.humanoid_asset, name)
            for name in self.hand_actions_left
        ]
        self.foot_right_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.foot_right
        ]
        self.foot_left_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.foot_left
        ]
        self.fingers_right_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.fingers_body_right
        ]
        self.fingers_left_handles = [
            self.gym.find_asset_rigid_body_index(self.humanoid_asset, name)
            for name in self.fingers_body_left
        ]

        self.fingertip_right_handles = to_torch(
            self.fingertip_right_handles, dtype=torch.long, device=self.device
        )
        self.hand_parts_right_handles = to_torch(
            self.hand_parts_right_handles, dtype=torch.long, device=self.device
        )
        self.hand_dof_right_handles = to_torch(
            self.hand_dof_right_handles, dtype=torch.long, device=self.device
        )
        self.hand_actions_right_handles = to_torch(
            self.hand_actions_right_handles, dtype=torch.long, device=self.device
        )
        self.fingertip_left_handles = to_torch(
            self.fingertip_left_handles, dtype=torch.long, device=self.device
        )
        self.hand_parts_left_handles = to_torch(
            self.hand_parts_left_handles, dtype=torch.long, device=self.device
        )
        self.hand_dof_left_handles = to_torch(
            self.hand_dof_left_handles, dtype=torch.long, device=self.device
        )
        self.hand_actions_left_handles = to_torch(
            self.hand_actions_left_handles, dtype=torch.long, device=self.device
        )
        self.foot_right_handles = to_torch(
            self.foot_right_handles, dtype=torch.long, device=self.device
        )
        self.foot_left_handles = to_torch(
            self.foot_left_handles, dtype=torch.long, device=self.device
        )
        self.fingers_right_handles = to_torch(
            self.fingers_right_handles, dtype=torch.long, device=self.device
        )
        self.fingers_left_handles = to_torch(
            self.fingers_left_handles, dtype=torch.long, device=self.device
        )
        return

    def _compute_humanoid_obs(self, env_ids=None):
        root_states = self._humanoid_root_states
        dof_pos = self._dof_pos.clone()
        dof_vel = self._dof_vel.clone()
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        right_wrist_pos = self._rigid_body_pos[:, self.hand_parts_right_handles[0], :].clone()  # 0 is right_palm
        right_wrist_rot = self._rigid_body_rot[:, self.hand_parts_right_handles[0], :].clone()
        right_dof_pos = self._dof_pos[:, self.hand_dof_right_handles].clone()
        right_dof_vel = self._dof_vel[:, self.hand_dof_right_handles].clone()
        right_fingertips_rot = self._rigid_body_rot[:, self.fingertip_right_handles, :].clone()
        right_fingertips_rot_flat = right_fingertips_rot.view(-1, 4)
        right_fingertips_pos = self._rigid_body_pos[:, self.fingertip_right_handles, :] + my_quat_rotate(
            right_fingertips_rot_flat, self.rfingers_rigid_body_com[:, [3, 7, 11, 15, 19]].view(-1, 3)
        ).view(right_fingertips_rot.shape[0], -1, 3)
        right_fingers_rot = self._rigid_body_rot[:, self.fingers_right_handles, :].clone()
        right_fingers_rot_flat = right_fingers_rot.view(-1, 4)
        right_fingers_pos = self._rigid_body_pos[:, self.fingers_right_handles, :] + my_quat_rotate(
            right_fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
        ).view(right_fingers_rot.shape[0], -1, 3)
        right_hand_actions = self.actions[:, self.hand_actions_right_handles]

        left_wrist_pos = self._rigid_body_pos[:, self.hand_parts_left_handles[0], :].clone()  # 0 is left_palm
        left_wrist_rot = self._rigid_body_rot[:, self.hand_parts_left_handles[0], :].clone()
        left_dof_pos = self._dof_pos[:, self.hand_dof_left_handles].clone()
        left_dof_vel = self._dof_vel[:, self.hand_dof_left_handles].clone()
        left_fingertips_rot = self._rigid_body_rot[:, self.fingertip_left_handles, :].clone()
        left_fingertips_rot_flat = left_fingertips_rot.view(-1, 4)
        left_fingertips_pos = self._rigid_body_pos[:, self.fingertip_left_handles, :] + my_quat_rotate(
            left_fingertips_rot_flat, self.lfingers_rigid_body_com[:, [3, 7, 11, 15, 19]].view(-1, 3)
        ).view(left_fingertips_rot.shape[0], -1, 3)
        left_fingers_rot = self._rigid_body_rot[:, self.fingers_left_handles, :].clone()
        left_fingers_rot_flat = left_fingers_rot.view(-1, 4)
        left_fingers_pos = self._rigid_body_pos[:, self.fingers_left_handles, :] + my_quat_rotate(
            left_fingers_rot_flat, self.lfingers_rigid_body_com.view(-1, 3)
        ).view(left_fingers_rot.shape[0], -1, 3)
        left_hand_actions = self.actions[:, self.hand_actions_left_handles]

        right_fingers_contact = self._contact_forces[:, self.fingers_right_handles]
        left_fingers_contact = self._contact_forces[:, self.fingers_left_handles]

        right_foot_rot = self._rigid_body_rot[:, self.foot_right_handles[0], :]
        right_foot_pos = self._rigid_body_pos[:, self.foot_right_handles[0], :] + my_quat_rotate(
            right_foot_rot, self.rfoot_rigid_body_com
        )
        
        left_foot_rot = self._rigid_body_rot[:, self.foot_left_handles[0], :]
        left_foot_pos = self._rigid_body_pos[:, self.foot_left_handles[0], :] + my_quat_rotate(
            left_foot_rot, self.lfoot_rigid_body_com
        )
        

        right_hand_goal_pos = self.right_hand_goal_pos.clone()
        left_hand_goal_pos = self.left_hand_goal_pos.clone()
        right_hand_goal_rot = self.right_hand_goal_rot.clone()
        left_hand_goal_rot = self.left_hand_goal_rot.clone()
        right_foot_goal_pos = self.right_foot_goal_pos.clone()
        left_foot_goal_pos = self.left_foot_goal_pos.clone()

        if env_ids is None:
            obs = compute_humanoid_observations_bouldering(
                root_states,
                dof_pos,
                dof_vel,
                key_body_pos,
                self._local_root_obs,
            )
            right_hand_obs = (
                build_pmp4setsip_hand_ip_observations(
                    right_wrist_pos,
                    right_wrist_rot,
                    right_dof_pos,
                    right_dof_vel,
                    right_fingertips_pos,
                    right_fingers_pos,
                    right_fingers_rot,
                    right_fingers_contact,
                    right_hand_goal_pos,
                    right_hand_goal_rot,
                    right_hand_actions,
                    self.init_rfingers_rot,
                    self.default_rfingers_facing_dir,
                    False
                )
            )
            left_hand_obs = (
                build_pmp4setsip_hand_ip_observations(
                    left_wrist_pos,
                    left_wrist_rot,
                    left_dof_pos,
                    left_dof_vel,
                    left_fingertips_pos,
                    left_fingers_pos,
                    left_fingers_rot,
                    left_fingers_contact,
                    left_hand_goal_pos,
                    left_hand_goal_rot,
                    left_hand_actions,
                    self.init_lfingers_rot,
                    self.default_lfingers_facing_dir,
                    True
                )
            )
            right_goal_foot_local_distance = build_goal_foot_local_distance(right_foot_pos, right_foot_rot,
                                                                            right_foot_goal_pos)
            left_goal_foot_local_distance = build_goal_foot_local_distance(left_foot_pos, left_foot_rot,
                                                                           left_foot_goal_pos)
        else:
            obs = compute_humanoid_observations_bouldering(
                root_states[env_ids],
                dof_pos[env_ids],
                dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
            )
            right_hand_obs = (
                build_pmp4setsip_hand_ip_observations(
                    right_wrist_pos[env_ids],
                    right_wrist_rot[env_ids],
                    right_dof_pos[env_ids],
                    right_dof_vel[env_ids],
                    right_fingertips_pos[env_ids],
                    right_fingers_pos[env_ids],
                    right_fingers_rot[env_ids],
                    right_fingers_contact[env_ids],
                    right_hand_goal_pos[env_ids],
                    right_hand_goal_rot[env_ids],
                    right_hand_actions[env_ids],
                    self.init_rfingers_rot[env_ids],
                    self.default_rfingers_facing_dir[env_ids],
                    False
                )
            )
            left_hand_obs = (
                build_pmp4setsip_hand_ip_observations(
                    left_wrist_pos[env_ids],
                    left_wrist_rot[env_ids],
                    left_dof_pos[env_ids],
                    left_dof_vel[env_ids],
                    left_fingertips_pos[env_ids],
                    left_fingers_pos[env_ids],
                    left_fingers_rot[env_ids],
                    left_fingers_contact[env_ids],
                    left_hand_goal_pos[env_ids],
                    left_hand_goal_rot[env_ids],
                    left_hand_actions[env_ids],
                    self.init_lfingers_rot[env_ids],
                    self.default_lfingers_facing_dir[env_ids],
                    True
                )
            )
            right_goal_foot_local_distance = build_goal_foot_local_distance(right_foot_pos[env_ids],
                                                                            right_foot_rot[env_ids],
                                                                            right_foot_goal_pos[env_ids])
            left_goal_foot_local_distance = build_goal_foot_local_distance(left_foot_pos[env_ids],
                                                                           left_foot_rot[env_ids],
                                                                           left_foot_goal_pos[env_ids])

        obs = torch.cat(
            (
                obs,
                right_hand_obs[:,13:],
                left_hand_obs[:,13:],
                right_goal_foot_local_distance,
                left_goal_foot_local_distance
            ),
            dim=-1
        )
        return obs

    def _load_rock_asset(self):
        asset_root = "../assets/mjcf/"
        asset_file = "rock.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        _rock_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        rock_num_shapes = self.gym.get_asset_rigid_shape_count(_rock_asset)
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(_rock_asset)
        for k in range(rock_num_shapes):
            rigid_shape_prop[k].filter = 0
            rigid_shape_prop[k].contact_offset = 0.002
            rigid_shape_prop[k].friction = 10.0
            rigid_shape_prop[k].rolling_friction = rigid_shape_prop[k].friction / 100.0
        self.gym.set_asset_rigid_shape_properties(_rock_asset, rigid_shape_prop)

        self._rock_asset = _rock_asset

        return

    @property
    def pmp4setsip_observation_space(self):
        return self._pmp4setsip_obs_space

    def fetch_pmp4setsip_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._pmp4setsip_obs_demo_buf is None:
            self._build_pmp4setsip_obs_demo_buf(num_samples)
        else:
            assert self._pmp4setsip_obs_demo_buf.shape[0] == num_samples

        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(
            np.expand_dims(motion_ids, axis=-1), [1, self._num_pmp4setsip_obs_steps]
        )
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_pmp4setsip_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        pmp4setsip_obs_demo = build_pmp4setsip_observations(
            root_states, dof_pos, dof_vel, key_pos, self._local_root_obs
        )

        right_dof_pos = dof_pos[:, self.hand_dof_right_handles]
        right_dof_vel = dof_vel[:, self.hand_dof_right_handles]

        right_hand_moref_obs = build_pmp4setsip_hand_moref_observations(
            right_dof_pos,
            right_dof_vel
        )

        left_dof_pos = dof_pos[:, self.hand_dof_left_handles]
        left_dof_vel = dof_vel[:, self.hand_dof_left_handles]

        left_hand_moref_obs = build_pmp4setsip_hand_moref_observations(
            left_dof_pos,
            left_dof_vel
        )

        right_motion_ids = self._motion_right_hand_ip.sample_motions(num_samples)
        right_motion_times0 = self._motion_right_hand_ip.sample_time(right_motion_ids)
        right_motion_ids = np.tile(
            np.expand_dims(right_motion_ids, axis=-1),
            [1, self._num_pmp4setsip_obs_steps],
        )
        right_motion_times = np.expand_dims(right_motion_times0, axis=-1)
        right_time_steps = -dt * np.arange(0, self._num_pmp4setsip_obs_steps)
        right_motion_times = right_motion_times + right_time_steps

        right_motion_ids = right_motion_ids.flatten()
        right_motion_times = right_motion_times.flatten()
        right_state_and_actions = (
            self._motion_right_hand_ip.get_motion_state_and_actions(
                right_motion_ids, right_motion_times
            )
        )

        left_motion_ids = self._motion_left_hand_ip.sample_motions(num_samples)
        left_motion_times0 = self._motion_left_hand_ip.sample_time(left_motion_ids)
        left_motion_ids = np.tile(
            np.expand_dims(left_motion_ids, axis=-1),
            [1, self._num_pmp4setsip_obs_steps],
        )
        left_motion_times = np.expand_dims(left_motion_times0, axis=-1)
        left_time_steps = -dt * np.arange(0, self._num_pmp4setsip_obs_steps)
        left_motion_times = left_motion_times + left_time_steps

        left_motion_ids = left_motion_ids.flatten()
        left_motion_times = left_motion_times.flatten()
        left_state_and_actions = self._motion_left_hand_ip.get_motion_state_and_actions(
            left_motion_ids, left_motion_times
        )

        obs = torch.cat(
            [
                pmp4setsip_obs_demo,
                right_state_and_actions,
                left_state_and_actions,
                right_hand_moref_obs,
                left_hand_moref_obs,
            ],
            dim=-1,
        )
        self._pmp4setsip_obs_demo_buf[:] = obs.view(self._pmp4setsip_obs_demo_buf.shape)

        pmp4setsip_obs_demo_flat = self._pmp4setsip_obs_demo_buf.view(
            -1, self.num_pmp4setsip_obs
        )
        return pmp4setsip_obs_demo_flat

    def _build_pmp4setsip_obs_demo_buf(self, num_samples):
        self._pmp4setsip_obs_demo_buf = torch.zeros(
            (
                num_samples,
                self._num_pmp4setsip_obs_steps,
                NUM_PMP4SETSIP_OBS_PER_STEP,
            ),
            device=self.device,
            dtype=torch.float,
        )
        return

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLibMPL(
            motion_file=motion_file,
            num_dofs=self.humanoid_num_dof,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device,
        )
        return

    def _load_hands_interactive_prior(self, hands_ip_file):
        self._motion_right_hand_ip = MotionLibIP(
            motion_file=hands_ip_file,
            device=self.device,
        )
        self._motion_left_hand_ip = MotionLibIP(
            motion_file=hands_ip_file,
            device=self.device,
        )
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_pmp4setsip_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (
                self._state_init == HumanoidRockBoulderingPHA4SetsIP.StateInit.Start
                or self._state_init
                == HumanoidRockBoulderingPHA4SetsIP.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self.actions[env_ids] = 0
        self.cur_targets[env_ids, :] = 0

        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        if (
                self._state_init == HumanoidRockBoulderingPHA4SetsIP.StateInit.Random
        ):
            motion_ids = self._motion_lib.sample_motions(num_envs)
            motion_right_hand_ip_ids = self._motion_right_hand_ip.sample_motions(num_envs)
            motion_left_hand_ip_ids = self._motion_left_hand_ip.sample_motions(num_envs)
            motion_times = self._motion_lib.sample_time(motion_ids)
            motion_right_hand_ip_times = self._motion_right_hand_ip.sample_time(motion_right_hand_ip_ids)
            motion_left_hand_ip_times = self._motion_left_hand_ip.sample_time(motion_left_hand_ip_ids)
            # # Try only the first motion
            # motion_right_hand_ip_ids = np.zeros(num_envs, dtype=np.int32)
            # motion_left_hand_ip_ids = np.zeros(num_envs, dtype=np.int32)
            # motion_times = np.zeros(num_envs)
            # motion_right_hand_ip_times = np.zeros(num_envs)
            # motion_left_hand_ip_times = np.zeros(num_envs)
        elif self._state_init == HumanoidRockBoulderingPHA4SetsIP.StateInit.Start:
            motion_times = np.zeros(num_envs)
            motion_right_hand_ip_times = np.zeros(num_envs)
            motion_left_hand_ip_times = np.zeros(num_envs)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )

        self.motion_ids[env_ids] = to_torch(
            motion_ids, dtype=torch.int64, device=self.device
        )

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

        self._reset_ref_right_hand_ip_motion_ids = motion_right_hand_ip_ids
        self._reset_ref_right_hand_ip_motion_times = motion_right_hand_ip_times

        self._reset_ref_left_hand_ip_motion_ids = motion_left_hand_ip_ids
        self._reset_ref_left_hand_ip_motion_times = motion_left_hand_ip_times
        return

    def _init_pmp4setsip_obs(self, env_ids):
        self._compute_pmp4setsip_observations(env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_pmp4setsip_obs_ref(
                self._reset_ref_env_ids,
                self._reset_ref_motion_ids,
                self._reset_ref_motion_times,
                self._reset_ref_right_hand_ip_motion_ids,
                self._reset_ref_right_hand_ip_motion_times,
                self._reset_ref_left_hand_ip_motion_ids,
                self._reset_ref_left_hand_ip_motion_times,
            )
        return

    def _init_pmp4setsip_obs_ref(
            self,
            env_ids,
            motion_ids,
            motion_times,
            right_motion_ids,
            right_motion_times,
            left_motion_ids,
            left_motion_times,
    ):
        dt = self.dt
        motion_ids = np.tile(
            np.expand_dims(motion_ids, axis=-1), [1, self._num_pmp4setsip_obs_steps - 1]
        )
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_pmp4setsip_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        pmp4setsip_obs_demo = build_pmp4setsip_observations(
            root_states, dof_pos, dof_vel, key_pos, self._local_root_obs
        )

        right_motion_ids = np.tile(
            np.expand_dims(right_motion_ids, axis=-1),
            [1, self._num_pmp4setsip_obs_steps - 1],
        )
        right_motion_times = np.expand_dims(right_motion_times, axis=-1)
        right_time_steps = -dt * (np.arange(0, self._num_pmp4setsip_obs_steps - 1) + 1)
        right_motion_times = right_motion_times + right_time_steps

        right_motion_ids = right_motion_ids.flatten()
        right_motion_times = right_motion_times.flatten()
        right_state_and_actions = (
            self._motion_right_hand_ip.get_motion_state_and_actions(
                right_motion_ids, right_motion_times
            )
        )

        left_motion_ids = np.tile(
            np.expand_dims(left_motion_ids, axis=-1),
            [1, self._num_pmp4setsip_obs_steps - 1],
        )
        left_motion_times = np.expand_dims(left_motion_times, axis=-1)
        left_time_steps = -dt * (np.arange(0, self._num_pmp4setsip_obs_steps - 1) + 1)
        left_motion_times = left_motion_times + left_time_steps

        left_motion_ids = left_motion_ids.flatten()
        left_motion_times = left_motion_times.flatten()
        left_state_and_actions = self._motion_left_hand_ip.get_motion_state_and_actions(
            left_motion_ids, left_motion_times
        )

        right_hand_moref_obs = right_state_and_actions[..., 13:42].clone()
        left_hand_moref_obs = left_state_and_actions[..., 13:42].clone()

        obs = torch.cat(
            [
                pmp4setsip_obs_demo,
                right_state_and_actions,
                left_state_and_actions,
                right_hand_moref_obs,
                left_hand_moref_obs
            ],
            dim=-1,
        )

        self._hist_pmp4setsip_obs_buf[env_ids] = obs.view(
            self._hist_pmp4setsip_obs_buf[env_ids].shape
        )
        return

    def _set_env_state(
            self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel
    ):
        self._humanoid_root_states[env_ids, 0] = -0.20
        self._humanoid_root_states[env_ids, 1] = -0.14
        self._humanoid_root_states[env_ids, 2] = 2.2
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7] = 1.0
        self._humanoid_root_states[env_ids, 8:] = 0.0

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        humanoid_env_ids_int32 = self._humanoid_actor_ids[env_ids]

        # Update rock wall positions
        self._reset_rocks(env_ids)

        right_hand_0_rock_env_ids_int32 = self._right_hand_0_rock_actor_ids[env_ids]
        left_hand_0_rock_env_ids_int32 = self._left_hand_0_rock_actor_ids[env_ids]
        right_foot_0_rock_env_ids_int32 = self._right_foot_0_rock_actor_ids[env_ids]
        left_foot_0_rock_env_ids_int32 = self._left_foot_0_rock_actor_ids[env_ids]
        hand_1_rock_env_ids_int32 = self._hand_1_rock_actor_ids[env_ids]
        foot_1_rock_env_ids_int32 = self._foot_1_rock_actor_ids[env_ids]
        hand_2_rock_env_ids_int32 = self._hand_2_rock_actor_ids[env_ids]
        foot_2_rock_env_ids_int32 = self._foot_2_rock_actor_ids[env_ids]
        hand_3_rock_env_ids_int32 = self._hand_3_rock_actor_ids[env_ids]
        foot_3_rock_env_ids_int32 = self._foot_3_rock_actor_ids[env_ids]
        hand_4_rock_env_ids_int32 = self._hand_4_rock_actor_ids[env_ids]
        foot_4_rock_env_ids_int32 = self._foot_4_rock_actor_ids[env_ids]

        actor_env_ids_int32 = torch.cat(
            (
                humanoid_env_ids_int32,
                right_hand_0_rock_env_ids_int32,
                left_hand_0_rock_env_ids_int32,
                right_foot_0_rock_env_ids_int32,
                left_foot_0_rock_env_ids_int32,
                hand_1_rock_env_ids_int32,
                foot_1_rock_env_ids_int32,
                hand_2_rock_env_ids_int32,
                foot_2_rock_env_ids_int32,
                hand_3_rock_env_ids_int32,
                foot_3_rock_env_ids_int32,
                hand_4_rock_env_ids_int32,
                foot_4_rock_env_ids_int32
            ),
            0
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_env_ids_int32),
            len(actor_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoid_env_ids_int32),
            len(humanoid_env_ids_int32),
        )

        return

    def _update_hist_pmp4setsip_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._pmp4setsip_obs_buf.shape[1] - 1)):
                self._pmp4setsip_obs_buf[:, i + 1] = self._pmp4setsip_obs_buf[:, i]
        else:
            for i in reversed(range(self._pmp4setsip_obs_buf.shape[1] - 1)):
                self._pmp4setsip_obs_buf[env_ids, i + 1] = self._pmp4setsip_obs_buf[
                    env_ids, i
                ]
        return

    def _compute_pmp4setsip_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        right_wrist_pos = self._rigid_body_pos[:, self.hand_parts_right_handles[0], :]  # 0 is right_palm
        right_wrist_rot = self._rigid_body_rot[:, self.hand_parts_right_handles[0], :]
        right_dof_pos = self._dof_pos[:, self.hand_dof_right_handles]
        right_dof_vel = self._dof_vel[:, self.hand_dof_right_handles]
        right_fingertips_rot = self._rigid_body_rot[:, self.fingertip_right_handles, :]
        right_fingertips_rot_flat = right_fingertips_rot.view(-1, 4)
        right_fingertips_pos = self._rigid_body_pos[:, self.fingertip_right_handles, :] + my_quat_rotate(
            right_fingertips_rot_flat, self.rfingers_rigid_body_com[:, [3, 7, 11, 15, 19]].view(-1, 3)
        ).view(right_fingertips_rot.shape[0], -1, 3)
        right_fingers_rot = self._rigid_body_rot[:, self.fingers_right_handles, :]
        right_fingers_rot_flat = right_fingers_rot.view(-1, 4)
        right_fingers_pos = self._rigid_body_pos[:, self.fingers_right_handles, :] + my_quat_rotate(
            right_fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
        ).view(right_fingers_rot.shape[0], -1, 3)
        right_hand_actions = self.actions[:, self.hand_actions_right_handles]

        left_wrist_pos = self._rigid_body_pos[:, self.hand_parts_left_handles[0], :]  # 0 is left_palm
        left_wrist_rot = self._rigid_body_rot[:, self.hand_parts_left_handles[0], :]
        left_dof_pos = self._dof_pos[:, self.hand_dof_left_handles]
        left_dof_vel = self._dof_vel[:, self.hand_dof_left_handles]
        left_fingertips_rot = self._rigid_body_rot[:, self.fingertip_left_handles, :]
        left_fingertips_rot_flat = left_fingertips_rot.view(-1, 4)
        left_fingertips_pos = self._rigid_body_pos[:, self.fingertip_left_handles, :] + my_quat_rotate(
            left_fingertips_rot_flat, self.lfingers_rigid_body_com[:, [3, 7, 11, 15, 19]].view(-1, 3)
        ).view(left_fingertips_rot.shape[0], -1, 3)
        left_fingers_rot = self._rigid_body_rot[:, self.fingers_left_handles, :]
        left_fingers_rot_flat = left_fingers_rot.view(-1, 4)
        left_fingers_pos = self._rigid_body_pos[:, self.fingers_left_handles, :] + my_quat_rotate(
            left_fingers_rot_flat, self.lfingers_rigid_body_com.view(-1, 3)
        ).view(left_fingers_rot.shape[0], -1, 3)
        left_hand_actions = self.actions[:, self.hand_actions_left_handles]

        right_fingers_contact = self._contact_forces[:, self.fingers_right_handles]
        left_fingers_contact = self._contact_forces[:, self.fingers_left_handles]

        right_hand_goal_pos = self.right_hand_goal_pos
        left_hand_goal_pos = self.left_hand_goal_pos
        right_hand_goal_rot = self.right_hand_goal_rot
        left_hand_goal_rot = self.left_hand_goal_rot

        right_hand_moref_obs = build_pmp4setsip_hand_moref_observations(
            right_dof_pos,
            right_dof_vel
        )
        left_hand_moref_obs = build_pmp4setsip_hand_moref_observations(
            left_dof_pos,
            left_dof_vel
        )

        if env_ids is None:
            self._curr_pmp4setsip_obs_buf[:, 0:105] = build_pmp4setsip_observations(
                self._humanoid_root_states,
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                self._local_root_obs,
            )
            self._curr_pmp4setsip_obs_buf[:, 105:192] = build_pmp4setsip_hand_ip_observations(
                right_wrist_pos,
                right_wrist_rot,
                right_dof_pos,
                right_dof_vel,
                right_fingertips_pos,
                right_fingers_pos,
                right_fingers_rot,
                right_fingers_contact,
                right_hand_goal_pos,
                right_hand_goal_rot,
                right_hand_actions,
                self.init_rfingers_rot,
                self.default_rfingers_facing_dir,
                False
            )
            self._curr_pmp4setsip_obs_buf[:, 192:279] = build_pmp4setsip_hand_ip_observations(
                left_wrist_pos,
                left_wrist_rot,
                left_dof_pos,
                left_dof_vel,
                left_fingertips_pos,
                left_fingers_pos,
                left_fingers_rot,
                left_fingers_contact,
                left_hand_goal_pos,
                left_hand_goal_rot,
                left_hand_actions,
                self.init_lfingers_rot,
                self.default_lfingers_facing_dir,
                True
            )
            self._curr_pmp4setsip_obs_buf[:, 279:308] = right_hand_moref_obs
            self._curr_pmp4setsip_obs_buf[:, 308:337] = left_hand_moref_obs
        else:
            self._curr_pmp4setsip_obs_buf[env_ids, 0:105] = build_pmp4setsip_observations(
                self._humanoid_root_states[env_ids],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
            )
            self._curr_pmp4setsip_obs_buf[env_ids, 105:192] = build_pmp4setsip_hand_ip_observations(
                right_wrist_pos[env_ids],
                right_wrist_rot[env_ids],
                right_dof_pos[env_ids],
                right_dof_vel[env_ids],
                right_fingertips_pos[env_ids],
                right_fingers_pos[env_ids],
                right_fingers_rot[env_ids],
                right_fingers_contact[env_ids],
                right_hand_goal_pos[env_ids],
                right_hand_goal_rot[env_ids],
                right_hand_actions[env_ids],
                self.init_rfingers_rot[env_ids],
                self.default_rfingers_facing_dir[env_ids],
                False
            )
            self._curr_pmp4setsip_obs_buf[env_ids, 192:279] = build_pmp4setsip_hand_ip_observations(
                left_wrist_pos[env_ids],
                left_wrist_rot[env_ids],
                left_dof_pos[env_ids],
                left_dof_vel[env_ids],
                left_fingertips_pos[env_ids],
                left_fingers_pos[env_ids],
                left_fingers_rot[env_ids],
                left_fingers_contact[env_ids],
                left_hand_goal_pos[env_ids],
                left_hand_goal_rot[env_ids],
                left_hand_actions[env_ids],
                self.init_lfingers_rot[env_ids],
                self.default_lfingers_facing_dir[env_ids],
                True
            )
            self._curr_pmp4setsip_obs_buf[env_ids, 279:308] = right_hand_moref_obs[env_ids]
            self._curr_pmp4setsip_obs_buf[env_ids, 308:337] = left_hand_moref_obs[env_ids]

        return

    def _compute_reward(self, actions):
        right_wrist_pos = self._rigid_body_pos[:, self.hand_parts_right_handles[0], :]  # 0 is right_palm
        right_wrist_rot = self._rigid_body_rot[:, self.hand_parts_right_handles[0], :]
        right_palm_pos = right_wrist_pos + my_quat_rotate(
            right_wrist_rot, self.rpalm_rigid_body_com
        )
        left_wrist_pos = self._rigid_body_pos[:, self.hand_parts_left_handles[0], :]  # 0 is left_palm
        left_wrist_rot = self._rigid_body_rot[:, self.hand_parts_left_handles[0], :]
        left_palm_pos = left_wrist_pos + my_quat_rotate(
            left_wrist_rot, self.lpalm_rigid_body_com
        )

        right_fingers_rot = self._rigid_body_rot[:, self.fingers_right_handles, :]
        right_fingers_rot_flat = right_fingers_rot.view(-1, 4)
        right_fingers_pos = self._rigid_body_pos[:, self.fingers_right_handles, :] + my_quat_rotate(
            right_fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
        ).view(right_fingers_rot.shape[0], -1, 3)

        left_fingers_rot = self._rigid_body_rot[:, self.fingers_left_handles, :]
        left_fingers_rot_flat = left_fingers_rot.view(-1, 4)
        left_fingers_pos = self._rigid_body_pos[:, self.fingers_left_handles, :] + my_quat_rotate(
            left_fingers_rot_flat, self.lfingers_rigid_body_com.view(-1, 3)
        ).view(left_fingers_rot.shape[0], -1, 3)

        right_foot_rot = self._rigid_body_rot[:, self.foot_right_handles[0], :]
        right_foot_pos = self._rigid_body_pos[:, self.foot_right_handles[0], :] + my_quat_rotate(
            right_foot_rot, self.rfoot_rigid_body_com
        )
        left_foot_rot = self._rigid_body_rot[:, self.foot_left_handles[0], :]
        left_foot_pos = self._rigid_body_pos[:, self.foot_left_handles[0], :] + my_quat_rotate(
            left_foot_rot, self.lfoot_rigid_body_com
        )
        

        right_hand_goal_pos = self.right_hand_goal_pos.clone()
        right_hand_goal_rot = self.right_hand_goal_rot.clone()
        left_hand_goal_pos = self.left_hand_goal_pos.clone()
        left_hand_goal_rot = self.left_hand_goal_rot.clone()
        right_foot_goal_pos = self.right_foot_goal_pos.clone()
        left_foot_goal_pos = self.left_foot_goal_pos.clone()

        dof_actuator_forces = self.dof_force_tensor[:, self.actuated_dof_indices]
        motor_efforts = self.motor_efforts
        max_motor_effort = self.max_motor_effort

        self.rew_buf[:] = compute_humanoid_reward(
            right_fingers_pos,
            right_wrist_pos,
            right_wrist_rot,
            right_hand_goal_pos,
            right_hand_goal_rot,
            left_fingers_pos,
            left_wrist_pos,
            left_wrist_rot,
            left_hand_goal_pos,
            left_hand_goal_rot,
            right_foot_pos,
            right_foot_goal_pos,
            left_foot_pos,
            left_foot_goal_pos,
            actions,
            dof_actuator_forces,
            motor_efforts,
            max_motor_effort,
            self.progress_buf
        )
        return

    def _compute_reset(self):
        right_wrist_pos = self._rigid_body_pos[:, self.hand_parts_right_handles[0], :].clone()  # 0 is right_palm
        right_wrist_rot = self._rigid_body_rot[:, self.hand_parts_right_handles[0], :].clone()  # 0 is right_palm
        right_palm_pos = right_wrist_pos + my_quat_rotate(
            right_wrist_rot, self.rpalm_rigid_body_com
        )
        left_wrist_pos = self._rigid_body_pos[:, self.hand_parts_left_handles[0], :].clone()  # 0 is left_palm
        left_wrist_rot = self._rigid_body_rot[:, self.hand_parts_left_handles[0], :].clone()  # 0 is left_palm
        left_palm_pos = left_wrist_pos + my_quat_rotate(
            left_wrist_rot, self.lpalm_rigid_body_com
        )
        right_hand_goal_pos = self.right_hand_goal_pos.clone()
        left_hand_goal_pos = self.left_hand_goal_pos.clone()

        right_foot_rot = self._rigid_body_rot[:, self.foot_right_handles[0], :].clone()
        right_foot_pos = self._rigid_body_pos[:, self.foot_right_handles[0], :].clone() + my_quat_rotate(
            right_foot_rot, self.rfoot_rigid_body_com
        )
        
        left_foot_rot = self._rigid_body_rot[:, self.foot_left_handles[0], :].clone()
        left_foot_pos = self._rigid_body_pos[:, self.foot_left_handles[0], :].clone() + my_quat_rotate(
            left_foot_rot, self.lfoot_rigid_body_com
        )
        
        right_foot_goal_pos = self.right_foot_goal_pos.clone()
        left_foot_goal_pos = self.left_foot_goal_pos.clone()

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf.clone(),
            self.progress_buf.clone(),
            self._contact_forces.clone(),
            self._contact_body_ids.clone(),
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_height,
            right_palm_pos,
            right_hand_goal_pos,
            left_palm_pos,
            left_hand_goal_pos,
            right_foot_pos,
            right_foot_goal_pos,
            left_foot_pos,
            left_foot_goal_pos
        )
        self.update_rock_goal_pos()
        return


#####################################################################
### =========================jit functions=========================###
#####################################################################
@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    # AMP + MPL
    dof_obs_size = 16
    dof_offsets = [
        0,  # 'right_wrist_PRO', 'right_wrist_UDEV', 'right_wrist_FLEX',
        3,  # 'right_thumb_ABD',
        4,  # 'right_thumb_MCP'
        5,  # 'right_thumb_PIP'
        6,  # 'right_thumb_DIP'
        7,  # 'right_index_ABD'
        8,  # 'right_index_MCP'
        9,  # 'right_index_PIP',
        10,  # 'right_index_DIP',
        11,  # 'right_middle_MCP',
        12,  # 'right_middle_PIP',
        13,  # 'right_middle_DIP',
        14,  # 'right_ring_ABD',
        15,  # 'right_ring_MCP',
        16,  # 'right_ring_PIP',
        17,  # 'right_ring_DIP',
        18,  # 'right_pinky_ABD',
        19,  # 'right_pinky_MCP',
        20,  # 'right_pinky_PIP',
        21,  # 'right_pinky_DIP',
        22,  # END
    ]
    NOT_ACTUATED_DOF_INDICES = [9, 10, 12, 13, 14, 16, 17, 20, 21]

    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset: (dof_offset + dof_size)]
        if dof_offset in NOT_ACTUATED_DOF_INDICES:
            continue
        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset: (dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs


@torch.jit.script
def build_pmp4setsip_hand_moref_observations(
        hand_dof_pos, hand_dof_vel
):
    # type: (Tensor, Tensor) -> Tensor

    dof_obs = dof_to_obs(hand_dof_pos)

    ACTUATED_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 18, 19]
    dof_vel = hand_dof_vel[:, ACTUATED_DOF_INDICES]

    moref_obs = torch.cat(
        (
            dof_obs,
            dof_vel,
        ),
        dim=-1,
    )
    return moref_obs


@torch.jit.script
def build_goal_foot_local_distance(
        foot_pos,
        foot_rot,
        goal_foot_pos,
):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    heading_rot = calc_heading_quat_inv(foot_rot)

    local_key_body_pos = goal_foot_pos - foot_pos

    local_end_pos = my_quat_rotate(heading_rot, local_key_body_pos)

    return local_end_pos


@torch.jit.script
def build_pmp4setsip_observations(
        root_states, dof_pos, dof_vel, key_body_pos, local_root_obs
):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs_mpl(dof_pos)

    BODY_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 33, 34, 35, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                        70, 71]
    dof_vel = dof_vel[:, BODY_DOF_INDICES]

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_humanoid_observations_bouldering(
        root_states,
        dof_pos,
        dof_vel,
        key_body_pos,
        local_root_obs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat(
        (1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs_mpl(dof_pos)

    BODY_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 33, 34, 35, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                        70, 71]
    dof_vel = dof_vel[:, BODY_DOF_INDICES]

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1)
    return obs


@torch.jit.script
def build_pmp4setsip_hand_ip_observations(
        hand_wrist_pos,
        hand_wrist_rot,
        hand_dof_pos,
        hand_dof_vel,
        fingertips_body_pos,
        fingers_pos,
        fingers_rot,
        fingers_contact,
        rope_pos,
        rope_rot,
        hand_actions,
        init_fingers_rot,
        default_fingers_facing_dir,
        is_left
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    dof_obs = dof_to_obs(hand_dof_pos)

    ACTUATED_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 18, 19]
    dof_vel = hand_dof_vel[:, ACTUATED_DOF_INDICES]

    wrist_quat_inv = quat_conjugate(hand_wrist_rot)

    # handle position in hand-centric coordinate
    rel_handle_pos = my_quat_rotate(wrist_quat_inv, rope_pos - hand_wrist_pos)

    offset1 = torch.zeros_like(rel_handle_pos)
    offset2 = offset1.clone()
    CAPSULE_HALF_LEN = 0.30
    offset1[:, 1] = CAPSULE_HALF_LEN
    offset2[:, 1] = -CAPSULE_HALF_LEN
    rel_handle_end1 = my_quat_rotate(quat_mul(wrist_quat_inv, rope_rot), offset1) + rel_handle_pos
    rel_handle_end2 = my_quat_rotate(quat_mul(wrist_quat_inv, rope_rot), offset2) + rel_handle_pos

    # finger facing dir in wrist coordinate
    fingers_rot_flat = fingers_rot.view(-1, 4)
    init_fingers_rot_flat = init_fingers_rot.view(-1, 4)
    fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_fingers_rot_flat))
    fingers_facing_dir_flat = my_quat_rotate(
        fingers_rot_delta_flat, default_fingers_facing_dir.view(-1, 3)
    )
    wrist_quat_inv_expand_flat = wrist_quat_inv[:, None].repeat(1, fingers_pos.shape[1], 1).view(-1, 4)
    fingers_facing_dir_flat = my_quat_rotate(wrist_quat_inv_expand_flat, fingers_facing_dir_flat)
    fingers_facing_dir = fingers_facing_dir_flat.view(-1, 20, 3)
    rel_fingers_contact = my_quat_rotate(wrist_quat_inv_expand_flat, fingers_contact.view(-1, 3)).view(
        -1, 20, 3
    )
    fingertips_handles = [7, 11, 15, 19]
    tip_valid_contact_marker = (
            (rel_fingers_contact[:, fingertips_handles] * fingers_facing_dir[:, fingertips_handles]).sum(dim=-1) < 0
    ).float()

    # Shadow_hand fingertips right pos wrt wrist p_line_e
    fingertips_local_pos = fingertips_body_pos - hand_wrist_pos.unsqueeze(1)
    fingertips_local_pos = fingertips_local_pos.view(
        fingertips_body_pos.shape[0] * 5, 3
    )
    fingertips_local_pos = quat_rotate(
        wrist_quat_inv.repeat(5, 1), fingertips_local_pos
    )
    fingertips_local_pos_flat = fingertips_local_pos.view(
        fingertips_body_pos.shape[0],
        -1,
    )

    fingers_facing_dir_to_contact_cosine = (normalize(rel_fingers_contact) * fingers_facing_dir).sum(dim=-1)

    if is_left:
        temp = rel_handle_end1.clone()
        rel_handle_end1 = rel_handle_end2.clone()
        rel_handle_end2 = temp
        fingertips_local_pos[..., 0] = -fingertips_local_pos[..., 0]
        fingertips_local_pos_flat = fingertips_local_pos.view(
            fingertips_body_pos.shape[0],
            -1,
        )

    obs = torch.cat(
        (
            hand_actions,
            dof_obs,
            dof_vel,
            rel_handle_end1,
            rel_handle_end2,
            tip_valid_contact_marker,
            fingertips_local_pos_flat,
            fingers_facing_dir_to_contact_cosine,
        ),
        dim=-1,
    )
    return obs


# Target keep pose
@torch.jit.script
def compute_humanoid_reward(
    right_fingers_pos,
    right_hand_wrist_pos,
    right_hand_wrist_rot,
    right_hand_goal_pos,
    right_hand_goal_rot,
    left_fingers_pos,
    left_hand_wrist_pos,
    left_hand_wrist_rot,
    left_hand_goal_pos,
    left_hand_goal_rot,
    right_foot_pos,
    right_foot_rock_pos,
    left_foot_pos,
    left_foot_rock_pos,
    actions,
    dof_actuator_forces,
    motor_efforts,
    max_motor_effort,
    progress_buf,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tensor
    right_hand_reaching_scale = torch.where(progress_buf < 30, 64.0, 16.0)
    left_hand_reaching_scale = torch.where(progress_buf < 30, 64.0, 16.0)
    right_foot_reaching_scale = torch.where(progress_buf < 30, 64.0, 16.0)
    left_foot_reaching_scale = torch.where(progress_buf < 30, 64.0, 16.0)
    offset_s = 0.05
    CAPSULE_HALF_LEN = 0.05

    # Right fingertips dist to handle
    right_wrist_quat_inv = quat_conjugate(right_hand_wrist_rot)
    right_rel_handle_pos = my_quat_rotate(right_wrist_quat_inv, right_hand_goal_pos - right_hand_wrist_pos)

    right_offset1 = torch.zeros_like(right_rel_handle_pos)
    right_offset2 = right_offset1.clone()
    right_offset1[:, 1] = CAPSULE_HALF_LEN
    right_offset2[:, 1] = -CAPSULE_HALF_LEN
    right_rel_handle_end1 = my_quat_rotate(right_hand_goal_rot, right_offset1) + right_hand_goal_pos
    right_rel_handle_end2 = my_quat_rotate(right_hand_goal_rot, right_offset2) + right_hand_goal_pos
    right_end1_to_x = right_fingers_pos - right_rel_handle_end1[:, None]
    right_end1_to_end2 = (right_rel_handle_end2 - right_rel_handle_end1)[:, None]
    right_end1_to_end2_norm = normalize(right_end1_to_end2)
    right_proj_end1_to_x = (right_end1_to_x * right_end1_to_end2_norm).sum(dim=-1,
                                                                           keepdim=True) * right_end1_to_end2_norm
    right_rel_portion = (right_proj_end1_to_x * right_end1_to_end2_norm).sum(dim=-1,
                                                                             keepdim=True) / right_end1_to_end2.pow(
        2).sum(
        dim=-1, keepdim=True
    ).sqrt()
    right_rel_portion_clipped = right_rel_portion.clip(0, 1)
    right_proj_end1_to_x_clipped = right_rel_portion_clipped * right_end1_to_end2
    right_fingers_to_target_disp = right_proj_end1_to_x_clipped - right_end1_to_x
    right_fingertips_to_target_disp = right_fingers_to_target_disp[:, [7, 11, 15, 19]]
    right_fingertips_to_target_dist = right_fingertips_to_target_disp.pow(2).sum(dim=-1).sqrt()
    right_hand_rope_dist = (
        torch.maximum(
            torch.zeros_like(right_fingertips_to_target_dist),
            right_fingertips_to_target_dist - 0.02,
        )
    ).pow(2).max(dim=-1).values

    rhand_reward = torch.exp(
        -right_hand_reaching_scale * right_hand_rope_dist)
    right_hand_reaching_reward = ((1.0 - offset_s) * rhand_reward + offset_s)

    # Left fingertips dist to handle
    left_wrist_quat_inv = quat_conjugate(left_hand_wrist_rot)
    left_rel_handle_pos = my_quat_rotate(left_wrist_quat_inv, left_hand_goal_pos - left_hand_wrist_pos)

    left_offset1 = torch.zeros_like(left_rel_handle_pos)
    left_offset2 = left_offset1.clone()
    left_offset1[:, 1] = CAPSULE_HALF_LEN
    left_offset2[:, 1] = -CAPSULE_HALF_LEN
    left_rel_handle_end1 = my_quat_rotate(left_hand_goal_rot, left_offset1) + left_hand_goal_pos
    left_rel_handle_end2 = my_quat_rotate(left_hand_goal_rot, left_offset2) + left_hand_goal_pos
    left_end1_to_x = left_fingers_pos - left_rel_handle_end1[:, None]
    left_end1_to_end2 = (left_rel_handle_end2 - left_rel_handle_end1)[:, None]
    left_end1_to_end2_norm = normalize(left_end1_to_end2)
    left_proj_end1_to_x = (left_end1_to_x * left_end1_to_end2_norm).sum(dim=-1,
                                                                        keepdim=True) * left_end1_to_end2_norm
    left_rel_portion = (left_proj_end1_to_x * left_end1_to_end2_norm).sum(dim=-1,
                                                                          keepdim=True) / left_end1_to_end2.pow(
        2).sum(
        dim=-1, keepdim=True
    ).sqrt()
    left_rel_portion_clipped = left_rel_portion.clip(0, 1)
    left_proj_end1_to_x_clipped = left_rel_portion_clipped * left_end1_to_end2
    left_fingers_to_target_disp = left_proj_end1_to_x_clipped - left_end1_to_x
    left_fingertips_to_target_disp = left_fingers_to_target_disp[:, [7, 11, 15, 19]]
    left_fingertips_to_target_dist = left_fingertips_to_target_disp.pow(2).sum(dim=-1).sqrt()
    left_hand_rope_dist = (
        torch.maximum(
            torch.zeros_like(left_fingertips_to_target_dist),
            left_fingertips_to_target_dist - 0.02,
        )
    ).pow(2).max(dim=-1).values

    lhand_reward = torch.exp(
        -left_hand_reaching_scale * left_hand_rope_dist)
    left_hand_reaching_reward = ((1.0 - offset_s) * lhand_reward + offset_s)

    right_foot_goal_local_pos = right_foot_rock_pos - right_foot_pos
    right_foot_goal_local_distance = torch.sum(
        right_foot_goal_local_pos * right_foot_goal_local_pos, dim=-1
    )
    right_foot_reaching_reward = (1.0 - offset_s) * torch.exp(
        -right_foot_reaching_scale * (right_foot_goal_local_distance)
    ) + offset_s

    left_foot_goal_local_pos = left_foot_rock_pos - left_foot_pos
    left_foot_goal_local_distance = torch.sum(
        left_foot_goal_local_pos * left_foot_goal_local_pos, dim=-1
    )
    left_foot_reaching_reward = (1.0 - offset_s) * torch.exp(
        -left_foot_reaching_scale * (left_foot_goal_local_distance)
    ) + offset_s

    # Could try electricity cost
    motor_effort_ratio = motor_efforts / max_motor_effort
    electricity_cost = torch.sum(torch.abs(actions * dof_actuator_forces) * motor_effort_ratio.unsqueeze(0), dim=-1)

    reward = ( right_hand_reaching_reward * left_hand_reaching_reward \
               + right_hand_reaching_reward * right_foot_reaching_reward \
               + right_hand_reaching_reward * left_foot_reaching_reward \
               + left_hand_reaching_reward * right_foot_reaching_reward \
               + left_hand_reaching_reward * left_foot_reaching_reward \
               + right_foot_reaching_reward * left_foot_reaching_reward) / 6.0

    return reward


@torch.jit.script
def compute_humanoid_reset(
        reset_buf,
        progress_buf,
        contact_buf,
        contact_body_ids,
        max_episode_length,
        enable_early_termination,
        termination_height,
        right_hand_pos,
        right_hand_goal_pos,
        left_hand_pos,
        left_hand_goal_pos,
        right_foot_pos,
        right_foot_goal_pos,
        left_foot_pos,
        left_foot_goal_pos,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        right_hand_local_goal_pos = right_hand_goal_pos - right_hand_pos
        right_hand_goal_local_distance = torch.sum(
            right_hand_local_goal_pos * right_hand_local_goal_pos, dim=-1
        )
        left_hand_local_goal_pos = left_hand_goal_pos - left_hand_pos
        left_hand_goal_local_distance = torch.sum(
            left_hand_local_goal_pos * left_hand_local_goal_pos, dim=-1
        )
        right_foot_local_goal_pos = right_foot_goal_pos - right_foot_pos
        right_foot_goal_local_distance = torch.sum(
            right_foot_local_goal_pos * right_foot_local_goal_pos, dim=-1
        )
        left_foot_local_goal_pos = left_foot_goal_pos - left_foot_pos
        left_foot_goal_local_distance = torch.sum(
            left_foot_local_goal_pos * left_foot_local_goal_pos, dim=-1
        )
        fall_right_hand_height = right_hand_goal_local_distance > (termination_height * termination_height)
        fall_left_hand_height = left_hand_goal_local_distance > (termination_height * termination_height)
        fall_right_foot_height = right_foot_goal_local_distance > (termination_height * termination_height)
        fall_left_foot_height = left_foot_goal_local_distance > (termination_height * termination_height)
        fall_hand_height = torch.logical_and(fall_right_hand_height, fall_left_hand_height)
        fall_foot_height = torch.logical_and(fall_right_foot_height, fall_left_foot_height)
        fall_height = torch.logical_or(fall_hand_height, fall_foot_height)
        has_fallen = torch.logical_or(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 5
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated