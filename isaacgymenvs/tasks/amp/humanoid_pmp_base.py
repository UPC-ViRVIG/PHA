# Copyright (c) 2018-2023, NVIDIA Corporation
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaacgymenvs.tasks.amp.humanoid_amp_base import HumanoidAMPBase

import os

from isaacgym import gymapi
import torch
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv


DOF_BODY_IDS = [1, 2, 3, 4,
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                    51, 52, 53, 54]
DOF_OFFSETS = [0,  3,  6,  9, 10,
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   25, 26, 27, 28, 29, 30, 31, 32, 35, 36,
                   39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                   51, 52, 53, 54, 55, 56, 57, 58, 61, 62,
                   65, 68, 69, 72]


class HumanoidPMPBase(HumanoidAMPBase):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        self.fingertips_right = [
            "right_thumb3",
            "right_index3",
            "right_middle3",
            "right_ring3",
            "right_pinky3",
        ]
        self.fingertips_left = [
            "left_thumb3",
            "left_index3",
            "left_middle3",
            "left_ring3",
            "left_pinky3",
        ]
        self.fingers_body_right = [
            "right_thumb0",
            "right_thumb1",
            "right_thumb2",
            "right_thumb3",
            "right_index0",
            "right_index1",
            "right_index2",
            "right_index3",
            "right_middle0",
            "right_middle1",
            "right_middle2",
            "right_middle3",
            "right_ring0",
            "right_ring1",
            "right_ring2",
            "right_ring3",
            "right_pinky0",
            "right_pinky1",
            "right_pinky2",
            "right_pinky3",
        ]
        self.fingers_body_left = [
            "left_thumb0",
            "left_thumb1",
            "left_thumb2",
            "left_thumb3",
            "left_index0",
            "left_index1",
            "left_index2",
            "left_index3",
            "left_middle0",
            "left_middle1",
            "left_middle2",
            "left_middle3",
            "left_ring0",
            "left_ring1",
            "left_ring2",
            "left_ring3",
            "left_pinky0",
            "left_pinky1",
            "left_pinky2",
            "left_pinky3",
        ]
        self.hand_parts_right = [
            "right_palm",
            "right_thumb0",
            "right_thumb1",
            "right_thumb2",
            "right_thumb3",
            "right_index0",
            "right_index1",
            "right_index2",
            "right_index3",
            "right_middle0",
            "right_middle1",
            "right_middle2",
            "right_middle3",
            "right_ring0",
            "right_ring1",
            "right_ring2",
            "right_ring3",
            "right_pinky0",
            "right_pinky1",
            "right_pinky2",
            "right_pinky3",
        ]
        self.hand_parts_left = [
            "left_palm",
            "left_thumb0",
            "left_thumb1",
            "left_thumb2",
            "left_thumb3",
            "left_index0",
            "left_index1",
            "left_index2",
            "left_index3",
            "left_middle0",
            "left_middle1",
            "left_middle2",
            "left_middle3",
            "left_ring0",
            "left_ring1",
            "left_ring2",
            "left_ring3",
            "left_pinky0",
            "left_pinky1",
            "left_pinky2",
            "left_pinky3",
        ]
        self.hand_dof_right = [
            "right_wrist_FLEX",
            "right_wrist_PRO",
            "right_wrist_UDEV",
            "right_thumb_ABD",
            "right_thumb_MCP",
            "right_thumb_PIP",
            "right_thumb_DIP",
            "right_index_ABD",
            "right_index_MCP",
            "right_index_PIP",
            "right_index_DIP",
            "right_middle_MCP",
            "right_middle_PIP",
            "right_middle_DIP",
            "right_ring_ABD",
            "right_ring_MCP",
            "right_ring_PIP",
            "right_ring_DIP",
            "right_pinky_ABD",
            "right_pinky_MCP",
            "right_pinky_PIP",
            "right_pinky_DIP",
        ]
        self.hand_dof_left = [
            "left_wrist_FLEX",
            "left_wrist_PRO",
            "left_wrist_UDEV",
            "left_thumb_ABD",
            "left_thumb_MCP",
            "left_thumb_PIP",
            "left_thumb_DIP",
            "left_index_ABD",
            "left_index_MCP",
            "left_index_PIP",
            "left_index_DIP",
            "left_middle_MCP",
            "left_middle_PIP",
            "left_middle_DIP",
            "left_ring_ABD",
            "left_ring_MCP",
            "left_ring_PIP",
            "left_ring_DIP",
            "left_pinky_ABD",
            "left_pinky_MCP",
            "left_pinky_PIP",
            "left_pinky_DIP",
        ]
        self.hand_actions_right = [
            "A_right_wrist_FLEX",
            "A_right_wrist_PRO",
            "A_right_wrist_UDEV",
            "A_right_thumb_ABD",
            "A_right_thumb_MCP",
            "A_right_thumb_PIP",
            "A_right_thumb_DIP",
            "A_right_index_ABD",
            "A_right_index_MCP",
            "A_right_middle_MCP",
            "A_right_ring_MCP",
            "A_right_pinky_ABD",
            "A_right_pinky_MCP",
        ]
        self.hand_actions_left = [
            "A_left_wrist_FLEX",
            "A_left_wrist_PRO",
            "A_left_wrist_UDEV",
            "A_left_thumb_ABD",
            "A_left_thumb_MCP",
            "A_left_thumb_PIP",
            "A_left_thumb_DIP",
            "A_left_index_ABD",
            "A_left_index_MCP",
            "A_left_middle_MCP",
            "A_left_ring_MCP",
            "A_left_pinky_ABD",
            "A_left_pinky_MCP",
        ]
        self.foot_right = [
            "right_foot",
        ]
        self.foot_left = [
            "left_foot",
        ]
        super().__init__(config, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)
        self.KEY_BODY_NAMES = ["right_palm",
                               "left_palm", "right_foot", "left_foot"]
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dof_per_env), dtype=torch.float, device=self.device)

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        self.KEY_BODY_NAMES = ["right_palm",
                               "left_palm", "right_foot", "left_foot"]
        return super()._build_key_body_ids_tensor(env_ptr, actor_handle)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.KEY_BODY_NAMES = ["right_palm",
                               "left_palm", "right_foot", "left_foot"]
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../../../assets')
        asset_file = "mjcf/amp_humanoid.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get(
                "assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # asset_options.fix_base_link = True # Set to true to debug and see object in t-pose
        asset_options.collapse_fixed_joints = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        
        # we re-arrange filter to mitigate contact calculation
        # Collision filter bitmask - shapes A and B only collide if (filterA & filterB) == 0.
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        for k in range(len(rigid_shape_prop)):
            rigid_shape_prop[k].friction = 0.2
            if k in range(0, 8):  # torso / right forearm
                rigid_shape_prop[k].filter = 0  # .... 000000
            elif k in range(8, 58):  # right palm
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(9, 13):  # thumb
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(13, 17):  # index
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(17, 21):  # middle
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(21, 25):  # ring
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(25, 29):  # pinky
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(29, 31):  # left arm / left fore arm
                rigid_shape_prop[k].filter = 0  # .... 000000
            elif k in range(31, 32):  # left palm
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(32, 36):  # thumb
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(36, 40):  # index
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(40, 44):  # middle
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(44, 48):  # ring
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(48, 52):  # pinky
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(52, 58):  # lower body
                rigid_shape_prop[k].filter = 0  # .... 000000
            # contact offset right hand and left hand
            if k in range(8, 29):
                rigid_shape_prop[k].contact_offset = 0.002
                rigid_shape_prop[k].friction = 5.0
            if k in range(31, 52):
                rigid_shape_prop[k].contact_offset = 0.002
                rigid_shape_prop[k].friction = 5.0
            # high friction to feet too
            if k == 54 or k == 57:
                rigid_shape_prop[k].friction = 5.0
            rigid_shape_prop[k].rolling_friction = rigid_shape_prop[k].friction / 100.0
        self.gym.set_asset_rigid_shape_properties(humanoid_asset, rigid_shape_prop)

        self.humanoid_asset = humanoid_asset

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.motor_efforts = motor_efforts

        # create force sensors at the feet
        self._right_hand_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, self.KEY_BODY_NAMES[0])
        self._left_hand_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, self.KEY_BODY_NAMES[1])
        self._right_foot_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, self.KEY_BODY_NAMES[2])
        self._left_foot_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, self.KEY_BODY_NAMES[3])
        sensor_pose = gymapi.Transform()

        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = True
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = True

        self.gym.create_asset_force_sensor(
            humanoid_asset, self._right_hand_idx, sensor_pose, sensor_props)
        self.gym.create_asset_force_sensor(
            humanoid_asset, self._left_hand_idx, sensor_pose, sensor_props)
        self.gym.create_asset_force_sensor(
            humanoid_asset, self._right_foot_idx, sensor_pose, sensor_props)
        self.gym.create_asset_force_sensor(
            humanoid_asset, self._left_foot_idx, sensor_pose, sensor_props)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.humanoid_num_bodies = self.gym.get_asset_rigid_body_count(
            humanoid_asset)
        self.humanoid_num_shapes = self.gym.get_asset_rigid_shape_count(
            humanoid_asset)
        self.humanoid_num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.humanoid_num_actuators = self.gym.get_asset_actuator_count(
            humanoid_asset)
        self.humanoid_num_tendons = self.gym.get_asset_tendon_count(
            humanoid_asset)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = [
            "T_right_index32_cpl",
            "T_right_index21_cpl",
            "T_right_middle32_cpl",
            "T_right_middle21_cpl",
            "T_right_ring32_cpl",
            "T_right_ring21_cpl",
            "T_right_pinky32_cpl",
            "T_right_pinky21_cpl",
            "T_left_index32_cpl",
            "T_left_index21_cpl",
            "T_left_middle32_cpl",
            "T_left_middle21_cpl",
            "T_left_ring32_cpl",
            "T_left_ring21_cpl",
            "T_left_pinky32_cpl",
            "T_left_pinky21_cpl"
        ]
        tendon_props = self.gym.get_asset_tendon_properties(humanoid_asset)

        for i in range(self.humanoid_num_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(humanoid_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(humanoid_asset, tendon_props)

        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(
            humanoid_asset, i) for i in range(self.humanoid_num_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(
            humanoid_asset, name) for name in actuated_dof_names]
        print(f'self.actuated_dof_indices: {self.actuated_dof_indices}')

        humanoid_dof_props = self.gym.get_asset_dof_properties(
            humanoid_asset)

        self.humanoid_dof_lower_limits = []
        self.humanoid_dof_upper_limits = []
        self.humanoid_dof_default_pos = []
        self.humanoid_dof_default_vel = []

        for i in range(self.humanoid_num_dof):
            self.humanoid_dof_lower_limits.append(
                humanoid_dof_props['lower'][i])
            self.humanoid_dof_upper_limits.append(
                humanoid_dof_props['upper'][i])
            self.humanoid_dof_default_pos.append(0.0)
            self.humanoid_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.humanoid_dof_lower_limits = to_torch(
            self.humanoid_dof_lower_limits, device=self.device)
        self.humanoid_dof_upper_limits = to_torch(
            self.humanoid_dof_upper_limits, device=self.device)
        self.humanoid_dof_default_pos = to_torch(
            self.humanoid_dof_default_pos, device=self.device)
        self.humanoid_dof_default_vel = to_torch(
            self.humanoid_dof_default_vel, device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self._build_env(i, env_ptr, humanoid_asset)

            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(
            self.envs[0], self.humanoid_handles[0])
        for j in range(self.humanoid_num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(
            self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(
            self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(
            env_ptr, self.humanoid_handles[0])
        self._contact_body_ids = self._build_contact_body_ids_tensor(
            env_ptr, self.humanoid_handles[0])

        if (self._pd_control):
            self._build_pd_action_offset_scale()

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

        rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_ptr, self.humanoid_handles[0])
        rpalm_rigid_body_com = torch.zeros(
            3, dtype=torch.float, device=self.device
        )
        rfingers_rigid_body_com = torch.zeros(
            (self.fingers_right_handles.shape[0], 3), dtype=torch.float, device=self.device
        )
        lpalm_rigid_body_com = torch.zeros(
            3, dtype=torch.float, device=self.device
        )
        lfingers_rigid_body_com = torch.zeros(
            (self.fingers_left_handles.shape[0], 3), dtype=torch.float, device=self.device
        )
        rpalm_body_index = self.hand_parts_right_handles[0]
        rpalm_rigid_body_com[0] = rigid_body_prop[rpalm_body_index].com.x
        rpalm_rigid_body_com[1] = rigid_body_prop[rpalm_body_index].com.y
        rpalm_rigid_body_com[2] = rigid_body_prop[rpalm_body_index].com.z
        self.rpalm_rigid_body_com = rpalm_rigid_body_com[None].repeat(self.num_envs, 1)
        for k in range(self.fingers_right_handles.shape[0]):
            rfingers_rigid_body_com[k, 0] = rigid_body_prop[k + rpalm_body_index + 1].com.x
            rfingers_rigid_body_com[k, 1] = rigid_body_prop[k + rpalm_body_index + 1].com.y
            rfingers_rigid_body_com[k, 2] = rigid_body_prop[k + rpalm_body_index + 1].com.z
        self.rfingers_rigid_body_com = rfingers_rigid_body_com[None].repeat(self.num_envs, 1, 1)
        lpalm_body_index = self.hand_parts_left_handles[0]
        lpalm_rigid_body_com[0] = rigid_body_prop[lpalm_body_index].com.x
        lpalm_rigid_body_com[1] = rigid_body_prop[lpalm_body_index].com.y
        lpalm_rigid_body_com[2] = rigid_body_prop[lpalm_body_index].com.z
        self.lpalm_rigid_body_com = lpalm_rigid_body_com[None].repeat(self.num_envs, 1)
        for k in range(self.fingers_left_handles.shape[0]):
            lfingers_rigid_body_com[k, 0] = rigid_body_prop[k + lpalm_body_index + 1].com.x
            lfingers_rigid_body_com[k, 1] = rigid_body_prop[k + lpalm_body_index + 1].com.y
            lfingers_rigid_body_com[k, 2] = rigid_body_prop[k + lpalm_body_index + 1].com.z
        self.lfingers_rigid_body_com = rfingers_rigid_body_com[None].repeat(self.num_envs, 1, 1)
        
        # com feet
        rfoot_body_index = self.foot_right_handles[0]
        rfoot_rigid_body_com = torch.zeros(
            3, dtype=torch.float, device=self.device
        )
        rfoot_rigid_body_com[0] = rigid_body_prop[rfoot_body_index].com.x
        rfoot_rigid_body_com[1] = rigid_body_prop[rfoot_body_index].com.y
        rfoot_rigid_body_com[2] = rigid_body_prop[rfoot_body_index].com.z
        self.rfoot_rigid_body_com = rfoot_rigid_body_com[None].repeat(self.num_envs, 1)

        lfoot_body_index = self.foot_left_handles[0]
        lfoot_rigid_body_com = torch.zeros(
            3, dtype=torch.float, device=self.device
        )
        lfoot_rigid_body_com[0] = rigid_body_prop[lfoot_body_index].com.x
        lfoot_rigid_body_com[1] = rigid_body_prop[lfoot_body_index].com.y
        lfoot_rigid_body_com[2] = rigid_body_prop[lfoot_body_index].com.z
        self.lfoot_rigid_body_com = lfoot_rigid_body_com[None].repeat(self.num_envs, 1)

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._humanoid_root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:,
                                                         self._key_body_ids, :]

        obs = compute_humanoid_observations_mpl(root_states, dof_pos, dof_vel,
                                                key_body_pos, self._local_root_obs)
        return obs


#####################################################################
### =========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs_mpl(pose):
    # type: (Tensor) -> Tensor
    dof_obs_size = 102 - 25 - 25 # full body without hands
    # dof_offsets = [abdomen_xyz, neck_xyz, right_shoulder_xyz, right_elbow_y
    # right_wrist_puf, right_thumb_ampd, right_index_ampd, right_middle_mpd,
    # right_ring_ampd, right_pinky_ampd, left_shoulder_xyz, left_elbow_y,
    # left_wrist_puf, left_thumb_ampd, left_index_ampd, left_middle_mpd,
    # left_ring_ampd, left_pinky_ampd, right_hip_xyz, right_knee_x,
    # right_ankle_xyz, left_hip_xyz, left_knee_x, left_ankle_xyz]
    # body_ids = [1, 2, 3, 4,
    # 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
    # 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    # 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
    # 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    # 51, 52, 53, 54]
    dof_offsets = [0,  3,  6,  9, 10,
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   25, 26, 27, 28, 29, 30, 31, 32, 35, 36,
                   39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                   51, 52, 53, 54, 55, 56, 57, 58, 61, 62,
                   65, 68, 69, 72]
    num_joints = len(dof_offsets) - 1
    # Exclude hands
    HANDS_DOF_INDICES = [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                         36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]
        if dof_offset in HANDS_DOF_INDICES:
            continue
        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(
            dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_observations_mpl(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
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

    obs = torch.cat((root_h, root_rot_obs, local_root_vel,
                    local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs
