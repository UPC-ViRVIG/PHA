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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import (
    quat_mul,
    quat_conjugate,
    quat_from_angle_axis,
    to_torch,
    get_axis_params,
    normalize,
    quat_rotate,
    my_quat_rotate,
    exp_map_to_quat,
    quat_to_tan_norm
)

from isaacgymenvs.tasks.base.vec_task import VecTask

DOF_OFFSETS = [
    0,  # 'right_wrist_PRO', 'right_wrist_UDEV', 'right_wrist_FLEX',
    3,  # 'right_thumb_ABD',
    4,  # 'right_thumb_MCP'
    5,  # 'right_thumb_PIP'
    6,  # 'right_thumb_DIP'
    7,  # 'right_index_ABD
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

CAPSULE_HALF_LEN = 0.3

NUM_OBS = 74
NUM_ACTIONS = 10

# AXIAL_FORCES_RATIO (percentage on how frequently axial forces are applied)
AXIAL_FORCES_RATIO = 0.9


class HumanoidRightHandGrabBar(VecTask):

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

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self._pd_control = self.cfg["env"]["pdControl"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get(
                "assetFileNameBlock", self.asset_files_dict["block"]
            )
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get(
                "assetFileNameEgg", self.asset_files_dict["egg"]
            )
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get(
                "assetFileNamePen", self.asset_files_dict["pen"]
            )

        self.up_axis = "z"

        self.fingertips = [
            "right_thumb3",
            "right_index3",
            "right_middle3",
            "right_ring3",
            "right_pinky3",
        ]
        self.num_fingertips = len(self.fingertips)

        self.hand_parts = [
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
        self.num_hand_parts = len(self.hand_parts)

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

        self.cfg["env"]["numObservations"] = NUM_OBS
        self.cfg["env"]["numActions"] = NUM_ACTIONS

        self._contact_bodies = self.cfg["env"]["contactBodies"]

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        self.actions = torch.zeros(self.num_envs, 13).to(self.device)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.hand_num_dof)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self._root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self._hand_root_states = self._root_states.view(self.num_envs, -1, 13)[:, 0]
        self._object_root_states = self._root_states.view(self.num_envs, -1, 13)[:, 1]
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states = self._initial_root_states.view(self.num_envs, -1, 13)
        self._initial_hand_root_states = self._initial_root_states[:, 0]

        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._hand_dof_state = self._dof_state[:, : self.hand_num_dof]
        self._dof_pos = self._dof_state.view(self.num_envs, self.hand_num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.hand_num_dof, 2)[..., 1]
        self._hand_dof_pos = self._dof_pos[:, : self.hand_num_dof]
        self._hand_dof_vel = self._dof_vel[:, : self.hand_num_dof]

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, -1, 13)[..., 0:3]
        self._hand_rigid_body_pos = self._rigid_body_pos[:, : self.hand_num_bodies]
        self.num_bodies = self._rigid_body_state.shape[1]

        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, -1, 3)[...,
                               :self.hand_num_shapes, :]
        self._hand_contact_forces = self._contact_forces[:, : self.hand_num_bodies]

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self.cur_targets = torch.zeros(
            (self.num_envs, self.hand_num_dof), dtype=torch.float, device=self.device
        )

        self.respawn_handle_pos = torch.FloatTensor([7.0000e-01, 7.9678e-07, 9.3000e-01]).to(self.device)
        self.respawn_handle_rot = torch.FloatTensor([0, 0, 0.7071068, 0.7071068]).to(self.device)

        self.global_indices = torch.arange(
            self.num_envs * 3, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.handle_force_limit = [50, 70]
        self.handle_torque_limit = [-20, 20]

        self.handle_end1_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.handle_end2_offset = torch.zeros_like(self.handle_end1_offset)
        self.handle_end1_offset[:, 0] = CAPSULE_HALF_LEN
        self.handle_end2_offset[:, 0] = -CAPSULE_HALF_LEN

        self.is_axial_forces = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.handle_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.handle_forces_mag = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.handle_torques = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.wrist_target_dof_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )  # 0, 2 --> random, 1 --> 0

        self.init_rfingers_rot = (
            torch.FloatTensor(self.cfg["misc"]["init_rfingers_rot"])[None].to(self.device).repeat(self.num_envs, 1, 1)
        )
        self.unit_x = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.unit_x[:, 0] = 1

        self.unit_y = torch.zeros_like(self.unit_x)
        self.unit_y[:, 1] = 1

        self.unit_z = torch.zeros_like(self.unit_x)
        self.unit_z[:, 2] = 1

        self.default_rfingers_facing_dir = torch.stack([-self.unit_z] * 4 + [self.unit_y] * 16, dim=1)

        return

    def update_epoch(self, epoch_num):
        super().update_epoch(epoch_num)
        if self.epoch_num < 2500:
            self.handle_force_limit = [50, 70]
            self.handle_torque_limit = [-20, 20]
        elif self.epoch_num < 5000:
            self.handle_force_limit = [50, 70]
            self.handle_torque_limit = [-20, 20]
        elif self.epoch_num < 7500:
            self.handle_force_limit = [50, 80]
            self.handle_torque_limit = [-23.3, 23.3]
        elif self.epoch_num < 10000:
            self.handle_force_limit = [50, 80]
            self.handle_torque_limit = [-23.3, 23.3]
        elif self.epoch_num < 12500:
            self.handle_force_limit = [50, 90]
            self.handle_torque_limit = [-26.6, 26.6]
        elif self.epoch_num < 15000:
            self.handle_force_limit = [50, 90]
            self.handle_torque_limit = [-26.6, 26.6]
        else:
            self.handle_foce_limite = [50, 100]
            self.handle_torque_limit = [-30, 30]

        return

    def create_sim(self):
        self.up_axis_idx = 2 if self.up_axis == "z" else 1  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        # self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self.compute_observations(env_ids)
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        )
        hand_asset_file = os.path.normpath("mjcf/right_mpl.xml")

        if "asset" in self.cfg["env"]:
            hand_asset_file = os.path.normpath(
                self.cfg["env"]["asset"].get("assetFileName", hand_asset_file)
            )

        # load hand asset
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        hand_asset = self.gym.load_asset(
            self.sim, asset_root, hand_asset_file, asset_options
        )
        # we re-arrange filter to mitigate contact calculation
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(hand_asset)
        for k in range(len(rigid_shape_prop)):
            if k in range(0, 1):  # lower arm
                rigid_shape_prop[k].filter = 0  # .... 000000
            if k in range(1, 22):  # palm
                rigid_shape_prop[k].filter = 1  # .... 000001
                rigid_shape_prop[k].contact_offset = 0.002

        self.gym.set_asset_rigid_shape_properties(hand_asset, rigid_shape_prop)

        actuator_props = self.gym.get_asset_actuator_properties(hand_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.motor_efforts[self.motor_efforts < 1] = 40
        print(f'motor_efforts: {motor_efforts}')

        self.hand_num_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.hand_num_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.hand_num_dof = self.gym.get_asset_dof_count(hand_asset)
        self.hand_num_joints = self.gym.get_asset_joint_count(hand_asset)
        self.hand_num_actuators = self.gym.get_asset_actuator_count(hand_asset)
        self.hand_num_tendons = self.gym.get_asset_tendon_count(hand_asset)
        self.dof_names = self.gym.get_asset_dof_names(hand_asset)

        mcp_names = ["right_index_MCP", "right_middle_MCP", "right_ring_MCP", "right_pinky_MCP"]
        abd_names = ["right_index_ABD", "right_pinky_ABD"]
        thumb_names = ["right_thumb_ABD", "right_thumb_MCP", "right_thumb_PIP", "right_thumb_DIP"]

        mcp_indices = [self.dof_names.index(n) for n in mcp_names]
        abd_indices = [self.dof_names.index(n) for n in abd_names]
        thumb_indices = [self.dof_names.index(n) for n in thumb_names]
        self.mcp_indices = to_torch(mcp_indices, dtype=torch.long, device=self.device)
        self.abd_indices = to_torch(abd_indices, dtype=torch.long, device=self.device)
        self.thumb_indices = to_torch(thumb_indices, dtype=torch.long, device=self.device)

        wrist_names = ["right_wrist_FLEX", "right_wrist_PRO", "right_wrist_UDEV"]
        wrist_idx = [self.gym.find_asset_dof_index(hand_asset, name) for name in wrist_names]
        self.motor_efforts[wrist_idx] = 40

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
        ]
        tendon_props = self.gym.get_asset_tendon_properties(hand_asset)

        for i in range(self.hand_num_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(hand_asset, tendon_props)

        self.actuated_dof_names = [
            self.gym.get_asset_actuator_joint_name(hand_asset, i)
            for i in range(self.hand_num_actuators)
        ]
        self.actuated_dof_indices = [
            self.gym.find_asset_dof_index(hand_asset, name)
            for name in self.actuated_dof_names
        ]

        # get hand dof properties, loaded by Isaac Gym from the MJCF file
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        hand_dof_props["driveMode"] = gymapi.DOF_MODE_POS
        hand_dof_props["effort"][self.actuated_dof_indices] = (
            self.motor_efforts.detach().cpu().numpy()
        )

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []

        self.num_dof = self.hand_num_dof

        for i in range(self.hand_num_dof):
            if hand_dof_props["lower"][i] > hand_dof_props["upper"][i]:
                self.hand_dof_lower_limits.append(hand_dof_props["upper"][i])
                self.hand_dof_upper_limits.append(hand_dof_props["lower"][i])
            else:
                self.hand_dof_lower_limits.append(hand_dof_props["lower"][i])
                self.hand_dof_upper_limits.append(hand_dof_props["upper"][i])

        self.hand_dof_stiffness = to_torch(hand_dof_props["stiffness"], device=self.device)
        self.hand_dof_damping = to_torch(hand_dof_props["damping"], device=self.device)

        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device
        )
        self.hand_dof_lower_limits = to_torch(
            self.hand_dof_lower_limits, device=self.device
        )
        self.hand_dof_upper_limits = to_torch(
            self.hand_dof_upper_limits, device=self.device
        )

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.fingertips
        ]
        self.hand_parts_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.hand_parts
        ]
        self.fingers_right_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.fingers_body_right
        ]

        if self._pd_control:
            self._build_pd_action_offset_scale()

        # create fingertip force sensors
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(
                hand_asset, ft_handle, sensor_pose
            )

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 5000
        object_asset_options.disable_gravity = True
        object_pose = gymapi.Transform(gymapi.Vec3(0, 0, 1.5), gymapi.Quat(0, 0, 0.7071068, 0.7071068))
        object_asset = self.gym.create_capsule(
            self.sim, 0.02, CAPSULE_HALF_LEN, object_asset_options
        )  # radius and half-length
        object_rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(object_asset)
        object_rigid_shape_prop[0].contact_offset = 0.002
        self.gym.set_asset_rigid_shape_properties(object_asset, object_rigid_shape_prop)

        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(*get_axis_params(1.0, self.up_axis_idx))
        hand_start_pose.r = gymapi.Quat(0, 0, 0, 1.0)

        self.start_rotation = torch.tensor(
            [hand_start_pose.r.x, hand_start_pose.r.y, hand_start_pose.r.z, hand_start_pose.r.w], device=self.device
        )

        # compute aggregate size
        max_agg_bodies = self.hand_num_bodies + 1
        max_agg_shapes = self.hand_num_shapes + 1

        self.envs = []
        self.hand_indices = []
        self.object_indices = []

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.fingertips
        ]
        self.hand_parts_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.hand_parts
        ]

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(
                env_ptr, hand_asset, hand_start_pose, "hand", i, -1, 0
            )

            for j in range(self.hand_num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, hand_actor, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.84375,
                    0.7215686274509804,
                    0.6352941176470588)
                )

            if self._pd_control:
                self.gym.set_actor_dof_properties(
                    env_ptr, hand_actor, hand_dof_props
                )

            hand_idx = self.gym.get_actor_index(
                env_ptr, hand_actor, gymapi.DOMAIN_SIM
            )
            self.hand_indices.append(hand_idx)

            # enable DOF force sensors
            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)

            # add object
            object_handle = self.gym.create_actor(
                env_ptr, object_asset, object_pose, "object", i, 0, 0
            )

            self.gym.set_rigid_body_color(
                env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.6863, 0.549)
            )

            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, hand_actor)

        # self.goal_states[:, self.up_axis_idx] -= 0.04

        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.hand_parts_handles = to_torch(
            self.hand_parts_handles, dtype=torch.long, device=self.device
        )
        self.fingers_right_handles = to_torch(
            self.fingers_right_handles, dtype=torch.long, device=self.device
        )

        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )

        rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_ptr, hand_actor)
        rfingers_rigid_body_com = torch.zeros(
            (self.fingers_right_handles.shape[0], 3), dtype=torch.float, device=self.device
        )
        rpalm_body_index = 1
        for k in range(self.fingers_right_handles.shape[0]):
            rfingers_rigid_body_com[k, 0] = rigid_body_prop[k + rpalm_body_index + 1].com.x
            rfingers_rigid_body_com[k, 1] = rigid_body_prop[k + rpalm_body_index + 1].com.y
            rfingers_rigid_body_com[k, 2] = rigid_body_prop[k + rpalm_body_index + 1].com.z
        self.rfingers_rigid_body_com = rfingers_rigid_body_com[None].repeat(self.num_envs, 1, 1)

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1

        lim_low = self.hand_dof_lower_limits.cpu().numpy()
        lim_high = self.hand_dof_upper_limits.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if dof_size == 3:
                if dof_offset == 0:
                    # Right wrist, Left wrist
                    curr_low = lim_low[dof_offset: (dof_offset + dof_size)]
                    curr_high = lim_high[dof_offset: (dof_offset + dof_size)]
                    curr_mid = 0.5 * (curr_high + curr_low)

                    # extend the action range to be a bit beyond the joint limits so that the motors
                    # don't lose their strength as they approach the joint limits
                    curr_scale = 0.5 * (curr_high - curr_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale
                    lim_low[dof_offset: (dof_offset + dof_size)] = curr_low
                    lim_high[dof_offset: (dof_offset + dof_size)] = curr_high
                else:
                    lim_low[dof_offset: (dof_offset + dof_size)] = -np.pi
                    lim_high[dof_offset: (dof_offset + dof_size)] = np.pi

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                if self.dof_names[dof_offset] in ["right_elbow", "left_elbow", "right_knee", "left_knee"]:
                    curr_scale = 0.7 * (curr_high - curr_low)
                else:
                    if dof_offset in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 18, 19]:
                        curr_scale = 0.8 * (curr_high - curr_low)
                    else:
                        curr_scale = 0.7 * (curr_high - curr_low)  # we don't care about it
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def compute_reward(self):
        target_pos = self._object_root_states[:, 0:3]
        target_rot = self._object_root_states[:, 3:7]
        fingers_rot = self._rigid_body_state[:, self.fingers_right_handles, 3:7]
        fingers_rot_flat = fingers_rot.view(-1, 4)
        fingers_pos = self._rigid_body_state[:, self.fingers_right_handles, 0:3] + my_quat_rotate(
            fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
        ).view(fingers_rot.shape[0], -1, 3)

        wrist_dof_pos = self._hand_dof_pos[:, :3]
        wrist_dof_vel = self._hand_dof_vel[:, :3]
        object_linvel = self._object_root_states[:, 7:10]
        object_angvel = self._object_root_states[:, 10:13]
        goal_wrist_pos = self.wrist_target_dof_pos
        actions = self.actions

        handle_end1_offset = self.handle_end1_offset
        handle_end2_offset = self.handle_end2_offset

        self.rew_buf[:] = compute_hand_reward(
            target_pos,
            target_rot,
            handle_end1_offset,
            handle_end2_offset,
            fingers_pos,
            fingers_rot,
            self.init_rfingers_rot.clone(),
            self.default_rfingers_facing_dir.clone(),
            self.fingertip_handles.clone(),
            wrist_dof_pos,
            wrist_dof_vel,
            object_linvel,
            object_angvel,
            goal_wrist_pos,
            self._approx_pd_force.clone(),
            self._contact_forces[:, self.fingers_right_handles].clone(),
            actions
        )

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._hand_contact_forces,
            self._contact_body_ids,
            self._hand_rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_height,
        )
        self._compute_scenario_specific_reset()

        return

    def _compute_scenario_specific_reset(self):
        # enable early termination
        palm_handle_dist = (
            (self._hand_root_states[:, :3] - self._object_root_states[:, 0:3]).pow(2).sum(dim=-1).sqrt()
        )
        reset_idx = palm_handle_dist > 0.7
        self.reset_buf[reset_idx] = 1
        self._terminate_buf[reset_idx] = 1

        if self.reset_buf.sum() == 0:  # execute only when _set_env_state is not called
            num_actors = int(self._root_states.shape[0] / self.num_envs)
            global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32, device=self.device)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(global_indices),
                len(global_indices),
            )

        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def compute_observations(self, env_ids=None):
        fingertips_rot = self._rigid_body_state[:, self.fingertip_handles, 3:7]
        fingertips_rot_flat = fingertips_rot.view(-1, 4)
        fingertips_pos = self._rigid_body_state[:, self.fingertip_handles, 0:3] + my_quat_rotate(
                    fingertips_rot_flat, self.rfingers_rigid_body_com[:,[3,7,11,15,19]].view(-1, 3)
                ).view(fingertips_rot.shape[0], -1, 3)
        hand_parts_pos = self._rigid_body_state[:, self.hand_parts_handles, 0:3]
        hand_parts_rot = self._rigid_body_state[:, self.hand_parts_handles, 3:7]

        right_wrist_pos = hand_parts_pos[:, 0]  # 0 is right_palm
        right_wrist_rot = hand_parts_rot[:, 0]  # 0 is right_palm
        right_dof_pos = self._hand_dof_pos
        right_dof_vel = self._hand_dof_vel
        right_fingers_rot = self._rigid_body_state[:, self.fingers_right_handles, 3:7]
        right_fingers_rot_flat = right_fingers_rot.view(-1, 4)
        right_fingers_pos = self._rigid_body_state[:, self.fingers_right_handles, 0:3] + my_quat_rotate(
                    right_fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
                ).view(right_fingers_rot.shape[0], -1, 3)
        right_fingers_contact = self._contact_forces[:, self.fingers_right_handles]
        object_pos = self._object_root_states[:, 0:3]
        object_rot = self._object_root_states[:, 3:7]

        right_hand_obs = (
            build_pmp4setsip_hand_ip_observations(
                right_wrist_pos,
                right_wrist_rot,
                right_dof_pos,
                right_dof_vel,
                fingertips_pos,
                right_fingers_pos,
                right_fingers_rot,
                right_fingers_contact,
                object_pos,
                object_rot,
                self.init_rfingers_rot.clone(),
                self.default_rfingers_facing_dir.clone()
            )
        )

        if env_ids is None:
            self.obs_buf[:] = right_hand_obs
        else:
            self.obs_buf[env_ids] = right_hand_obs[env_ids]

        return

    def _reset_actors(self, env_ids):
        # Humanoid Hand
        self._hand_root_states[env_ids] = self._initial_hand_root_states[env_ids]
        # hand pos
        self._hand_root_states[env_ids, 0] += 0.41
        self._hand_root_states[env_ids, 2] = 0.98

        init_z_angle = torch.zeros(env_ids.shape[0]).to(self.device).uniform_(20, 40) * np.pi / 180
        init_z_rot = quat_from_angle_axis(
            angle=init_z_angle, axis=torch.FloatTensor([0.0, 0.0, 1.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )

        y_angle = torch.zeros(env_ids.shape[0]).to(self.device).uniform_(-100, -80) * np.pi / 180
        y_rot = quat_from_angle_axis(
            angle=y_angle, axis=torch.FloatTensor([0.0, 1.0, 0.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )

        z_angle = torch.zeros(env_ids.shape[0]).to(self.device).uniform_(-10, 10) * np.pi / 180
        z_rot = quat_from_angle_axis(
            angle=z_angle, axis=torch.FloatTensor([0.0, 0.0, 1.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )
        self._hand_root_states[env_ids, 3:7] = quat_mul(
            z_rot, quat_mul(y_rot, quat_mul(init_z_rot, self._hand_root_states[env_ids, 3:7]))
        )

        palm_pos = my_quat_rotate(
            self._hand_root_states[env_ids, 3:7],
            torch.FloatTensor([0, 0, -0.24]).repeat(env_ids.shape[0], 1).to(self.device),
        )
        disp_by_rand = palm_pos - torch.FloatTensor([0.24, 0, 0]).to(self.device)[None]
        self._hand_root_states[env_ids, :3] -= disp_by_rand

        self._hand_dof_pos[env_ids] = self._pd_action_offset

        # random initialization
        # fingers
        mcp_offset = torch.zeros(env_ids.shape[0], 1).uniform_(-0.8, 0.8).to(self.device)
        mcp_scale = torch.zeros(env_ids.shape[0], 4).uniform_(-0.15, 0.15).to(self.device)
        mcp_actions = mcp_offset + mcp_scale
        abd_actions = torch.zeros(env_ids.shape[0], 2).uniform_(-1.0, 1.0).to(self.device)

        self._hand_dof_pos[env_ids[:, None], self.mcp_indices[None]] += (
                mcp_actions * self._pd_action_scale[self.mcp_indices]
        )
        self._hand_dof_pos[env_ids[:, None], self.abd_indices[None]] += (
                abd_actions * self._pd_action_scale[self.abd_indices]
        )

        # for thumbs
        thumb_abd_actions = torch.zeros(env_ids.shape[0], 1).uniform_(-0.5, 0.5).to(self.device)
        thumb_else_offset = torch.zeros(env_ids.shape[0], 1).uniform_(-0.8, 0.8).to(self.device)
        thumb_else_scale = torch.zeros(env_ids.shape[0], 3).uniform_(-0.15, 0.15).to(self.device)
        thumb_actions = torch.cat([thumb_abd_actions, thumb_else_offset + thumb_else_scale], dim=-1)
        self._hand_dof_pos[env_ids[:, None], self.thumb_indices[None]] += (
                thumb_actions * self._pd_action_scale[self.thumb_indices]
        )

        # random_vel
        self._hand_dof_vel[env_ids] = 0
        self._hand_dof_vel[env_ids[:, None], self.actuated_dof_indices[None]] = (
                torch.zeros_like(self._hand_dof_vel[env_ids[:, None], self.actuated_dof_indices[None]]).uniform_(
                    -0.05, 0.05
                )
                * self._pd_action_scale[self.actuated_dof_indices]
        )

        # wrist
        self._hand_dof_pos[env_ids, 0] = 0
        self._hand_dof_pos[env_ids, 1] = 1.57 - init_z_angle  # 1.57
        self._hand_dof_pos[env_ids, 2] = 0

        self.cur_targets[env_ids, : self.hand_num_dof] = self._pd_action_offset

        # Else states
        self._object_root_states[env_ids, :3] = self.respawn_handle_pos[None]
        self._object_root_states[env_ids, 3:7] = self.respawn_handle_rot[None]
        self._object_root_states[env_ids, 7:] = 0.0

        # handle pos randomize
        handle_y_range = (-0.5, 0.5)
        self._object_root_states[env_ids, 1] = torch.zeros_like(self._object_root_states[env_ids, 1]).uniform_(
            *handle_y_range
        )

        # additional hand rot randomize according to handle
        handle_y_offset = (-0.78, 0) # if MODE == "right" else (0, 0.78)
        wrist_dof_offset = torch.zeros_like(init_z_angle).uniform_(*handle_y_offset)
        self._hand_dof_pos[env_ids, 1] += wrist_dof_offset
        rot_offset = quat_from_angle_axis(
            angle=-wrist_dof_offset, axis=torch.FloatTensor([1.0, 0.0, 0.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )

        new_handle_rot = quat_mul(rot_offset, self._object_root_states[env_ids, 3:7])
        handle_pos_offset = self._object_root_states[env_ids, :3] - self._hand_root_states[env_ids, :3]
        new_handle_pos = self._hand_root_states[env_ids, :3] + my_quat_rotate(rot_offset, handle_pos_offset)
        self._object_root_states[env_ids, :3] = new_handle_pos
        self._object_root_states[env_ids, 3:7] = new_handle_rot

        self.handle_force_limit = [5, 10]
        self.handle_torque_limit = [-5, 5]
        handle_forces_mag = torch.zeros_like(self.handle_forces[env_ids, :1]).uniform_(*self.handle_force_limit)
        handle_forces_mag_dir = torch.bernoulli(torch.ones_like(self.handle_forces[env_ids, :1]) * 0.5) * 2 - 1
        handle_forces_mag = handle_forces_mag * handle_forces_mag_dir
        handle_torques_mag = torch.zeros_like(self.handle_torques[env_ids, :1]).uniform_(*self.handle_torque_limit)

        handle_forces_dir = normalize(torch.zeros_like(self.handle_forces[env_ids]).uniform_(-1, 1))
        handle_torques_dir = normalize(torch.zeros_like(self.handle_torques[env_ids]).uniform_(-1, 1))

        self.handle_forces_mag[env_ids] = handle_forces_mag
        self.handle_forces[env_ids] = handle_forces_mag * handle_forces_dir
        self.handle_torques[env_ids] = handle_torques_mag * handle_torques_dir
        self.is_axial_forces[env_ids] = torch.bernoulli(
            torch.ones_like(self.handle_forces_mag[env_ids, 0]) * AXIAL_FORCES_RATIO
        )

        # wrist_target_dof_pos
        self.wrist_target_dof_pos[env_ids, 0] = 0
        self.wrist_target_dof_pos[env_ids, 1] = 0
        self.wrist_target_dof_pos[env_ids, 2] = 0

        num_actors = int(self._root_states.shape[0] / self.num_envs)
        global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(global_indices),
            len(global_indices),
        )

        if self.num_dof != self.hand_num_dof:
            dof_actor_ids = torch.tensor([0] + self.else_dof_actor_indices, dtype=torch.int64, device=self.device)
            multi_env_ids_int32 = global_indices.view(self.num_envs, -1)[
                env_ids[:, None], dof_actor_ids[None]
            ].flatten()  # 0 is humanoid index
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_int32),
                len(multi_env_ids_int32),
            )
        else:
            multi_env_ids_int32 = global_indices.view(self.num_envs, -1)[env_ids, :1].flatten()
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_int32),
                len(multi_env_ids_int32),
            )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]] = actions.to(self.device).clone()
        self.actions[:, [9, 10, 12]] = self.actions[:, 8, None].clone()
        if self._pd_control:
            self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(self.cur_targets)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
            self._approx_pd_force = torch.abs(
                self.hand_dof_stiffness[None]
                * (self.cur_targets[..., : self.hand_num_dof] - self._hand_dof_pos)
                - self.hand_dof_damping[None] * self._hand_dof_vel
            )

        # apply force
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        axial_force_ids = torch.where(self.is_axial_forces)[0]

        handle_idx = 0  # self.else_rigid_body_names.index("capsule")
        forces[:, self.hand_num_bodies + handle_idx] = (
                self.handle_forces * (1.0 + self.progress_buf / 30)[:, None]
        )
        forces[axial_force_ids, self.hand_num_bodies + handle_idx] = (
                my_quat_rotate(self._object_root_states[axial_force_ids, 3:7], self.unit_x[axial_force_ids])
                * self.handle_forces_mag[axial_force_ids]
                * (1.0 + self.progress_buf[axial_force_ids] / 30)[:, None]
        )
        torques[:, self.hand_num_bodies + handle_idx] = (
                self.handle_torques[:] * (1.0 + self.progress_buf / 30)[:, None]
        )

        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE
        )

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self.compute_observations()
        self.compute_reward()
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        return

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            try:
                assert body_id != -1
            except:
                raise ValueError("%s does not exist" % (body_name))
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        scaled_action = (
                self._pd_action_offset[self.actuated_dof_indices]
                + self._pd_action_scale[self.actuated_dof_indices] * action
        )
        self.cur_targets[:, self.actuated_dof_indices] = scaled_action
        return


#####################################################################
### =========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    target_pos,
    target_rot,
    handle_end1_offset,
    handle_end2_offset,
    fingers_pos,
    fingers_rot,
    init_fingers_rot,
    default_rfingers_facing_dir,
    tip_check_idx,
    wrist_pos,
    wrist_vel,
    object_linvel,
    object_angvel,
    wrist_pos_goal,
    _approx_pd_force,
    fingers_contact,
    actions
):
    ###############################################################
    # # (A) handle reward
    v_rod_square = torch.sum(object_linvel * object_linvel, dim=-1)
    w_rod_square = torch.sum(object_angvel * object_angvel, dim=-1)

    handle_speed_reward = torch.exp(-v_rod_square)
    handle_linear_reward = handle_speed_reward

    handle_ang_speed_reward = torch.exp(-0.1 * w_rod_square)
    handle_ang_reward = handle_ang_speed_reward
    r_rod = 0.3 * handle_linear_reward + 0.7 * handle_ang_reward
    ###############################################################

    ###############################################################
    # # (B) finger reward
    # I should adjust 128 to a number of the size of the rock
    handle_end1_pos = my_quat_rotate(target_rot, handle_end1_offset) + target_pos
    handle_end2_pos = my_quat_rotate(target_rot, handle_end2_offset) + target_pos
    end1_to_x = fingers_pos - handle_end1_pos[:, None]
    end1_to_end2 = (handle_end2_pos - handle_end1_pos)[:, None]
    end1_to_end2_norm = normalize(end1_to_end2)
    proj_end1_to_x = (end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) * end1_to_end2_norm
    rel_portion = (proj_end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) / end1_to_end2.pow(2).sum(
        dim=-1, keepdim=True
    ).sqrt()
    rel_portion_clipped = rel_portion.clip(0, 1)
    proj_end1_to_x_clipped = rel_portion_clipped * end1_to_end2
    fingers_to_target_disp = proj_end1_to_x_clipped - end1_to_x
    fingers_to_target_dist = fingers_to_target_disp.pow(2).sum(dim=-1).sqrt()
    # print(f'fingers_to_target_dist: {fingers_to_target_dist[0]}')
    r_fin = torch.exp(
        -128
        * (torch.maximum(torch.zeros_like(fingers_to_target_dist), fingers_to_target_dist - 0.02))
        .pow(2)
        .max(dim=-1)
        .values
    )
    ###############################################################

    ###############################################################
    # # (C) is_valid_tip_contact / mcp_max_reward
    a_mcp = (actions[:, [3, 4, 8, 9, 10, 12]] + 1.0) / 2.0
    # a_mcp_mean: Tensor [num_envs]
    a_mcp_mean = torch.mean(a_mcp, dim=-1)
    # a_mcp_mean_comp_abs: Tensor [num_envs]
    a_mcp_mean_comp_abs = torch.abs(1.0 - a_mcp_mean)
    # r_mcp: Tensor [num_envs]
    r_mcp = torch.exp(-3 * a_mcp_mean_comp_abs * a_mcp_mean_comp_abs)

    fingers_rot_flat = fingers_rot.view(-1, 4)
    init_fingers_rot_flat = init_fingers_rot.view(-1, 4)
    fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_fingers_rot_flat))
    fingers_facing_dir_flat = my_quat_rotate(
        fingers_rot_delta_flat, default_rfingers_facing_dir.view(-1, 3)
    )
    fingers_facing_dir = fingers_facing_dir_flat.view(-1, 20, 3)
    # Removing lower arm and hand palm indices since only fingers are considered
    tip_check_idx -= 2
    is_valid_tip_contact = torch.all(
        (fingers_contact[:, tip_check_idx[1:]] * fingers_facing_dir[:, tip_check_idx[1:]]).sum(dim=-1) < 0, dim=-1
    ).float()
    fist_dist = (
        torch.maximum(
            torch.zeros_like(fingers_to_target_disp[:, tip_check_idx, 0]),
            0.9
            - (normalize(fingers_to_target_disp[:, tip_check_idx]) * fingers_facing_dir[:, tip_check_idx]).sum(
                dim=-1
            ),
        )
        .max(dim=-1)
        .values
    )
    r_tip = torch.exp(-3 * fist_dist.pow(2))
    ###############################################################

    ##################################################################
    # # (D) wrist reward
    wrist_pos_error_square = torch.mean((wrist_pos_goal - wrist_pos) ** 2, dim=-1)
    wrist_speed_square = torch.mean(wrist_vel ** 2, dim=-1)
    r_wrist = torch.exp(-3 * wrist_pos_error_square) * torch.exp(
        -0.1 * wrist_speed_square
    )
    ##################################################################

    ##################################################################
    # # (E) torque minimizing reward
    energy_idx = [0, 1, 2, 7, 18]  # /with wrist and index, pinky ABD
    torque_energy_reward = torch.exp(-0.01 * _approx_pd_force[:, energy_idx].pow(2).mean(dim=-1))
    ##################################################################

    reward = (
         r_rod * r_fin
         + r_rod * r_mcp
         + r_rod * r_tip
         + r_rod * r_wrist
         + r_fin * r_mcp
         + r_fin * r_tip
         + r_fin * r_wrist
         + r_mcp * r_tip
         + r_mcp * r_wrist
         + r_tip * r_wrist
    ) / 10.0

    return reward


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
        init_fingers_rot,
        default_fingers_facing_dir
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    dof_obs = dof_to_obs(hand_dof_pos)

    ACTUATED_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 18, 19]
    dof_vel = hand_dof_vel[:, ACTUATED_DOF_INDICES]

    wrist_quat_inv = quat_conjugate(hand_wrist_rot)

    # handle position in hand-centric coordinate
    rel_handle_pos = my_quat_rotate(wrist_quat_inv, rope_pos - hand_wrist_pos)

    offset1 = torch.zeros_like(rel_handle_pos)
    offset2 = offset1.clone()
    CAPSULE_HALF_LEN = 0.30
    offset1[:, 0] = CAPSULE_HALF_LEN
    offset2[:, 0] = -CAPSULE_HALF_LEN
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
    fingertips_handles = [7, 11, 15, 19] # index, middle, ring, pinky
    tip_valid_contact_marker = (
            (rel_fingers_contact[:, fingertips_handles] * fingers_facing_dir[:, fingertips_handles]).sum(dim=-1) < 0
    ).float()

    # hand fingertips right pos wrt wrist p_line_e
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

    obs = torch.cat(
        (
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


@torch.jit.script
def compute_humanoid_reset(
        reset_buf,
        progress_buf,
        contact_buf,
        contact_body_ids,
        rigid_body_pos,
        max_episode_length,
        enable_early_termination,
        termination_height,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
