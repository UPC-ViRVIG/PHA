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

from ..poselib.poselib.core.rotation3d import *

from isaacgymenvs.utils.torch_jit_utils import quat_to_exp_map, quat_to_angle_axis, normalize_angle

from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.tasks.amp.humanoid_pmp_base import DOF_BODY_IDS, DOF_OFFSETS


class MotionLibMPL(MotionLib):    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                if body_id == 5: # right_palm
                    joint_q = quat_mul(self.right_palm_quat.expand(local_rot.shape[0], -1), joint_q)
                if body_id == 28: # left_palm
                    joint_q = quat_mul(self.left_palm_quat.expand(local_rot.shape[0], -1), joint_q)
                joint_exp_map = quat_to_exp_map(joint_q)
                if body_id == 28:  # left_palm
                    joint_exp_map[...,1] *= -1 # flip the wrist rotation around y axis for left hand to match with right hand
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                if body_id == 6:  # right_thumb0
                    joint_q = quat_mul(self.right_thumb0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    right_thumb1_offset = -joint_theta * joint_axis[..., 2]
                    joint_theta = joint_theta * joint_axis[..., 1]
                elif body_id == 10:  # right_index0
                    joint_q = quat_mul(self.right_index0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    right_index1_offset = joint_theta * joint_axis[..., 0]
                    joint_theta = joint_theta * joint_axis[..., 2]
                elif body_id == 18:  # right_ring0
                    joint_q = quat_mul(self.right_ring0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    right_ring1_offset = joint_theta * joint_axis[..., 0]
                    joint_theta = -joint_theta * joint_axis[..., 2]
                elif body_id == 22:  # right_pinky0
                    joint_q = quat_mul(self.right_pinky0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    right_pinky1_offset = joint_theta * joint_axis[..., 0]
                    joint_theta = -joint_theta * joint_axis[..., 2]

                elif body_id in [7, 8, 9]:  # right_thumb1, 2, 3
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    joint_theta = -joint_theta * joint_axis[..., 2]
                    if body_id == 7:
                        joint_theta += right_thumb1_offset
                elif body_id in [11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24,
                                 25]:  # right_index/middle/ring/pinky1, 2, 3
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    joint_theta = joint_theta * joint_axis[..., 0]
                    if body_id == 11:
                        joint_theta += right_index1_offset
                    elif body_id == 15:
                        temp_joint_q = local_rot[:, 14]
                        temp_joint_q = quat_mul(self.right_middle0_quat.expand(local_rot.shape[0], -1), temp_joint_q)
                        temp_joint_theta, temp_joint_axis = quat_to_angle_axis(temp_joint_q)
                        right_middle1_offset = temp_joint_theta * temp_joint_axis[..., 0]
                        joint_theta += right_middle1_offset
                    elif body_id == 19:
                        joint_theta += right_ring1_offset
                    elif body_id == 23:
                        joint_theta += right_pinky1_offset


                elif body_id == 27:  # left_lower_arm
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    joint_theta = joint_theta * joint_axis[..., 1]

                elif body_id == 29:  # left_thumb0
                    joint_q = quat_mul(self.left_thumb0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    left_thumb1_offset = joint_theta * joint_axis[..., 2]
                    joint_theta = -joint_theta * joint_axis[..., 1]
                elif body_id == 33:  # left_index0
                    joint_q = quat_mul(self.left_index0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    left_index1_offset = joint_theta * joint_axis[..., 0]
                    joint_theta = -joint_theta * joint_axis[..., 2]
                elif body_id == 41:  # left_ring0
                    joint_q = quat_mul(self.left_ring0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    left_ring1_offset = joint_theta * joint_axis[..., 0]
                    joint_theta = joint_theta * joint_axis[..., 2]
                elif body_id == 45:  # left_pinky
                    joint_q = quat_mul(self.left_pinky0_quat.expand(local_rot.shape[0], -1), joint_q)
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    left_pinky1_offset = joint_theta * joint_axis[..., 0]
                    joint_theta = joint_theta * joint_axis[..., 2]

                elif body_id in [30, 31, 32]:  # left_thumb1, 2, 3
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    joint_theta = joint_theta * joint_axis[..., 2]
                    if body_id == 30:
                        joint_theta += left_thumb1_offset
                elif body_id in [34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48]:  # left_index/middle/ring/pinky1, 2, 3
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    joint_theta = joint_theta * joint_axis[..., 0]
                    if body_id == 34:
                        joint_theta += left_index1_offset
                    elif body_id == 38:
                        temp_joint_q = local_rot[:, 37]
                        temp_joint_q = quat_mul(self.left_middle0_quat.expand(local_rot.shape[0], -1), temp_joint_q)
                        temp_joint_theta, temp_joint_axis = quat_to_angle_axis(temp_joint_q)
                        left_middle1_offset = temp_joint_theta * temp_joint_axis[..., 0]
                        joint_theta += left_middle1_offset
                    elif body_id == 42:
                        joint_theta += left_ring1_offset
                    elif body_id == 46:
                        joint_theta += left_pinky1_offset
                else:
                    joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                    joint_theta = joint_theta * joint_axis[..., 1]  # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS
        dof_x_axis_ids = [11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48]
        dof_y_axis_ids = [4, 6, 27, 29, 50, 53]
        dof_z_axis_ids = [7, 8, 9, 10, 18, 22, 30, 31, 32, 33, 41, 45]

        dof_vel = np.zeros([self._num_dof])

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel.numpy()

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                if (body_id in dof_x_axis_ids):
                    dof_vel[joint_offset] = joint_vel[0] # joint is along x axis
                elif (body_id in dof_y_axis_ids):
                    dof_vel[joint_offset] = joint_vel[1] # joint is along y axis
                elif (body_id in dof_z_axis_ids):
                    dof_vel[joint_offset] = joint_vel[2] # joint is along z axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel