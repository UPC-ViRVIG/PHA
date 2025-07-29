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

from isaacgymenvs.utils.torch_jit_utils import to_torch

from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib


class MotionLibIP(MotionLib):    
    def __init__(self, motion_file, device):
        self._device = device
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return
    
    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            ip_motions = torch.load(curr_file)[:,1:].cpu()
            for i in range(ip_motions.shape[0]):
                curr_motion = ip_motions[i]
                motion_fps = 30.0
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                self._motions.append(curr_motion)
                self._motion_lengths.append(curr_len)
            
                curr_weight = motion_weights[f]
                self._motion_weights.append(curr_weight)
                self._motion_files.append(str(i) + "_" + curr_file)


        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return
    
    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            ip_motions = np.load(curr_file)[:-10,:]
            SAVE_NUM = 999990
            SAVE_MAX_HORIZON = 30
            num_motions = SAVE_NUM // SAVE_MAX_HORIZON
            for i in range(num_motions):
                curr_motion = ip_motions[i*SAVE_MAX_HORIZON:(i+1)*SAVE_MAX_HORIZON]
                motion_fps = 30.0
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                self._motions.append(curr_motion)
                self._motion_lengths.append(curr_len)
            
                curr_weight = motion_weights[f]
                self._motion_weights.append(curr_weight)
                self._motion_files.append(str(i) + "_" + curr_file)


        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def get_motion_state_and_actions(self, motion_ids, motion_times):
        n = len(motion_ids)
        motion_states_and_actions = np.empty([n, 87])

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion_states_and_actions = self._motions[uid]
            motion_states_and_actions[ids, :] = curr_motion_states_and_actions[frame_idx[ids]]

        motion_states_and_actions = to_torch(motion_states_and_actions, device=self._device)

        return motion_states_and_actions
    
    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)

        frame_idx = (phase * (num_frames - 1)).astype(int)

        return frame_idx