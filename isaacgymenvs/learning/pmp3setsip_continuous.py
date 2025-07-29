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

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgymenvs.utils.torch_jit_utils import to_torch

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch
from torch import nn

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_3discs_agent as common_3discs_agent

from tensorboardX import SummaryWriter


class PMP3SetsIPAgent(common_3discs_agent.Common3DiscsAgent):

    def __init__(self, base_name, params):
        self._num_pmp3setsip_obs_steps = params["config"]["numAMPObsSteps"]
        super().__init__(base_name, params)

        if self.normalize_value:
            self.value_mean_std = (
                self.central_value_net.model.value_mean_std
                if self.has_central_value
                else self.model.value_mean_std
            )
        if self._normalize_pmp3setsip_input:
            print(
                f"self._pmp3setsip_observation_space.shape: {self._pmp3setsip_observation_space.shape}"
            )
            self._pmp3setsip_input_mean_std = RunningMeanStd(
                self._pmp3setsip_observation_space.shape
            ).to(self.ppo_device)

        return

    def init_tensors(self):
        super().init_tensors()
        self._build_pmp3setsip_buffers()
        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_pmp3setsip_input:
            self._pmp3setsip_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_pmp3setsip_input:
            self._pmp3setsip_input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_pmp3setsip_input:
            state["pmp3setsip_input_mean_std"] = (
                self._pmp3setsip_input_mean_std.state_dict()
            )
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_pmp3setsip_input:
            self._pmp3setsip_input_mean_std.load_state_dict(
                weights["pmp3setsip_input_mean_std"]
            )
        return

    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs, _ = self._env_reset_done()
            self.experience_buffer.update_data("obses", n, self.obs["obs"])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)
            self.experience_buffer.update_data(
                "pmp3setsip_obs", n, infos["pmp3setsip_obs"]
            )

            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if self.vec_env.env.viewer and (n == (self.horizon_length - 1)):
                self._pmp3setsip_debug(infos)

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]

        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_pmp3setsip_obs = self.experience_buffer.tensor_dict["pmp3setsip_obs"]
        reshaped_mb_pmp3setsip_obs = mb_pmp3setsip_obs.view(
            *mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps, 337
        )
        proc_mb_pmp3setsip_obs = self._preproc_pmp3setsip_obs(mb_pmp3setsip_obs)
        reshaped_proc_mb_pmp3setsip_obs = proc_mb_pmp3setsip_obs.view(
            *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps, 337
        )
        proc_mb_pmp3setsip_obs_upper = torch.cat(
            (
                reshaped_proc_mb_pmp3setsip_obs[..., 13:39],
                reshaped_proc_mb_pmp3setsip_obs[..., 65:79],
            ),
            dim=-1,
        )
        proc_mb_pmp3setsip_obs_upper = proc_mb_pmp3setsip_obs_upper.view(
            *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 40
        )

        proc_mb_pmp3setsip_obs_right_hand = torch.cat(
            (
                reshaped_proc_mb_pmp3setsip_obs[..., 118:147],
                reshaped_proc_mb_pmp3setsip_obs[..., 93:96],
            ),
            dim=-1,
        )
        proc_mb_pmp3setsip_obs_right_hand = proc_mb_pmp3setsip_obs_right_hand.view(
            *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )

        proc_mb_pmp3setsip_obs_left_hand = torch.cat(
            (
                reshaped_proc_mb_pmp3setsip_obs[..., 205:234],
                reshaped_proc_mb_pmp3setsip_obs[..., 96:99],
            ),
            dim=-1,
        )
        proc_mb_pmp3setsip_obs_left_hand = proc_mb_pmp3setsip_obs_left_hand.view(
            *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )

        proc_mb_pmp3setsip_obs_ip_right_hand = torch.cat(
            (reshaped_proc_mb_pmp3setsip_obs[..., 105:192],), dim=-1,
            # (reshaped_proc_mb_pmp3setsip_obs[..., 105:172],), dim=-1,
        )
        proc_mb_pmp3setsip_obs_ip_right_hand = (
            proc_mb_pmp3setsip_obs_ip_right_hand.view(
                *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 87
                # *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 67
            )
        )

        proc_mb_pmp3setsip_obs_ip_left_hand = torch.cat(
            (reshaped_proc_mb_pmp3setsip_obs[..., 192:279],), dim=-1,
            # (reshaped_proc_mb_pmp3setsip_obs[..., 192:259],), dim=-1,
        )
        proc_mb_pmp3setsip_obs_ip_left_hand = proc_mb_pmp3setsip_obs_ip_left_hand.view(
            *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 87
            # *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 67
        )

        right_hand_goal = (reshaped_mb_pmp3setsip_obs[..., 147:150] + reshaped_mb_pmp3setsip_obs[..., 150:153])/2.0
        left_hand_goal = (reshaped_mb_pmp3setsip_obs[..., 234:237] + reshaped_mb_pmp3setsip_obs[..., 237:240])/2.0
        rel_pos_mb_pmp3setsip_obs_ip_right_hand = right_hand_goal
        rel_pos_mb_pmp3setsip_obs_ip_left_hand = left_hand_goal

        pmp3setsip_rewards = self._calc_pmp3setsip_rewards(
            proc_mb_pmp3setsip_obs_upper,
            proc_mb_pmp3setsip_obs_right_hand,
            proc_mb_pmp3setsip_obs_left_hand,
            proc_mb_pmp3setsip_obs_ip_right_hand,
            proc_mb_pmp3setsip_obs_ip_left_hand,
            rel_pos_mb_pmp3setsip_obs_ip_right_hand,
            rel_pos_mb_pmp3setsip_obs_ip_left_hand,
        )
        mb_combined_rewards = self._combine_rewards(mb_rewards, pmp3setsip_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_combined_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            a2c_common.swap_and_flatten01, self.tensor_list
        )
        batch_dict["task_rewards"] = a2c_common.swap_and_flatten01(mb_rewards)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        for k, v in pmp3setsip_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["pmp3setsip_obs"] = batch_dict["pmp3setsip_obs"]
        self.dataset.values_dict["pmp3setsip_obs_demo"] = batch_dict["pmp3setsip_obs_demo"]
        self.dataset.values_dict["pmp3setsip_obs_replay"] = batch_dict["pmp3setsip_obs_replay"]
        return

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get("rnn_masks", None)

        self._update_pmp3setsip_demos()
        num_obs_samples = batch_dict["pmp3setsip_obs"].shape[0]
        pmp3setsip_obs_demo = self._pmp3setsip_obs_demo_buffer.sample(num_obs_samples)[
            "pmp3setsip_obs"
        ]
        batch_dict["pmp3setsip_obs_demo"] = pmp3setsip_obs_demo

        if self._pmp3setsip_replay_buffer.get_total_count() == 0:
            batch_dict["pmp3setsip_obs_replay"] = batch_dict["pmp3setsip_obs"]
        else:
            batch_dict["pmp3setsip_obs_replay"] = self._pmp3setsip_replay_buffer.sample(
                num_obs_samples
            )["pmp3setsip_obs"]

        self.set_train()

        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == "legacy":
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr,
                        self.entropy_coef,
                        self.epoch_num,
                        0,
                        curr_train_info["kl"].item(),
                    )
                    self.update_discs_lr(self.last_lr)
                    self.update_actor_lr(self.last_lr)
                    self.update_critic_lr(self.last_lr)

                if train_info is None:
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info["kl"])

            if self.schedule_type == "standard":
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.update_lr(self.last_lr)

        if self.schedule_type == "standard_epoch":
            self.last_lr, self.entropy_coef = self.scheduler.update(
                self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
            )
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_pmp3setsip_obs(batch_dict["pmp3setsip_obs"])

        train_info["play_time"] = play_time
        train_info["update_time"] = update_time
        train_info["total_time"] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        pmp3setsip_obs = input_dict["pmp3setsip_obs"][0:self._pmp3setsip_minibatch_size]
        pmp3setsip_obs = self._preproc_pmp3setsip_obs(pmp3setsip_obs)

        reshaped_pmp3setsip_obs = pmp3setsip_obs.view(
            *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps, 337
        )

        pmp3setsip_obs_upper = torch.cat(
            (
                reshaped_pmp3setsip_obs[..., 13:39],
                reshaped_pmp3setsip_obs[..., 65:79],
            ),
            dim=-1,
        )
        pmp3setsip_obs_upper = pmp3setsip_obs_upper.view(
            *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 40
        )

        pmp3setsip_obs_right_hand = torch.cat(
            (
                reshaped_pmp3setsip_obs[..., 118:147],
                reshaped_pmp3setsip_obs[..., 93:96],
            ),
            dim=-1,
        )
        pmp3setsip_obs_right_hand = pmp3setsip_obs_right_hand.view(
            *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )

        pmp3setsip_obs_left_hand = torch.cat(
            (
                reshaped_pmp3setsip_obs[..., 205:234],
                reshaped_pmp3setsip_obs[..., 96:99],
            ),
            dim=-1,
        )
        pmp3setsip_obs_left_hand = pmp3setsip_obs_left_hand.view(
            *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )

        pmp3setsip_obs_ip_right_hand = torch.cat(
            (reshaped_pmp3setsip_obs[..., 105:192],), dim=-1
            # (reshaped_pmp3setsip_obs[..., 105:172],), dim=-1
        )
        pmp3setsip_obs_ip_right_hand = pmp3setsip_obs_ip_right_hand.view(
            *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 87
            # *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 67
        )

        pmp3setsip_obs_ip_left_hand = torch.cat(
            (reshaped_pmp3setsip_obs[..., 192:279],), dim=-1
            # (reshaped_pmp3setsip_obs[..., 192:259],), dim=-1
        )
        pmp3setsip_obs_ip_left_hand = pmp3setsip_obs_ip_left_hand.view(
            *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 87
            # *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 67
        )

        pmp3setsip_obs_replay = input_dict["pmp3setsip_obs_replay"][
            0 : self._pmp3setsip_minibatch_size
        ]
        pmp3setsip_obs_replay = self._preproc_pmp3setsip_obs(pmp3setsip_obs_replay)

        reshaped_pmp3setsip_obs_replay = pmp3setsip_obs_replay.view(
            *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps, 337
        )

        pmp3setsip_obs_replay_upper = torch.cat(
            (
                reshaped_pmp3setsip_obs_replay[..., 13:39],
                reshaped_pmp3setsip_obs_replay[..., 65:79],
            ),
            dim=-1,
        )
        pmp3setsip_obs_replay_upper = pmp3setsip_obs_replay_upper.view(
            *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps * 40
        )

        pmp3setsip_obs_replay_right_hand = torch.cat(
            (
                reshaped_pmp3setsip_obs_replay[..., 118:147],
                reshaped_pmp3setsip_obs_replay[..., 93:96],
            ),
            dim=-1,
        )
        pmp3setsip_obs_replay_right_hand = pmp3setsip_obs_replay_right_hand.view(
            *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )

        pmp3setsip_obs_replay_left_hand = torch.cat(
            (
                reshaped_pmp3setsip_obs_replay[..., 205:234],
                reshaped_pmp3setsip_obs_replay[..., 96:99],
            ),
            dim=-1,
        )
        pmp3setsip_obs_replay_left_hand = pmp3setsip_obs_replay_left_hand.view(
            *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )

        pmp3setsip_obs_replay_ip_right_hand = torch.cat(
            (reshaped_pmp3setsip_obs_replay[..., 105:192],), dim=-1
            # (reshaped_pmp3setsip_obs_replay[..., 105:172],), dim=-1
        )
        pmp3setsip_obs_replay_ip_right_hand = pmp3setsip_obs_replay_ip_right_hand.view(
            *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps * 87
            # *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps * 67
        )

        pmp3setsip_obs_replay_ip_left_hand = torch.cat(
            (reshaped_pmp3setsip_obs_replay[..., 192:279],), dim=-1
            # (reshaped_pmp3setsip_obs_replay[..., 192:259], ), dim=-1
        )
        pmp3setsip_obs_replay_ip_left_hand = pmp3setsip_obs_replay_ip_left_hand.view(
            *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps * 87
            # *pmp3setsip_obs_replay.shape[:-1], self._num_pmp3setsip_obs_steps * 67
        )

        pmp3setsip_obs_demo = input_dict["pmp3setsip_obs_demo"][
            0 : self._pmp3setsip_minibatch_size
        ]
        pmp3setsip_obs_demo = self._preproc_pmp3setsip_obs(pmp3setsip_obs_demo)
        # pmp3setsip_obs_demo.requires_grad_(True)

        # Here we use the additional 29+29 features for the right and left hand motion reference discs
        reshaped_pmp3setsip_obs_demo = pmp3setsip_obs_demo.view(
            *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps, 337
        )

        pmp3setsip_obs_demo_upper = torch.cat(
            (
                reshaped_pmp3setsip_obs_demo[..., 13:39],
                reshaped_pmp3setsip_obs_demo[..., 65:79],
            ),
            dim=-1,
        )
        pmp3setsip_obs_demo_upper = pmp3setsip_obs_demo_upper.view(
            *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps * 40
        )
        pmp3setsip_obs_demo_upper.requires_grad_(True)

        pmp3setsip_obs_demo_right_hand = torch.cat(
            (
                reshaped_pmp3setsip_obs_demo[..., 279:308],
                reshaped_pmp3setsip_obs_demo[..., 93:96],
            ),
            dim=-1,
        )
        pmp3setsip_obs_demo_right_hand = pmp3setsip_obs_demo_right_hand.view(
            *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )
        pmp3setsip_obs_demo_right_hand.requires_grad_(True)

        pmp3setsip_obs_demo_left_hand = torch.cat(
            (
                reshaped_pmp3setsip_obs_demo[..., 308:337],
                reshaped_pmp3setsip_obs_demo[..., 96:99],
            ),
            dim=-1,
        )
        pmp3setsip_obs_demo_left_hand = pmp3setsip_obs_demo_left_hand.view(
            *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps * 32
        )
        pmp3setsip_obs_demo_left_hand.requires_grad_(True)

        pmp3setsip_obs_demo_ip_right_hand = torch.cat(
            (reshaped_pmp3setsip_obs_demo[..., 105:192],), dim=-1
            # (reshaped_pmp3setsip_obs_demo[..., 105:172],), dim=-1
        )
        pmp3setsip_obs_demo_ip_right_hand = pmp3setsip_obs_demo_ip_right_hand.view(
            *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps * 87
            # *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps * 67
        )
        pmp3setsip_obs_demo_ip_right_hand.requires_grad_(True)

        pmp3setsip_obs_demo_ip_left_hand = torch.cat(
            (reshaped_pmp3setsip_obs_demo[..., 192:279],), dim=-1
            # (reshaped_pmp3setsip_obs_demo[..., 192:259],), dim=-1
        )
        pmp3setsip_obs_demo_ip_left_hand = pmp3setsip_obs_demo_ip_left_hand.view(
            *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps * 87
            # *pmp3setsip_obs_demo.shape[:-1], self._num_pmp3setsip_obs_steps * 67
        )
        pmp3setsip_obs_demo_ip_left_hand.requires_grad_(True)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
            "pmp3setsip_obs_upper": pmp3setsip_obs_upper,
            "pmp3setsip_obs_right_hand": pmp3setsip_obs_right_hand,
            "pmp3setsip_obs_left_hand": pmp3setsip_obs_left_hand,
            "pmp3setsip_obs_ip_right_hand": pmp3setsip_obs_ip_right_hand,
            "pmp3setsip_obs_ip_left_hand": pmp3setsip_obs_ip_left_hand,
            "pmp3setsip_obs_replay_upper": pmp3setsip_obs_replay_upper,
            "pmp3setsip_obs_replay_right_hand": pmp3setsip_obs_replay_right_hand,
            "pmp3setsip_obs_replay_left_hand": pmp3setsip_obs_replay_left_hand,
            "pmp3setsip_obs_replay_ip_right_hand": pmp3setsip_obs_replay_ip_right_hand,
            "pmp3setsip_obs_replay_ip_left_hand": pmp3setsip_obs_replay_ip_left_hand,
            "pmp3setsip_obs_demo_upper": pmp3setsip_obs_demo_upper,
            "pmp3setsip_obs_demo_right_hand": pmp3setsip_obs_demo_right_hand,
            "pmp3setsip_obs_demo_left_hand": pmp3setsip_obs_demo_left_hand,
            "pmp3setsip_obs_demo_ip_right_hand": pmp3setsip_obs_demo_ip_right_hand,
            "pmp3setsip_obs_demo_ip_left_hand": pmp3setsip_obs_demo_ip_left_hand,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            disc_agent_logit_upper = res_dict["disc_agent_logit_upper"]
            disc_agent_replay_logit_upper = res_dict["disc_agent_replay_logit_upper"]
            disc_demo_logit_upper = res_dict["disc_demo_logit_upper"]
            disc_agent_logit_right_hand = res_dict["disc_agent_logit_right_hand"]
            disc_agent_replay_logit_right_hand = res_dict[
                "disc_agent_replay_logit_right_hand"
            ]
            disc_demo_logit_right_hand = res_dict["disc_demo_logit_right_hand"]
            disc_agent_logit_left_hand = res_dict["disc_agent_logit_left_hand"]
            disc_agent_replay_logit_left_hand = res_dict[
                "disc_agent_replay_logit_left_hand"
            ]
            disc_demo_logit_left_hand = res_dict["disc_demo_logit_left_hand"]
            disc_agent_logit_ip_right_hand = res_dict["disc_agent_logit_ip_right_hand"]
            disc_agent_replay_logit_ip_right_hand = res_dict[
                "disc_agent_replay_logit_ip_right_hand"
            ]
            disc_demo_logit_ip_right_hand = res_dict["disc_demo_logit_ip_right_hand"]
            disc_agent_logit_ip_left_hand = res_dict["disc_agent_logit_ip_left_hand"]
            disc_agent_replay_logit_ip_left_hand = res_dict[
                "disc_agent_replay_logit_ip_left_hand"
            ]
            disc_demo_logit_ip_left_hand = res_dict["disc_demo_logit_ip_left_hand"]

            a_info = self._actor_loss(
                old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip
            )
            a_loss = a_info["actor_loss"]

            c_info = self._critic_loss(
                value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
            )
            c_loss = c_info["critic_loss"]

            b_loss = self.bound_loss(mu)

            losses, sum_mask = torch_ext.apply_masks(
                [
                    a_loss.unsqueeze(1),
                    c_loss,
                    entropy.unsqueeze(1),
                    b_loss.unsqueeze(1),
                ],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            disc_agent_cat_logit_upper = torch.cat(
                [disc_agent_logit_upper, disc_agent_replay_logit_upper], dim=0
            )
            disc_agent_cat_logit_right_hand = torch.cat(
                [disc_agent_logit_right_hand, disc_agent_replay_logit_right_hand], dim=0
            )
            disc_agent_cat_logit_left_hand = torch.cat(
                [disc_agent_logit_left_hand, disc_agent_replay_logit_left_hand], dim=0
            )
            disc_agent_cat_logit_ip_right_hand = torch.cat(
                [disc_agent_logit_ip_right_hand, disc_agent_replay_logit_ip_right_hand],
                dim=0,
            )
            disc_agent_cat_logit_ip_left_hand = torch.cat(
                [disc_agent_logit_ip_left_hand, disc_agent_replay_logit_ip_left_hand],
                dim=0,
            )

            # TODO: Implement this
            set_id = 0
            # print(f'pmp3setsip_obs_demo.shape: {pmp3setsip_obs_demo.shape}')
            disc_info_upper = self._disc_loss(
                disc_agent_cat_logit_upper,
                disc_demo_logit_upper,
                pmp3setsip_obs_demo_upper,
                set_id,
            )
            set_id = 1
            disc_info_right_hand = self._disc_loss(
                disc_agent_cat_logit_right_hand,
                disc_demo_logit_right_hand,
                pmp3setsip_obs_demo_right_hand,
                set_id,
            )
            set_id = 2
            disc_info_left_hand = self._disc_loss(
                disc_agent_cat_logit_left_hand,
                disc_demo_logit_left_hand,
                pmp3setsip_obs_demo_left_hand,
                set_id,
            )
            set_id = 3
            disc_info_ip_right_hand = self._disc_loss(
                disc_agent_cat_logit_ip_right_hand,
                disc_demo_logit_ip_right_hand,
                pmp3setsip_obs_demo_ip_right_hand,
                set_id,
            )
            set_id = 4
            disc_info_ip_left_hand = self._disc_loss(
                disc_agent_cat_logit_ip_left_hand,
                disc_demo_logit_ip_left_hand,
                pmp3setsip_obs_demo_ip_left_hand,
                set_id,
            )

            # disc_loss = (
            #     disc_info_upper["disc_loss"]
            #     + disc_info_right_hand["disc_loss"]
            #     + disc_info_left_hand["disc_loss"]
            #     + disc_info_ip_right_hand["disc_loss"]
            #     + disc_info_ip_left_hand["disc_loss"]
            # ) / 6.0

            # disc_loss = (
            #     disc_info_upper["disc_loss"]
            #     + disc_info_ip_right_hand["disc_loss"]
            #     + disc_info_ip_left_hand["disc_loss"]
            # ) / 4.0

            disc_0_loss = disc_info_upper["disc_loss_0"]
            disc_1_loss = disc_info_right_hand["disc_loss_1"]
            disc_2_loss = disc_info_left_hand["disc_loss_2"]
            disc_3_loss = disc_info_ip_right_hand["disc_loss_3"]
            disc_4_loss = disc_info_ip_left_hand["disc_loss_4"]

            # print(f'disc_info_upper["disc_loss"]: {disc_info_upper["disc_loss"]}')
            # print(f'disc_info_ip_right_hand["disc_loss"]: {disc_info_ip_right_hand["disc_loss"]}')
            # print(f'disc_info_ip_left_hand["disc_loss"]: {disc_info_ip_left_hand["disc_loss"]}')

            actor_loss = (
                a_loss
                + self.bounds_loss_coef * b_loss
            )

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.disc_0_scaler.scale(disc_0_loss).backward()
        # TODO: Refactor this ugliest code of the year
        # TODO: Support truncate grads
        if self.truncate_grads:
            print(f'Truncate grads not supported yet!')
            # if self.multi_gpu:
            #     self.optimizer.synchronize()
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     with self.optimizer.skip_synchronize():
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            # else:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
        else:
            self.disc_0_scaler.step(self.disc_0_optimizer)
            self.disc_0_scaler.update()

        self.disc_1_scaler.scale(disc_1_loss).backward()
        # TODO: Refactor this ugliest code of the year
        # TODO: Support truncate grads
        if self.truncate_grads:
            print(f'Truncate grads not supported yet!')
            # if self.multi_gpu:
            #     self.optimizer.synchronize()
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     with self.optimizer.skip_synchronize():
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            # else:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
        else:
            self.disc_1_scaler.step(self.disc_1_optimizer)
            self.disc_1_scaler.update()

        self.disc_2_scaler.scale(disc_2_loss).backward()
        # TODO: Refactor this ugliest code of the year
        # TODO: Support truncate grads
        if self.truncate_grads:
            print(f'Truncate grads not supported yet!')
            # if self.multi_gpu:
            #     self.optimizer.synchronize()
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     with self.optimizer.skip_synchronize():
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            # else:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
        else:
            self.disc_2_scaler.step(self.disc_2_optimizer)
            self.disc_2_scaler.update()

        self.disc_3_scaler.scale(disc_3_loss).backward()
        # TODO: Refactor this ugliest code of the year
        # TODO: Support truncate grads
        if self.truncate_grads:
            print(f'Truncate grads not supported yet!')
            # if self.multi_gpu:
            #     self.optimizer.synchronize()
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     with self.optimizer.skip_synchronize():
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            # else:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
        else:
            self.disc_3_scaler.step(self.disc_3_optimizer)
            self.disc_3_scaler.update()

        self.disc_4_scaler.scale(disc_4_loss).backward()
        # TODO: Refactor this ugliest code of the year
        # TODO: Support truncate grads
        if self.truncate_grads:
            print(f'Truncate grads not supported yet!')
            # if self.multi_gpu:
            #     self.optimizer.synchronize()
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     with self.optimizer.skip_synchronize():
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            # else:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
        else:
            self.disc_4_scaler.step(self.disc_4_optimizer)
            self.disc_4_scaler.update()

        self.actor_scaler.scale(actor_loss).backward()
        # TODO: Refactor this ugliest code of the year
        # TODO: Support truncate grads
        if self.truncate_grads:
            print(f'Truncate grads not supported yet!')
            # if self.multi_gpu:
            #     self.optimizer.synchronize()
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     with self.optimizer.skip_synchronize():
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            # else:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
        else:
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()

        self.critic_scaler.scale(c_loss).backward()
        # TODO: Refactor this ugliest code of the year
        # TODO: Support truncate grads
        if self.truncate_grads:
            print(f'Truncate grads not supported yet!')
            # if self.multi_gpu:
            #     self.optimizer.synchronize()
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     with self.optimizer.skip_synchronize():
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            # else:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
        else:
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.train_result = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
            "b_loss": b_loss,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info_upper)
        self.train_result.update(disc_info_right_hand)
        self.train_result.update(disc_info_left_hand)
        self.train_result.update(disc_info_ip_right_hand)
        self.train_result.update(disc_info_ip_left_hand)

        return

    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config["task_reward_w"]
        self._disc_reward_w = config["disc_reward_w"]
        self._disc_motion_reference_reward_w = config["disc_motion_reference_reward_w"]
        self._disc_interactive_prior_reward_w = config["disc_interactive_prior_reward_w"]

        self._pmp3setsip_observation_space = self.env_info[
            "pmp3setsip_observation_space"
        ]
        self._pmp3setsip_batch_size = int(config["pmp3setsip_batch_size"])
        self._pmp3setsip_minibatch_size = int(config["pmp3setsip_minibatch_size"])
        assert self._pmp3setsip_minibatch_size <= self.minibatch_size

        self._disc_coef = config["disc_coef"]
        self._disc_logit_reg = config["disc_logit_reg"]
        self._disc_grad_penalty = config["disc_grad_penalty"]
        self._disc_weight_decay = config["disc_weight_decay"]
        self._disc_reward_scale = config["disc_reward_scale"]
        self._normalize_pmp3setsip_input = config.get(
            "normalize_pmp3setsip_input", True
        )
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config["pmp3setsip_input_shape"] = self._pmp3setsip_observation_space.shape
        config["numAMPObsSteps"] = self._num_pmp3setsip_obs_steps
        return config

    def _init_train(self):
        super()._init_train()
        self._init_pmp3setsip_demo_buf()
        return

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo, set_id):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_discs_logit_weights()[set_id]
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        # print(
        #     f'disc_demo_logit.shape: {disc_demo_logit.shape} obs_demo.shape: {obs_demo.shape}')
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if self._disc_weight_decay != 0:
            disc_weights = self.model.a2c_network.get_discs_weights()[set_id]
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(
            disc_agent_logit, disc_demo_logit
        )

        disc_info = {
            f"disc_loss_{set_id}": disc_loss,
            f"disc_grad_penalty_{set_id}": disc_grad_penalty,
            f"disc_logit_loss_{set_id}": disc_logit_loss,
            f"disc_agent_acc_{set_id}": disc_agent_acc,
            f"disc_demo_acc_{set_id}": disc_demo_acc,
            f"disc_agent_logit_{set_id}": disc_agent_logit,
            f"disc_demo_logit_{set_id}": disc_demo_logit,
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_pmp3setsip_obs_demo(self, num_samples):
        pmp3setsip_obs_demo = self.vec_env.env.fetch_pmp3setsip_obs_demo(num_samples)
        return pmp3setsip_obs_demo

    def _build_pmp3setsip_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        print(f"batch_shape: {batch_shape}")
        print(
            f"self._pmp3setsip_observation_space.shape: {self._pmp3setsip_observation_space.shape}"
        )
        self.experience_buffer.tensor_dict["pmp3setsip_obs"] = torch.zeros(
            batch_shape + self._pmp3setsip_observation_space.shape,
            device=self.ppo_device,
        )
        print(
            f'self.experience_buffer.tensor_dict["pmp3setsip_obs"].shape: {self.experience_buffer.tensor_dict["pmp3setsip_obs"].shape}'
        )

        pmp3setsip_obs_demo_buffer_size = int(
            self.config["pmp3setsip_obs_demo_buffer_size"]
        )
        self._pmp3setsip_obs_demo_buffer = replay_buffer.ReplayBuffer(
            pmp3setsip_obs_demo_buffer_size, self.ppo_device
        )

        self._pmp3setsip_replay_keep_prob = self.config["pmp3setsip_replay_keep_prob"]
        replay_buffer_size = int(self.config["pmp3setsip_replay_buffer_size"])
        self._pmp3setsip_replay_buffer = replay_buffer.ReplayBuffer(
            replay_buffer_size, self.ppo_device
        )

        self.tensor_list += ["pmp3setsip_obs"]
        return

    def _init_pmp3setsip_demo_buf(self):
        buffer_size = self._pmp3setsip_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._pmp3setsip_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_pmp3setsip_obs_demo(self._pmp3setsip_batch_size)
            self._pmp3setsip_obs_demo_buffer.store({"pmp3setsip_obs": curr_samples})

        return

    def _update_pmp3setsip_demos(self):
        new_pmp3setsip_obs_demo = self._fetch_pmp3setsip_obs_demo(
            self._pmp3setsip_batch_size
        )
        self._pmp3setsip_obs_demo_buffer.store(
            {"pmp3setsip_obs": new_pmp3setsip_obs_demo}
        )
        return

    def _preproc_pmp3setsip_obs(self, pmp3setsip_obs):
        if self._normalize_pmp3setsip_input:
            pmp3setsip_obs = self._pmp3setsip_input_mean_std(pmp3setsip_obs)
        return pmp3setsip_obs

    def _combine_rewards(self, task_rewards, pmp3setsip_rewards):
        disc_r = pmp3setsip_rewards["disc_rewards"]
        # print(f'task_rewards.shape: {task_rewards.shape}')
        # print(
        #     f'Agent _task_reward_w: {self._task_reward_w} task_rewards[0,95]: {task_rewards[0,95]} _disc_reward_w: {self._disc_reward_w} disc_r[0,95]: {disc_r[0,95]}')
        combined_rewards = (
            self._task_reward_w * task_rewards + +self._disc_reward_w * disc_r
        )
        return combined_rewards

    def _eval_discs(
        self,
        proc_pmp3setsip_obs_upper,
        proc_pmp3setsip_obs_right_hand,
        proc_pmp3setsip_obs_left_hand,
        proc_pmp3setsip_obs_ip_right_hand,
        proc_pmp3setsip_obs_ip_left_hand,
    ):
        return self.model.a2c_network.eval_discs(
            proc_pmp3setsip_obs_upper,
            proc_pmp3setsip_obs_right_hand,
            proc_pmp3setsip_obs_left_hand,
            proc_pmp3setsip_obs_ip_right_hand,
            proc_pmp3setsip_obs_ip_left_hand,
        )

    def _calc_pmp3setsip_rewards(
        self,
        pmp3setsip_obs_upper,
        pmp3setsip_obs_right_hand,
        pmp3setsip_obs_left_hand,
        pmp3setsip_obs_ip_right_hand,
        pmp3setsip_obs_ip_left_hand,
        rel_pos_pmp3setsip_obs_ip_right_hand,
        rel_pos_pmp3setsip_obs_ip_left_hand,
    ):
        (
            disc_r,
            disc_r_upper,
            disc_r_right_hand,
            disc_r_left_hand,
            disc_r_ip_right_hand,
            disc_r_ip_left_hand,
        ) = self._calc_disc_rewards(
            pmp3setsip_obs_upper,
            pmp3setsip_obs_right_hand,
            pmp3setsip_obs_left_hand,
            pmp3setsip_obs_ip_right_hand,
            pmp3setsip_obs_ip_left_hand,
            rel_pos_pmp3setsip_obs_ip_right_hand,
            rel_pos_pmp3setsip_obs_ip_left_hand,
        )
        output = {
            "disc_rewards": disc_r,
            "disc_rewards_0": disc_r_upper,
            "disc_rewards_1": disc_r_right_hand,
            "disc_rewards_2": disc_r_left_hand,
            "disc_rewards_3": disc_r_ip_right_hand,
            "disc_rewards_4": disc_r_ip_left_hand,
        }
        return output

    def _calc_disc_rewards(
        self,
        pmp3setsip_obs_upper,
        pmp3setsip_obs_right_hand,
        pmp3setsip_obs_left_hand,
        pmp3setsip_obs_ip_right_hand,
        pmp3setsip_obs_ip_left_hand,
        rel_pos_pmp3setsip_obs_ip_right_hand,
        rel_pos_pmp3setsip_obs_ip_left_hand,
    ):
        with torch.no_grad():
            (
                disc_logits_upper,
                disc_logits_right_hand,
                disc_logits_left_hand,
                disc_logits_ip_right_hand,
                disc_logits_ip_left_hand,
            ) = self._eval_discs(
                pmp3setsip_obs_upper,
                pmp3setsip_obs_right_hand,
                pmp3setsip_obs_left_hand,
                pmp3setsip_obs_ip_right_hand,
                pmp3setsip_obs_ip_left_hand,
            )
            prob_upper = 1.0 / (1.0 + torch.exp(-disc_logits_upper))
            disc_r_upper = -torch.log(
                torch.maximum(1 - prob_upper, torch.tensor(0.0001, device=self.device))
            ) * self._disc_motion_reference_reward_w

            prob_right_hand = 1.0 / (1.0 + torch.exp(-disc_logits_right_hand))
            disc_r_right_hand = -torch.log(
                torch.maximum(
                    1 - prob_right_hand, torch.tensor(0.0001, device=self.device)
                )
            ) * self._disc_motion_reference_reward_w

            prob_left_hand = 1.0 / (1.0 + torch.exp(-disc_logits_left_hand))
            disc_r_left_hand = -torch.log(
                torch.maximum(
                    1 - prob_left_hand, torch.tensor(0.0001, device=self.device)
                )
            ) * self._disc_motion_reference_reward_w

            prob_ip_right_hand = 1.0 / (1.0 + torch.exp(-disc_logits_ip_right_hand))
            disc_r_ip_right_hand = -torch.log(
                torch.maximum(
                    1 - prob_ip_right_hand, torch.tensor(0.0001, device=self.device)
                )
            ) * self._disc_interactive_prior_reward_w

            prob_ip_left_hand = 1.0 / (1.0 + torch.exp(-disc_logits_ip_left_hand))
            disc_r_ip_left_hand = -torch.log(
                torch.maximum(
                    1 - prob_ip_left_hand, torch.tensor(0.0001, device=self.device)
                )
            ) * self._disc_interactive_prior_reward_w

            sigma_right = self._gaussian_kernel(rel_pos_pmp3setsip_obs_ip_right_hand)
            sigma_left = self._gaussian_kernel(rel_pos_pmp3setsip_obs_ip_left_hand)
            # TODO: Update distance from hand to point to hand to bar
            sigma_right = 1.0
            sigma_left = 1.0

            disc_r = self._disc_reward_scale * (
                disc_r_upper
                * (
                    sigma_right * disc_r_ip_right_hand
                    + (1.0 - sigma_right) * disc_r_right_hand
                )
                + disc_r_upper
                * (
                    sigma_left * disc_r_ip_left_hand
                    + (1.0 - sigma_left) * disc_r_left_hand
                )
                + (
                    sigma_right * disc_r_ip_right_hand
                    + (1.0 - sigma_right) * disc_r_right_hand
                )
                * (
                    sigma_left * disc_r_ip_left_hand
                    + (1.0 - sigma_left) * disc_r_left_hand
                )
            )
            # print(f'disc_r[0,101]: \t\t{disc_r[0,101]}')
            # print(f'disc_r_upper[0,101]: \t{disc_r_upper[0,101]}')
            # print(f'sigma_right[0,101]: \t{sigma_right[0,101]} \tdisc_r_ip_right_hand[0,101]: \t{disc_r_ip_right_hand[0,101]} \tdisc_r_right_hand[0,101]: \t{disc_r_right_hand[0,101]}')
            # print(f'sigma_left[0,101]: \t{sigma_left[0,101]} \tdisc_r_ip_left_hand[0,101]: \t{disc_r_ip_left_hand[0,101]} \tdisc_r_left_hand[0,101]: \t{disc_r_left_hand[0,101]}')
            # print(f'------------------------------------')
            # print(f'sigma_right: {sigma_right.shape} disc_r_right_hand: {disc_r_right_hand.shape} disc_r_ip_right_hand: {disc_r_ip_right_hand.shape} disc_r: {disc_r.shape}')
        return disc_r, disc_r_upper, disc_r_right_hand, disc_r_left_hand, disc_r_ip_right_hand, disc_r_ip_left_hand

    def _gaussian_kernel(self, pmp3setsip_obs_hand_rock_rel_pos):
        gamma = 4000.0
        # print(f'pmp3setsip_obs_hand_rock_rel_pos.shape: {pmp3setsip_obs_hand_rock_rel_pos.shape}')
        # pmp3setsip_obs_hand_rock_rel_pos = pmp3setsip_obs_hand.view(
        #     *pmp3setsip_obs_hand.shape[:-1], self._num_pmp3setsip_obs_steps, 75
        # )[..., 0, 59:62]
        # pmp3setsip_obs_hand_rock_rel_pos = pmp3setsip_obs_hand[..., 0, 59:62]
        distance = torch.sum(pmp3setsip_obs_hand_rock_rel_pos * pmp3setsip_obs_hand_rock_rel_pos, dim=-1)
        # print(f'distance.shape: {distance.shape}')
        distance = torch.min(distance, dim=-1)[0]
        # print(f'distance.shape: {distance.shape}')
        distance = distance ** (1.0 / 2.0)
        distance = distance.unsqueeze(-1)
        # Apply formula
        phi = torch.ones(distance.shape, device=self.device)
        phi = torch.where(
            distance > 0.10, torch.exp(-gamma * (distance - 0.10) ** 3), phi
        )
        # print(f'distance[101]: {distance[:,101]} phi[101]: {phi[:,101]}')
        return phi

    def _store_replay_pmp3setsip_obs(self, pmp3setsip_obs):
        buf_size = self._pmp3setsip_replay_buffer.get_buffer_size()
        buf_total_count = self._pmp3setsip_replay_buffer.get_total_count()
        if buf_total_count > buf_size:
            keep_probs = to_torch(
                np.array([self._pmp3setsip_replay_keep_prob] * pmp3setsip_obs.shape[0]),
                device=self.ppo_device,
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            pmp3setsip_obs = pmp3setsip_obs[keep_mask]

        self._pmp3setsip_replay_buffer.store({"pmp3setsip_obs": pmp3setsip_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        train_info["task_rewards"] = batch_dict["task_rewards"]
        train_info["disc_rewards"] = batch_dict["disc_rewards"]
        train_info["disc_rewards_0"] = batch_dict["disc_rewards_0"]
        train_info["disc_rewards_1"] = batch_dict["disc_rewards_1"]
        train_info["disc_rewards_2"] = batch_dict["disc_rewards_2"]
        train_info["disc_rewards_3"] = batch_dict["disc_rewards_3"]
        train_info["disc_rewards_4"] = batch_dict["disc_rewards_4"]
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        NUM_DISCS = 5
        for id in range(NUM_DISCS):
            self.writer.add_scalar(
                f"losses/disc_loss_{id}",
                torch_ext.mean_list(train_info[f"disc_loss_{id}"]).item(),
                frame,
            )
            self.writer.add_scalar(
                f"info/disc_agent_acc_{id}",
                torch_ext.mean_list(train_info[f"disc_agent_acc_{id}"]).item(),
                frame,
            )
            self.writer.add_scalar(
                f"info/disc_demo_acc_{id}",
                torch_ext.mean_list(train_info[f"disc_demo_acc_{id}"]).item(),
                frame,
            )
            self.writer.add_scalar(
                f"info/disc_agent_logit_{id}",
                torch_ext.mean_list(train_info[f"disc_agent_logit_{id}"]).item(),
                frame,
            )
            self.writer.add_scalar(
                f"info/disc_demo_logit_{id}",
                torch_ext.mean_list(train_info[f"disc_demo_logit_{id}"]).item(),
                frame,
            )
            self.writer.add_scalar(
                f"info/disc_grad_penalty_{id}",
                torch_ext.mean_list(train_info[f"disc_grad_penalty_{id}"]).item(),
                frame,
            )
            self.writer.add_scalar(
                f"info/disc_logit_loss_{id}",
                torch_ext.mean_list(train_info[f"disc_logit_loss_{id}"]).item(),
                frame,
            )
            disc_reward_std, disc_reward_mean = torch.std_mean(train_info[f"disc_rewards_{id}"])
            self.writer.add_scalar(f"info/disc_reward_{id}_mean", disc_reward_mean.item(), frame)
            self.writer.add_scalar(f"info/disc_reward_{id}_std", disc_reward_std.item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info["disc_rewards"])
        self.writer.add_scalar("info/disc_reward_mean", disc_reward_mean.item(), frame)
        self.writer.add_scalar("info/disc_reward_std", disc_reward_std.item(), frame)
        task_reward_std, task_reward_mean = torch.std_mean(train_info["task_rewards"])
        self.writer.add_scalar("info/task_reward_mean", task_reward_mean.item(), frame)
        self.writer.add_scalar("info/task_reward_std", task_reward_std.item(), frame)
        return

    def _pmp3setsip_debug(self, info):
        return
