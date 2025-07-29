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

import torch 

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import isaacgymenvs.learning.common_player as common_player


class PMP3SetsIPPlayerContinuous(common_player.CommonPlayer):

    def __init__(self, params):
        config = params['config']
        self._num_pmp3setsip_obs_steps = params["config"]["numAMPObsSteps"]

        self._normalize_pmp3setsip_input = config.get('normalize_pmp3setsip_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        self._print_disc_prediction = config.get('print_disc_prediction', True)
        
        super().__init__(params)
        return

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_pmp3setsip_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._pmp3setsip_input_mean_std.load_state_dict(checkpoint['pmp3setsip_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_pmp3setsip_input:
            self._pmp3setsip_input_mean_std = RunningMeanStd(config['pmp3setsip_input_shape']).to(self.device)
            self._pmp3setsip_input_mean_std.eval()
        return

    def _post_step(self, info):
        super()._post_step(info)
        if self._print_disc_prediction:
            self._pmp3setsip_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['pmp3setsip_input_shape'] = self.env.pmp3setsip_observation_space.shape
        else:
            config['pmp3setsip_input_shape'] = self.env_info['pmp3setsip_observation_space']

        return config

    def _pmp3setsip_debug(self, info):
        with torch.no_grad():
            pmp3setsip_obs = info['pmp3setsip_obs']
            reshaped_mb_pmp3setsip_obs = pmp3setsip_obs.view(
                *pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps, 337
            )
            proc_mb_pmp3setsip_obs = self._preproc_pmp3setsip_obs(pmp3setsip_obs)
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
            )
            proc_mb_pmp3setsip_obs_ip_right_hand = (
                proc_mb_pmp3setsip_obs_ip_right_hand.view(
                    *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 87
                )
            )

            proc_mb_pmp3setsip_obs_ip_left_hand = torch.cat(
                (reshaped_proc_mb_pmp3setsip_obs[..., 192:279],), dim=-1,
            )
            proc_mb_pmp3setsip_obs_ip_left_hand = proc_mb_pmp3setsip_obs_ip_left_hand.view(
                *proc_mb_pmp3setsip_obs.shape[:-1], self._num_pmp3setsip_obs_steps * 87
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
            disc_reward_upper = pmp3setsip_rewards['disc_rewards_upper']
            disc_reward_right_hand = pmp3setsip_rewards['disc_rewards_right_hand']
            disc_reward_left_hand = pmp3setsip_rewards['disc_rewards_left_hand']
            disc_reward_ip_right_hand = pmp3setsip_rewards['disc_rewards_ip_right_hand']
            disc_reward_ip_left_hand = pmp3setsip_rewards['disc_rewards_ip_left_hand']
            disc_sigma_right = pmp3setsip_rewards['sigma_right']
            disc_sigma_left = pmp3setsip_rewards['sigma_left']

            disc_reward_upper = disc_reward_upper.cpu().numpy()[0, 0]
            disc_reward_right_hand = disc_reward_right_hand.cpu().numpy()[0, 0]
            disc_reward_left_hand = disc_reward_left_hand.cpu().numpy()[0, 0]
            disc_reward_ip_right_hand = disc_reward_ip_right_hand.cpu().numpy()[0, 0]
            disc_reward_ip_left_hand = disc_reward_ip_left_hand.cpu().numpy()[0, 0]
            disc_sigma_right = disc_sigma_right.cpu().numpy()[0, 0]
            disc_sigma_left = disc_sigma_left.cpu().numpy()[0, 0]

            print(f"disc_reward_upper:\t{disc_reward_upper}")
            print(f"disc_reward_right_hand:\t{disc_reward_right_hand}")
            print(f"disc_reward_left_hand:\t{disc_reward_left_hand}")
            print(f"disc_reward_ip_right_hand:\t{disc_reward_ip_right_hand}")
            print(f"disc_reward_ip_left_hand:\t{disc_reward_ip_left_hand}")
            print(f"disc_sigma_right:\t{disc_sigma_right}")
            print(f"disc_sigma_left:\t{disc_sigma_left}")
            print("-----------------------------")
        return

    def _preproc_pmp3setsip_obs(self, pmp3setsip_obs):
        if self._normalize_pmp3setsip_input:
            pmp3setsip_obs = self._pmp3setsip_input_mean_std(pmp3setsip_obs)
        return pmp3setsip_obs

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
            disc_r_upper,
            disc_r_right_hand,
            disc_r_left_hand,
            disc_r_ip_right_hand,
            disc_r_ip_left_hand,
            sigma_right,
            sigma_left,
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
            'disc_rewards_upper': disc_r_upper,
            'disc_rewards_right_hand': disc_r_right_hand,
            'disc_rewards_left_hand': disc_r_left_hand,
            'disc_rewards_ip_right_hand': disc_r_ip_right_hand,
            'disc_rewards_ip_left_hand': disc_r_ip_left_hand,
            'sigma_right': sigma_right,
            'sigma_left': sigma_left,
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
            disc_r_upper = -torch.log(torch.maximum(1 - prob_upper, torch.tensor(0.0001, device=self.device)))
            disc_r_upper *= self._disc_reward_scale

            prob_right_hand = 1.0 / (1.0 + torch.exp(-disc_logits_right_hand))
            disc_r_right_hand = -torch.log(torch.maximum(1 - prob_right_hand, torch.tensor(0.0001, device=self.device)))
            disc_r_right_hand *= self._disc_reward_scale

            prob_left_hand = 1.0 / (1.0 + torch.exp(-disc_logits_left_hand))
            disc_r_left_hand = -torch.log(torch.maximum(1 - prob_left_hand, torch.tensor(0.0001, device=self.device)))
            disc_r_left_hand *= self._disc_reward_scale

            prob_ip_right_hand = 1.0 / (1.0 + torch.exp(-disc_logits_ip_right_hand))
            disc_r_ip_right_hand = -torch.log(torch.maximum(1 - prob_ip_right_hand, torch.tensor(0.0001, device=self.device)))
            disc_r_ip_right_hand *= self._disc_reward_scale

            prob_ip_left_hand = 1.0 / (1.0 + torch.exp(-disc_logits_ip_left_hand))
            disc_r_ip_left_hand = -torch.log(torch.maximum(1 - prob_ip_left_hand, torch.tensor(0.0001, device=self.device)))
            disc_r_ip_left_hand *= self._disc_reward_scale

            sigma_right = self._gaussian_kernel(rel_pos_pmp3setsip_obs_ip_right_hand)
            sigma_left = self._gaussian_kernel(rel_pos_pmp3setsip_obs_ip_left_hand)
        return disc_r_upper, disc_r_right_hand, disc_r_left_hand, disc_r_ip_right_hand, disc_r_ip_left_hand, sigma_right, sigma_left
    
    def _gaussian_kernel(self, pmp3setsip_obs_hand_rock_rel_pos):
        gamma = 4000.0
        distance = torch.sum(pmp3setsip_obs_hand_rock_rel_pos * pmp3setsip_obs_hand_rock_rel_pos, dim=-1)
        distance = torch.min(distance, dim=-1)[0]
        distance = distance ** (1.0 / 2.0)
        distance = distance.unsqueeze(-1)
        phi = torch.ones(distance.shape, device=self.device)
        phi = torch.where(
            distance > 0.10, torch.exp(-gamma * (distance - 0.10) ** 3), phi
        )
        return phi
