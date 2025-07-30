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

import isaacgymenvs.learning.pha4_player as pha4_player


class PHA4SetsIPPlayerContinuous(pha4_player.PHA4Player):

    def __init__(self, params):
        config = params['config']

        self._normalize_pmp4setsip_input = config.get('normalize_pmp4setsip_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        self._print_disc_prediction = config.get('print_disc_prediction', False)
        
        super().__init__(params)
        return

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_pmp4setsip_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._pmp4setsip_input_mean_std.load_state_dict(checkpoint['pmp4setsip_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_pmp4setsip_input:
            self._pmp4setsip_input_mean_std = RunningMeanStd(config['pmp4setsip_input_shape']).to(self.device)
            self._pmp4setsip_input_mean_std.eval()
        return

    def _post_step(self, info):
        super()._post_step(info)
        if self._print_disc_prediction:
            self._pmp4setsip_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['pmp4setsip_input_shape'] = self.env.pmp4setsip_observation_space.shape
        else:
            config['pmp4setsip_input_shape'] = self.env_info['pmp4setsip_observation_space']

        return config

    def _pmp4setsip_debug(self, info):
        with torch.no_grad():
            pmp4setsip_obs = info['pmp4setsip_obs']
            pmp4setsip_obs = pmp4setsip_obs[0:1]
            pmp4setsip_rewards = self._calc_pmp4setsip_rewards(pmp4setsip_obs.to(self.device))
            disc_reward_upper = pmp4setsip_rewards['disc_rewards_upper']
            disc_reward_lower = pmp4setsip_rewards['disc_rewards_lower']
            disc_reward_right_hand = pmp4setsip_rewards['disc_rewards_right_hand']
            disc_reward_left_hand = pmp4setsip_rewards['disc_rewards_left_hand']
            disc_reward_ip_right_hand = pmp4setsip_rewards['disc_rewards_ip_right_hand']
            disc_reward_ip_left_hand = pmp4setsip_rewards['disc_rewards_ip_left_hand']

            disc_reward_upper = disc_reward_upper.cpu().numpy()[0, 0]
            disc_reward_lower = disc_reward_lower.cpu().numpy()[0, 0]
            disc_reward_right_hand = disc_reward_right_hand.cpu().numpy()[0, 0]
            disc_reward_left_hand = disc_reward_left_hand.cpu().numpy()[0, 0]
            disc_reward_ip_right_hand = disc_reward_ip_right_hand.cpu().numpy()[0, 0]
            disc_reward_ip_left_hand = disc_reward_ip_left_hand.cpu().numpy()[0, 0]
            print(f"disc_reward_upper:\t{disc_reward_upper}")
            print(f"disc_reward_lower:\t{disc_reward_lower}")
            print(f"disc_reward_right_hand:\t{disc_reward_right_hand}")
            print(f"disc_reward_left_hand:\t{disc_reward_left_hand}")
            print(f"disc_reward_ip_right_hand:\t{disc_reward_ip_right_hand}")
            print(f"disc_reward_ip_left_hand:\t{disc_reward_ip_left_hand}")
            print("-----------------------------")
        return

    def _preproc_pmp4setsip_obs(self, pmp4setsip_obs):
        if self._normalize_pmp4setsip_input:
            pmp4setsip_obs = self._pmp4setsip_input_mean_std(pmp4setsip_obs)
        return pmp4setsip_obs

    def _eval_discs(self, pmp4setsip_obs):
        proc_pmp4setsip_obs = self._preproc_pmp4setsip_obs(pmp4setsip_obs)
        return self.model.a2c_network.eval_discs(proc_pmp4setsip_obs)

    def _calc_pmp4setsip_rewards(self, pmp4setsip_obs):
        (
            disc_r_upper,
            disc_r_lower,
            disc_r_right_hand,
            disc_r_left_hand,
            disc_r_ip_right_hand,
            disc_r_ip_left_hand
        ) = self._calc_disc_rewards(pmp4setsip_obs)
        output = {
            'disc_rewards_upper': disc_r_upper,
            'disc_rewards_lower': disc_r_lower,
            'disc_rewards_right_hand': disc_r_right_hand,
            'disc_rewards_left_hand': disc_r_left_hand,
            'disc_rewards_ip_right_hand': disc_r_ip_right_hand,
            'disc_rewards_ip_left_hand': disc_r_ip_left_hand,
        }
        return output

    def _calc_disc_rewards(self, pmp4setsip_obs):
        with torch.no_grad():
            (
                disc_logits_upper,
                disc_logits_lower,
                disc_logits_right_hand,
                disc_logits_left_hand,
                disc_logits_ip_right_hand,
                disc_logits_ip_left_hand
            ) = self._eval_discs(pmp4setsip_obs)
            prob_upper = 1.0 / (1.0 + torch.exp(-disc_logits_upper))
            disc_r_upper = -torch.log(torch.maximum(1 - prob_upper, torch.tensor(0.0001, device=self.device)))
            disc_r_upper *= self._disc_reward_scale

            prob_lower = 1.0 / (1.0 + torch.exp(-disc_logits_lower))
            disc_r_lower = -torch.log(torch.maximum(1 - prob_lower, torch.tensor(0.0001, device=self.device)))
            disc_r_lower *= self._disc_reward_scale

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
        return disc_r_upper, disc_r_lower, disc_r_right_hand, disc_r_left_hand, disc_r_ip_right_hand, disc_r_ip_left_hand
