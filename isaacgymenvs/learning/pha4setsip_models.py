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

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
import torch


class ModelPHA4SetsIPContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('pha4setsip', **config)

        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)

        return self.Network(net,
                            obs_shape=obs_shape,
                            normalize_value=normalize_value,
                            normalize_input=normalize_input,
                            value_size=value_size)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, obs_shape, normalize_value, normalize_input, value_size):
            self.num_actors = 4
            nn.Module.__init__(self)
            self.obs_shape = obs_shape
            self.normalize_value = normalize_value
            self.normalize_input = normalize_input
            self.value_size = value_size

            if normalize_value:
                self.value_mean_std = RunningMeanStd(
                    (self.value_size,))  # GeneralizedMovingStats((self.value_size,)) #
            if normalize_input:
                if isinstance(obs_shape, dict):
                    self.running_mean_std_full_obs = RunningMeanStdObs(obs_shape)
                    self.running_mean_std_right_hand = RunningMeanStdObs(74)
                    self.running_mean_std_left_hand = RunningMeanStdObs(74)
                else:
                    self.running_mean_std_full_obs = RunningMeanStd(obs_shape)
                    self.running_mean_std_right_hand = RunningMeanStd(74)
                    self.running_mean_std_left_hand = RunningMeanStd(74)
            self.a2c_network = a2c_network
            return

        def norm_obs(self, observation):
            with torch.no_grad():
                full_obs = self.running_mean_std_full_obs(observation) if self.normalize_input else observation
                right_hand_obs = self.running_mean_std_right_hand(observation[...,105:105+74]) if self.normalize_input else observation[...,105:105+74]
                left_hand_obs = self.running_mean_std_left_hand(observation[...,105+74:105+74+74]) if self.normalize_input else observation[...,105+74:105+74+74]
                return full_obs, right_hand_obs, left_hand_obs

        def forward(self, input_dict, eval_harl=False):
            is_train = input_dict.get('is_train', True)
            prev_actions = [
                input_dict.get('prev_actions_0', None),
                input_dict.get('prev_actions_1', None),
                input_dict.get('prev_actions_2', None),
                input_dict.get('prev_actions_3', None)
            ]
            mu_list, logstd_list, value = self.a2c_network(input_dict)
            result = {}
            if is_train:
                result['values'] = value
            else:
                result['values'] = self.denorm_value(value)

            sigma_list = []
            distr_list = []
            for actor_id in range(self.num_actors):
                sigma_list.append(torch.exp(logstd_list[actor_id]))
                distr_list.append(torch.distributions.Normal(mu_list[actor_id], sigma_list[actor_id], validate_args=False))
                if is_train:
                    entropy = distr_list[actor_id].entropy().sum(dim=-1)
                    prev_neglogp = self.neglogp(prev_actions[actor_id], mu_list[actor_id], sigma_list[actor_id], logstd_list[actor_id])
                    result[f'prev_neglogp_{actor_id}'] = torch.squeeze(prev_neglogp)
                    result[f'entropy_{actor_id}'] = entropy
                    result[f'mus_{actor_id}'] = mu_list[actor_id]
                    result[f'sigmas_{actor_id}'] = sigma_list[actor_id]
                else:
                    if eval_harl:
                        prev_neglogp = self.neglogp(prev_actions[actor_id], mu_list[actor_id], sigma_list[actor_id], logstd_list[actor_id])
                        result[f'prev_neglogp_{actor_id}'] = torch.squeeze(prev_neglogp)
                    else:
                        selected_action = distr_list[actor_id].sample()
                        neglogp = self.neglogp(selected_action, mu_list[actor_id], sigma_list[actor_id], logstd_list[actor_id])
                        result[f'neglogpacs_{actor_id}'] = torch.squeeze(neglogp)
                        result[f'actions_{actor_id}'] = selected_action
                        result[f'mus_{actor_id}'] = mu_list[actor_id]
                        result[f'sigmas_{actor_id}'] = sigma_list[actor_id]

            if (is_train):
                pmp4setsip_obs_lower = input_dict['pmp4setsip_obs_lower']
                pmp4setsip_obs_upper = input_dict['pmp4setsip_obs_upper']
                pmp4setsip_obs_right_hand = input_dict['pmp4setsip_obs_right_hand']
                pmp4setsip_obs_left_hand = input_dict['pmp4setsip_obs_left_hand']
                pmp4setsip_obs_ip_right_hand = input_dict['pmp4setsip_obs_ip_right_hand']
                pmp4setsip_obs_ip_left_hand = input_dict['pmp4setsip_obs_ip_left_hand']
                (
                    disc_agent_logit_lower,
                    disc_agent_logit_upper,
                    disc_agent_logit_right_hand,
                    disc_agent_logit_left_hand,
                    disc_agent_logit_ip_right_hand,
                    disc_agent_logit_ip_left_hand
                ) = self.a2c_network.eval_discs(
                    pmp4setsip_obs_lower,
                    pmp4setsip_obs_upper,
                    pmp4setsip_obs_right_hand,
                    pmp4setsip_obs_left_hand,
                    pmp4setsip_obs_ip_right_hand,
                    pmp4setsip_obs_ip_left_hand
                )
                result["disc_agent_logit_lower"] = disc_agent_logit_lower
                result["disc_agent_logit_upper"] = disc_agent_logit_upper
                result["disc_agent_logit_right_hand"] = disc_agent_logit_right_hand
                result["disc_agent_logit_left_hand"] = disc_agent_logit_left_hand
                result["disc_agent_logit_ip_right_hand"] = disc_agent_logit_ip_right_hand
                result["disc_agent_logit_ip_left_hand"] = disc_agent_logit_ip_left_hand

                pmp4setsip_obs_replay_lower = input_dict['pmp4setsip_obs_replay_lower']
                pmp4setsip_obs_replay_upper = input_dict['pmp4setsip_obs_replay_upper']
                pmp4setsip_obs_replay_right_hand = input_dict['pmp4setsip_obs_replay_right_hand']
                pmp4setsip_obs_replay_left_hand = input_dict['pmp4setsip_obs_replay_left_hand']
                pmp4setsip_obs_replay_ip_right_hand = input_dict['pmp4setsip_obs_replay_ip_right_hand']
                pmp4setsip_obs_replay_ip_left_hand = input_dict['pmp4setsip_obs_replay_ip_left_hand']
                (
                    disc_agent_replay_logit_lower,
                    disc_agent_replay_logit_upper,
                    disc_agent_replay_logit_right_hand,
                    disc_agent_replay_logit_left_hand,
                    disc_agent_replay_logit_ip_right_hand,
                    disc_agent_replay_logit_ip_left_hand
                ) = self.a2c_network.eval_discs(
                    pmp4setsip_obs_replay_lower,
                    pmp4setsip_obs_replay_upper,
                    pmp4setsip_obs_replay_right_hand,
                    pmp4setsip_obs_replay_left_hand,
                    pmp4setsip_obs_replay_ip_right_hand,
                    pmp4setsip_obs_replay_ip_left_hand
                )
                result["disc_agent_replay_logit_lower"] = disc_agent_replay_logit_lower
                result["disc_agent_replay_logit_upper"] = disc_agent_replay_logit_upper
                result["disc_agent_replay_logit_right_hand"] = disc_agent_replay_logit_right_hand
                result["disc_agent_replay_logit_left_hand"] = disc_agent_replay_logit_left_hand
                result["disc_agent_replay_logit_ip_right_hand"] = disc_agent_replay_logit_ip_right_hand
                result["disc_agent_replay_logit_ip_left_hand"] = disc_agent_replay_logit_ip_left_hand

                pmp4setsip_demo_obs_lower = input_dict['pmp4setsip_obs_demo_lower']
                pmp4setsip_demo_obs_upper = input_dict['pmp4setsip_obs_demo_upper']
                pmp4setsip_demo_obs_right_hand = input_dict['pmp4setsip_obs_demo_right_hand']
                pmp4setsip_demo_obs_left_hand = input_dict['pmp4setsip_obs_demo_left_hand']
                pmp4setsip_demo_obs_ip_right_hand = input_dict['pmp4setsip_obs_demo_ip_right_hand']
                pmp4setsip_demo_obs_ip_left_hand = input_dict['pmp4setsip_obs_demo_ip_left_hand']
                (
                    disc_demo_logit_lower,
                    disc_demo_logit_upper,
                    disc_demo_logit_right_hand,
                    disc_demo_logit_left_hand,
                    disc_demo_logit_ip_right_hand,
                    disc_demo_logit_ip_left_hand
                ) = self.a2c_network.eval_discs(
                    pmp4setsip_demo_obs_lower,
                    pmp4setsip_demo_obs_upper,
                    pmp4setsip_demo_obs_right_hand,
                    pmp4setsip_demo_obs_left_hand,
                    pmp4setsip_demo_obs_ip_right_hand,
                    pmp4setsip_demo_obs_ip_left_hand
                )
                result["disc_demo_logit_lower"] = disc_demo_logit_lower
                result["disc_demo_logit_upper"] = disc_demo_logit_upper
                result["disc_demo_logit_right_hand"] = disc_demo_logit_right_hand
                result["disc_demo_logit_left_hand"] = disc_demo_logit_left_hand
                result["disc_demo_logit_ip_right_hand"] = disc_demo_logit_ip_right_hand
                result["disc_demo_logit_ip_left_hand"] = disc_demo_logit_ip_left_hand

            return result
