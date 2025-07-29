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

from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelPMP4SetsIPContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('pmp4setsip', **config)
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)

        return self.Network(net, obs_shape=obs_shape,
                            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)

            if (is_train):
                pmp4setsip_obs_upper = input_dict['pmp4setsip_obs_upper']
                pmp4setsip_obs_lower = input_dict['pmp4setsip_obs_lower']
                pmp4setsip_obs_right_hand = input_dict['pmp4setsip_obs_right_hand']
                pmp4setsip_obs_left_hand = input_dict['pmp4setsip_obs_left_hand']
                pmp4setsip_obs_ip_right_hand = input_dict['pmp4setsip_obs_ip_right_hand']
                pmp4setsip_obs_ip_left_hand = input_dict['pmp4setsip_obs_ip_left_hand']
                (
                    disc_agent_logit_upper,
                    disc_agent_logit_lower,
                    disc_agent_logit_right_hand,
                    disc_agent_logit_left_hand,
                    disc_agent_logit_ip_right_hand,
                    disc_agent_logit_ip_left_hand
                ) = self.a2c_network.eval_discs(
                    pmp4setsip_obs_upper,
                    pmp4setsip_obs_lower,
                    pmp4setsip_obs_right_hand,
                    pmp4setsip_obs_left_hand,
                    pmp4setsip_obs_ip_right_hand,
                    pmp4setsip_obs_ip_left_hand
                )
                result["disc_agent_logit_upper"] = disc_agent_logit_upper
                result["disc_agent_logit_lower"] = disc_agent_logit_lower
                result["disc_agent_logit_right_hand"] = disc_agent_logit_right_hand
                result["disc_agent_logit_left_hand"] = disc_agent_logit_left_hand
                result["disc_agent_logit_ip_right_hand"] = disc_agent_logit_ip_right_hand
                result["disc_agent_logit_ip_left_hand"] = disc_agent_logit_ip_left_hand

                pmp4setsip_obs_replay_upper = input_dict['pmp4setsip_obs_replay_upper']
                pmp4setsip_obs_replay_lower = input_dict['pmp4setsip_obs_replay_lower']
                pmp4setsip_obs_replay_right_hand = input_dict['pmp4setsip_obs_replay_right_hand']
                pmp4setsip_obs_replay_left_hand = input_dict['pmp4setsip_obs_replay_left_hand']
                pmp4setsip_obs_replay_ip_right_hand = input_dict['pmp4setsip_obs_replay_ip_right_hand']
                pmp4setsip_obs_replay_ip_left_hand = input_dict['pmp4setsip_obs_replay_ip_left_hand']
                (
                    disc_agent_replay_logit_upper,
                    disc_agent_replay_logit_lower,
                    disc_agent_replay_logit_right_hand,
                    disc_agent_replay_logit_left_hand,
                    disc_agent_replay_logit_ip_right_hand,
                    disc_agent_replay_logit_ip_left_hand
                ) = self.a2c_network.eval_discs(
                    pmp4setsip_obs_replay_upper,
                    pmp4setsip_obs_replay_lower,
                    pmp4setsip_obs_replay_right_hand,
                    pmp4setsip_obs_replay_left_hand,
                    pmp4setsip_obs_replay_ip_right_hand,
                    pmp4setsip_obs_replay_ip_left_hand
                )
                result["disc_agent_replay_logit_upper"] = disc_agent_replay_logit_upper
                result["disc_agent_replay_logit_lower"] = disc_agent_replay_logit_lower
                result["disc_agent_replay_logit_right_hand"] = disc_agent_replay_logit_right_hand
                result["disc_agent_replay_logit_left_hand"] = disc_agent_replay_logit_left_hand
                result["disc_agent_replay_logit_ip_right_hand"] = disc_agent_replay_logit_ip_right_hand
                result["disc_agent_replay_logit_ip_left_hand"] = disc_agent_replay_logit_ip_left_hand

                pmp4setsip_demo_obs_upper = input_dict['pmp4setsip_obs_demo_upper']
                pmp4setsip_demo_obs_lower = input_dict['pmp4setsip_obs_demo_lower']
                pmp4setsip_demo_obs_right_hand = input_dict['pmp4setsip_obs_demo_right_hand']
                pmp4setsip_demo_obs_left_hand = input_dict['pmp4setsip_obs_demo_left_hand']
                pmp4setsip_demo_obs_ip_right_hand = input_dict['pmp4setsip_obs_demo_ip_right_hand']
                pmp4setsip_demo_obs_ip_left_hand = input_dict['pmp4setsip_obs_demo_ip_left_hand']
                (
                    disc_demo_logit_upper,
                    disc_demo_logit_lower,
                    disc_demo_logit_right_hand,
                    disc_demo_logit_left_hand,
                    disc_demo_logit_ip_right_hand,
                    disc_demo_logit_ip_left_hand
                ) = self.a2c_network.eval_discs(
                    pmp4setsip_demo_obs_upper,
                    pmp4setsip_demo_obs_lower,
                    pmp4setsip_demo_obs_right_hand,
                    pmp4setsip_demo_obs_left_hand,
                    pmp4setsip_demo_obs_ip_right_hand,
                    pmp4setsip_demo_obs_ip_left_hand
                )
                result["disc_demo_logit_upper"] = disc_demo_logit_upper
                result["disc_demo_logit_lower"] = disc_demo_logit_lower
                result["disc_demo_logit_right_hand"] = disc_demo_logit_right_hand
                result["disc_demo_logit_left_hand"] = disc_demo_logit_left_hand
                result["disc_demo_logit_ip_right_hand"] = disc_demo_logit_ip_right_hand
                result["disc_demo_logit_ip_left_hand"] = disc_demo_logit_ip_left_hand

            return result
