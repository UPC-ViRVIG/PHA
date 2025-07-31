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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class PHA3SetsIPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self._num_pmp3setsip_obs_steps = kwargs.get('numAMPObsSteps', 2)
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            assert (not self.has_cnn, 'cnn not supported')
            assert (not self.has_rnn, 'rnn not supported')
            assert (not self.is_discrete, 'discrete actions not supported')
            assert (not self.is_multi_discrete, 'multidiscrete actions not supported')

            self.num_actors = kwargs.pop('num_agents', 1)
            actor_cnn_list = []
            actor_mlp_list = []
            for i in range(self.num_actors):
                actor_cnn_list.append(nn.Sequential())
                actor_mlp_list.append(nn.Sequential())

            self.actor_cnn_list = nn.ModuleList(actor_cnn_list)
            self.actor_mlp_list = nn.ModuleList(actor_mlp_list)
            self.critic_cnn = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            mlp_input_shape_list = self._calc_input_size(input_shape, self.actor_cnn_list)

            out_size_list = []
            in_mlp_shape_list = mlp_input_shape_list
            if len(self.units) == 0:
                out_size_list = mlp_input_shape_list
            else:
                for i in range(self.num_actors):
                    out_size_list.append(self.units[-1])

            for i in range(self.num_actors):
                mlp_args = {
                    'input_size' : in_mlp_shape_list[i], 
                    'units' : self.units, 
                    'activation' : self.activation, 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                self.actor_mlp_list[i] = self._build_mlp(**mlp_args)

            mlp_args = {
                'input_size': in_mlp_shape_list[0],
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size_list[0], self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_continuous:
                mu_list = []
                mu_act_list = []
                mu_init_list = []
                sigma_act_list = []
                self.sigma_list = []
                sigma_init_list = []
                for i in range(self.num_actors):
                    mu_list.append(torch.nn.Linear(out_size_list[i], actions_num[i]))
                    mu_act_list.append(self.activations_factory.create(self.space_config['mu_activation']))
                    mu_init_list.append(self.init_factory.create(**self.space_config['mu_init']))
                    sigma_act_list.append(self.activations_factory.create(self.space_config['sigma_activation']))
                    sigma_init_list.append(self.init_factory.create(**self.space_config['sigma_init']))
                    if self.fixed_sigma:
                        self.sigma_list.append(nn.Parameter(torch.zeros(actions_num[i], requires_grad=True, dtype=torch.float32), requires_grad=True).to(device='cuda'))
                    else:
                        self.sigma_list.append(torch.nn.Linear(out_size_list[i], actions_num[i]).to(device='cuda'))
                
                self.mu_list = nn.ModuleList(mu_list)
                self.mu_act_list = nn.ModuleList(mu_act_list)
                self.sigma_act_list = nn.ModuleList(sigma_act_list)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                for i in range(self.num_actors):
                    mu_init_list[i](self.mu_list[i].weight)
                    if self.fixed_sigma:
                        sigma_init_list[i](self.sigma_list[i])
                    else:
                        sigma_init_list[i](self.sigma_list[i].weight)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    for i in range(self.num_actors):
                        sigma_init_list[i] = self.init_factory.create(
                            **self.space_config['sigma_init'])
                        self.sigma_list[i] = nn.Parameter(torch.zeros(
                            actions_num[i], requires_grad=False, dtype=torch.float32), requires_grad=False).to(device='cuda')
                        sigma_init_list[i](self.sigma_list[i])

            pmp_upper_body_input_shape = (self._num_pmp3setsip_obs_steps * 40,)
            pmp_right_hand_body_input_shape = (self._num_pmp3setsip_obs_steps * 32,)
            pmp_left_hand_body_input_shape = (self._num_pmp3setsip_obs_steps * 32,)
            pmp_ip_right_hand_body_input_shape = (self._num_pmp3setsip_obs_steps * 87,)
            pmp_ip_left_hand_body_input_shape = (self._num_pmp3setsip_obs_steps * 87,)
            self._build_discs(
                pmp_upper_body_input_shape,
                pmp_right_hand_body_input_shape,
                pmp_left_hand_body_input_shape,
                pmp_ip_right_hand_body_input_shape,
                pmp_ip_left_hand_body_input_shape
            )

            return
        
        def _calc_input_size(self, input_shape,cnn_layers_list=None):
            if cnn_layers_list is None:
                assert(len(input_shape) == 1)
                return input_shape[0]
            else:
                out_input_shape = [torch.tensor([i], device='cuda') for i in range(self.num_actors)]
                for i in range(self.num_actors):
                    actor_input_shape = input_shape
                    # TODO: Should be more flexible. actors 1 and 2 are hands. 74 is policy prior obs
                    if (i is 1 or i is 2):
                        actor_input_shape = (74,)
                    out_input_shape[i] = nn.Sequential(*cnn_layers_list[i])(torch.rand(1, *(actor_input_shape))).flatten(1).data.size(1)
                return out_input_shape
        
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            obs_right_hand = obs_dict['obs_right_hand']
            obs_left_hand = obs_dict['obs_left_hand']

            if self.separate:
                c_out = obs
                mu_list = []
                sigma_list = []
                for actor_id in range(self.num_actors):
                    a_out = obs
                    # TODO: Should be more flexible. 1 for right and 2 for left hand
                    if(actor_id is 1):
                        a_out = obs_right_hand
                    elif (actor_id is 2):
                        a_out = obs_left_hand
                    curr_a_out = self.actor_cnn_list[actor_id](a_out)
                    curr_a_out = curr_a_out.contiguous().view(curr_a_out.size(0), -1)
                    curr_a_out = self.actor_mlp_list[actor_id](curr_a_out)
                    if self.is_continuous:
                        mu_list.append(self.mu_act_list[actor_id](self.mu_list[actor_id](curr_a_out)))
                        if self.fixed_sigma:
                            sigma_list.append(mu_list[actor_id] * 0.0 + self.sigma_act_list[actor_id](self.sigma_list[actor_id]))
                        else:
                            sigma_list.append(self.sigma_act_list[actor_id](self.sigma_list[actor_id](curr_a_out)))

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)
                c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                
                return mu_list, sigma_list, value

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

        def eval_discs(
                self,
                pmp_obs_upper,
                pmp_obs_right_hand,
                pmp_obs_left_hand,
                pmp_obs_ip_right_hand,
                pmp_obs_ip_left_hand,
            ):
            disc_mlp_upper_out = self._disc_mlp_upper(pmp_obs_upper)
            disc_logits_upper = self._disc_logits_upper(disc_mlp_upper_out)

            disc_mlp_right_hand_out = self._disc_mlp_right_hand(pmp_obs_right_hand)
            disc_logits_right_hand = self._disc_logits_right_hand(disc_mlp_right_hand_out)

            disc_mlp_left_hand_out = self._disc_mlp_left_hand(pmp_obs_left_hand)
            disc_logits_left_hand = self._disc_logits_left_hand(disc_mlp_left_hand_out)

            disc_mlp_ip_right_hand_out = self._disc_mlp_ip_right_hand(pmp_obs_ip_right_hand)
            disc_logits_ip_right_hand = self._disc_logits_ip_right_hand(disc_mlp_ip_right_hand_out)

            disc_mlp_ip_left_hand_out = self._disc_mlp_ip_left_hand(pmp_obs_ip_left_hand)
            disc_logits_ip_left_hand = self._disc_logits_ip_left_hand(disc_mlp_ip_left_hand_out)

            return (
                disc_logits_upper,
                disc_logits_right_hand,
                disc_logits_left_hand,
                disc_logits_ip_right_hand,
                disc_logits_ip_left_hand
            )

        def get_discs_logit_weights(self):
            return (
                torch.flatten(self._disc_logits_upper.weight),
                torch.flatten(self._disc_logits_right_hand.weight),
                torch.flatten(self._disc_logits_left_hand.weight),
                torch.flatten(self._disc_logits_ip_right_hand.weight),
                torch.flatten(self._disc_logits_ip_left_hand.weight)
            )

        def get_discs_weights(self):
            weights_upper = []
            for m in self._disc_mlp_upper.modules():
                if isinstance(m, nn.Linear):
                    weights_upper.append(torch.flatten(m.weight))

            weights_upper.append(torch.flatten(self._disc_logits_upper.weight))

            weights_right_hand = []
            for m in self._disc_mlp_right_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_right_hand.append(torch.flatten(m.weight))

            weights_right_hand.append(torch.flatten(self._disc_logits_right_hand.weight))

            weights_left_hand = []
            for m in self._disc_mlp_left_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_left_hand.append(torch.flatten(m.weight))

            weights_left_hand.append(torch.flatten(self._disc_logits_left_hand.weight))

            weights_ip_right_hand = []
            for m in self._disc_mlp_ip_right_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_ip_right_hand.append(torch.flatten(m.weight))

            weights_ip_right_hand.append(torch.flatten(self._disc_logits_ip_right_hand.weight))

            weights_ip_left_hand = []
            for m in self._disc_mlp_ip_left_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_ip_left_hand.append(torch.flatten(m.weight))

            weights_ip_left_hand.append(torch.flatten(self._disc_logits_ip_left_hand.weight))

            return (
                weights_upper,
                weights_right_hand,
                weights_left_hand,
                weights_ip_right_hand,
                weights_ip_left_hand
            )

        def _build_discs(
                self,
                input_shape_upper_body,
                input_shape_right_hand_body,
                input_shape_left_hand_body,
                input_shape_ip_right_hand_body,
                input_shape_ip_left_hand_body
            ):
            self._disc_mlp_upper = nn.Sequential()
            self._disc_mlp_right_hand = nn.Sequential()
            self._disc_mlp_left_hand = nn.Sequential()
            self._disc_mlp_ip_right_hand = nn.Sequential()
            self._disc_mlp_ip_left_hand = nn.Sequential()

            mlp_upper_args = {
                'input_size': input_shape_upper_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_right_hand_args = {
                'input_size': input_shape_right_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_left_hand_args = {
                'input_size': input_shape_left_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_ip_right_hand_args = {
                'input_size': input_shape_ip_right_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_ip_left_hand_args = {
                'input_size': input_shape_ip_left_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }

            self._disc_mlp_upper = self._build_mlp(**mlp_upper_args)
            self._disc_mlp_right_hand = self._build_mlp(**mlp_right_hand_args)
            self._disc_mlp_left_hand = self._build_mlp(**mlp_left_hand_args)
            self._disc_mlp_ip_right_hand = self._build_mlp(**mlp_ip_right_hand_args)
            self._disc_mlp_ip_left_hand = self._build_mlp(**mlp_ip_left_hand_args)

            mlp_out_size = self._disc_units[-1]
            self._disc_logits_upper = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_right_hand = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_left_hand = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_ip_right_hand = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_ip_left_hand = torch.nn.Linear(mlp_out_size, 1)

            mlp_init_upper = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_upper.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_upper(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            mlp_init_right_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_right_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_right_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            mlp_init_left_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_left_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_left_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            mlp_init_ip_right_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_ip_right_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_ip_right_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            mlp_init_ip_left_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_ip_left_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_ip_left_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            torch.nn.init.uniform_(
                self._disc_logits_upper.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_upper.bias)

            torch.nn.init.uniform_(
                self._disc_logits_right_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_right_hand.bias)

            torch.nn.init.uniform_(
                self._disc_logits_left_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_left_hand.bias)

            torch.nn.init.uniform_(
                self._disc_logits_ip_right_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_ip_right_hand.bias)

            torch.nn.init.uniform_(
                self._disc_logits_ip_left_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_ip_left_hand.bias)

            return

    def build(self, name, **kwargs):
        net = PHA3SetsIPBuilder.Network(self.params, **kwargs)
        return net
