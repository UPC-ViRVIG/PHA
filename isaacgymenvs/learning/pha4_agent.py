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

from datetime import datetime
import os

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch import  model_builder
from rl_games.common import schedulers
from rl_games.common import vecenv
from rl_games.common import common_losses
from rl_games.common.diagnostics import DefaultDiagnostics
import gym

import torch
from torch import optim

from . import amp_datasets as amp_datasets
from .marl4_experience import MARL4ExperienceBuffer

from tensorboardX import SummaryWriter


class PHA4Agent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, params):
        self.config = config = params['config']
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'

        # This helps in PBT when we need to restart an experiment with the exact same name, rather than
        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.config = config
        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.load_networks(params)

        self.multi_gpu = config.get('multi_gpu', False)

        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        self.curr_frames = 0

        if self.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group("nccl", rank=self.global_rank, world_size=self.world_size)

            self.device_name = 'cuda:' + str(self.local_rank)
            config['device'] = self.device_name
            if self.global_rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.global_rank == 0:
            self.diagnostics = PpoDiagnostics()
        else:
            self.diagnostics = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.vec_env = None
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = config.get('vec_env', None)

        self.ppo_device = config.get('device', 'cuda:0')
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)

        if self.has_central_value:
            self.state_space = self.env_info.get('state_space', None)
            if isinstance(self.state_space,gym.spaces.Dict):
                self.state_shape = {}
                for k,v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            if self.max_epochs == -1 and self.max_frames == -1:
                print("Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one.")
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.max_epochs

                if self.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.max_frames

                self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']

        # seq_length is used only with rnn policy and value functions
        if 'seq_len' in config:
            print('WARNING: seq_len is deprecated, use seq_length instead')

        self.seq_length = self.config.get('seq_length', 4)
        self.bptt_len = self.config.get('bptt_length', self.seq_length) # not used right now. Didn't show that it is usefull
        self.zero_rnn_on_done = self.config.get('zero_rnn_on_done', True)

        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        print('current training device:', self.ppo_device)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_length # it is used only for current rnn implementation

        self.batch_size = self.horizon_length * self.num_actors
        self.batch_size_envs = self.horizon_length * self.num_actors

        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)

        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.mini_epochs_num = self.config['mini_epochs']

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.disc_0_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.disc_1_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.disc_2_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.disc_3_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.disc_4_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.disc_5_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.actor_0_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.actor_1_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.actor_2_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.actor_3_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.critic_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.global_rank == 0:
            writer = SummaryWriter(self.summaries_dir)
            if self.population_based_training:
                self.writer = IntervalSummaryWriter(writer, self.config)
            else:
                self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')
        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5)
            self.advantage_mean_std = GeneralizedMovingStats((1,), momentum=momentum).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        #self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug

        config = params['config']
        self._load_config_params(config)

        self.is_discrete = False
        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.clip_actions = config.get('clip_actions', True)

        self.network_path = self.nn_dir
        
        net_config = self._build_net_config()
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.seq_len = config['seq_len']

        self.disc_0_optimizer = optim.Adam(
            list(self.model.a2c_network._disc_mlp_lower.parameters()) +
            list(self.model.a2c_network._disc_logits_lower.parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.disc_1_optimizer = optim.Adam(
            list(self.model.a2c_network._disc_mlp_upper.parameters()) +
            list(self.model.a2c_network._disc_logits_upper.parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.disc_2_optimizer = optim.Adam(
            list(self.model.a2c_network._disc_mlp_right_hand.parameters()) +
            list(self.model.a2c_network._disc_logits_right_hand.parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.disc_3_optimizer = optim.Adam(
            list(self.model.a2c_network._disc_mlp_left_hand.parameters()) +
            list(self.model.a2c_network._disc_logits_left_hand.parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.disc_4_optimizer = optim.Adam(
            list(self.model.a2c_network._disc_mlp_ip_right_hand.parameters()) +
            list(self.model.a2c_network._disc_logits_ip_right_hand.parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.disc_5_optimizer = optim.Adam(
            list(self.model.a2c_network._disc_mlp_ip_left_hand.parameters()) +
            list(self.model.a2c_network._disc_logits_ip_left_hand.parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.actor_0_optimizer = optim.Adam(
            list(self.model.a2c_network.actor_mlp_list[0].parameters()) +
            list(self.model.a2c_network.mu_list[0].parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.actor_1_optimizer = optim.Adam(
            list(self.model.a2c_network.actor_mlp_list[1].parameters()) +
            list(self.model.a2c_network.mu_list[1].parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.actor_2_optimizer = optim.Adam(
            list(self.model.a2c_network.actor_mlp_list[2].parameters()) +
            list(self.model.a2c_network.mu_list[2].parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.actor_3_optimizer = optim.Adam(
            list(self.model.a2c_network.actor_mlp_list[3].parameters()) +
            list(self.model.a2c_network.mu_list[3].parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )
        self.critic_optimizer = optim.Adam(
            list(self.model.a2c_network.critic_mlp.parameters()) +
            list(self.model.a2c_network.value.parameters()),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay
        )

        if self.has_central_value:
            cv_config = {
                'state_shape' : torch_ext.shape_whc_to_cwh(self.state_shape),
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device,
                'num_agents' : self.num_agents,
                'num_steps' : self.horizon_length,
                'num_actors' : self.num_actors,
                'num_actions' : self.actions_num,
                'seq_len' : self.seq_len,
                'model' : self.central_value_config['network'],
                'config' : self.central_value_config,
                'writter' : self.writer,
                'multi_gpu' : self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)

        self._policy_prior_checkpoint = config.get('policyPriorCheckpoint', None)
        
        return
    
    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs["obs"])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            # Normalize obs only once and not once per actor
            (input_dict['obs'],input_dict['obs_right_hand'],input_dict['obs_left_hand']) = self.model.norm_obs(input_dict['obs'])
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)
        has_central_value_net = self.config.get('central_value_config') is not  None
        if has_central_value_net:
            print('Adding Central Value Network')
            if 'model' not in params['config']['central_value_config']:
                params['config']['central_value_config']['model'] = {'name': 'central_value'}
            network = builder.load(params['config']['central_value_config'])
            self.config['central_value_config']['network'] = network

    def init_tensors(self):
        batch_size = self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = MARL4ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_length
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
            self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]
            
        self.update_list = ['actions_0', 'neglogpacs_0', 'mus_0', 'sigmas_0', 'values',
                            'actions_1', 'neglogpacs_1', 'mus_1', 'sigmas_1',
                            'actions_2', 'neglogpacs_2', 'mus_2', 'sigmas_2',
                            'actions_3', 'neglogpacs_3', 'mus_3', 'sigmas_3']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.tensor_list += ['next_obses']
        return

    def schedule_env(self):
        self.vec_env.env.update_epoch(self.epoch_num)
        return
    
    def train(self):
        print(f'_policy_prior_checkpoint: {self._policy_prior_checkpoint}')
        if (self._policy_prior_checkpoint):
            policy_prior_checkpoint = torch_ext.load_checkpoint(self._policy_prior_checkpoint)
            self.set_policy_prior_weights(policy_prior_checkpoint)
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs


        self.model_output_file = os.path.join(self.network_path, 
            self.config['name'] + '_{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now()))

        self._init_train()

        # global rank of the GPU
        # multi-gpu training is not currently supported for AMP
        self.global_rank = int(os.getenv("RANK", "0"))

        while True:
            epoch_num = self.update_epoch()
            self.schedule_env()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame

            if self.global_rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                self._log_train_info(train_info, frame)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if self.save_freq > 0:
                    if (epoch_num % self.save_freq == 0):
                        self.save(self.model_output_file + "_" + str(epoch_num))

                if epoch_num > self.max_epochs:
                    self.save(self.model_output_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num

                update_time = 0
        return
    
    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['disc_0_optimizer'] = self.disc_0_optimizer.state_dict()
        state['disc_1_optimizer'] = self.disc_1_optimizer.state_dict()
        state['disc_2_optimizer'] = self.disc_2_optimizer.state_dict()
        state['disc_3_optimizer'] = self.disc_3_optimizer.state_dict()
        state['disc_4_optimizer'] = self.disc_4_optimizer.state_dict()
        state['disc_5_optimizer'] = self.disc_5_optimizer.state_dict()
        state['actor_0_optimizer'] = self.actor_0_optimizer.state_dict()
        state['actor_1_optimizer'] = self.actor_1_optimizer.state_dict()
        state['actor_2_optimizer'] = self.actor_2_optimizer.state_dict()
        state['actor_3_optimizer'] = self.actor_3_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state
    
    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        self.disc_0_optimizer.load_state_dict(weights['disc_0_optimizer'])
        self.disc_1_optimizer.load_state_dict(weights['disc_1_optimizer'])
        self.disc_2_optimizer.load_state_dict(weights['disc_2_optimizer'])
        self.disc_3_optimizer.load_state_dict(weights['disc_3_optimizer'])
        self.disc_4_optimizer.load_state_dict(weights['disc_4_optimizer'])
        self.disc_5_optimizer.load_state_dict(weights['disc_5_optimizer'])
        self.actor_0_optimizer.load_state_dict(weights['actor_0_optimizer'])
        self.actor_1_optimizer.load_state_dict(weights['actor_1_optimizer'])
        self.actor_2_optimizer.load_state_dict(weights['actor_2_optimizer'])
        self.actor_3_optimizer.load_state_dict(weights['actor_3_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def set_policy_prior_weights(self, weights):
        model_dict = self.model.state_dict()
        # 2. overwrite entries in the existing state dict
        # Right hand
        model_dict['a2c_network.actor_mlp_list.2.0.weight'] = weights['model']['a2c_network.actor_mlp.0.weight']
        model_dict['a2c_network.actor_mlp_list.2.0.bias'] = weights['model']['a2c_network.actor_mlp.0.bias']
        model_dict['a2c_network.actor_mlp_list.2.2.weight'] = weights['model']['a2c_network.actor_mlp.2.weight']
        model_dict['a2c_network.actor_mlp_list.2.2.bias'] = weights['model']['a2c_network.actor_mlp.2.bias']
        model_dict['a2c_network.mu_list.2.weight'] = weights['model']['a2c_network.mu.weight']
        model_dict['a2c_network.mu_list.2.bias'] = weights['model']['a2c_network.mu.bias']
        model_dict['running_mean_std_right_hand.running_mean'] = weights['model']['running_mean_std.running_mean']
        model_dict['running_mean_std_right_hand.running_var'] = weights['model']['running_mean_std.running_var']
        model_dict['running_mean_std_right_hand.count'] = weights['model']['running_mean_std.count']
        # Left hand
        model_dict['a2c_network.actor_mlp_list.3.0.weight'] = weights['model']['a2c_network.actor_mlp.0.weight']
        model_dict['a2c_network.actor_mlp_list.3.0.bias'] = weights['model']['a2c_network.actor_mlp.0.bias']
        model_dict['a2c_network.actor_mlp_list.3.2.weight'] = weights['model']['a2c_network.actor_mlp.2.weight']
        model_dict['a2c_network.actor_mlp_list.3.2.bias'] = weights['model']['a2c_network.actor_mlp.2.bias']
        model_dict['a2c_network.mu_list.3.weight'] = weights['model']['a2c_network.mu.weight']
        model_dict['a2c_network.mu_list.3.bias'] = weights['model']['a2c_network.mu.bias']
        model_dict['running_mean_std_left_hand.running_mean'] = weights['model']['running_mean_std.running_mean']
        model_dict['running_mean_std_left_hand.running_var'] = weights['model']['running_mean_std.running_var']
        model_dict['running_mean_std_left_hand.count'] = weights['model']['running_mean_std.count']
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

        # Optimizer too
        self.actor_2_optimizer.load_state_dict(weights['actor_optimizer'])
        self.actor_3_optimizer.load_state_dict(weights['actor_optimizer'])

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device=self.ppo_device))**2
            mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device=self.ppo_device))**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
            'num_agents': self.num_agents
        }
        return config

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = self.config['actions_num']

        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        return

    def _init_train(self):
        return

    def _env_reset_done(self):
        obs, done_env_ids = self.vec_env.reset_done()
        return self.obs_to_tensors(obs), done_env_ids

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict['obs']

        processed_obs = self._preproc_obs(obs)
        if self.normalize_input:
            processed_obs = self.model.norm_obs(processed_obs)[0] # full obs normalization
        value = self.model.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _actor_loss(
        self,
        old_action_log_probs_batch,
        action_log_probs,
        advantage,
        curr_e_clip,
        id,
        new_action_log_probs_0 = None,
        action_log_probs_0 = None,
        new_action_log_probs_1 = None,
        action_log_probs_1 = None,
        new_action_log_probs_2 = None,
        action_log_probs_2 = None
    ):
        clip_frac = None
        if new_action_log_probs_2 != None:
            ratio_harl = torch.exp(action_log_probs_0 - new_action_log_probs_0 + action_log_probs_1 - new_action_log_probs_1 + action_log_probs_2 - new_action_log_probs_2)
        elif new_action_log_probs_1 != None:
            ratio_harl = torch.exp(action_log_probs_0 - new_action_log_probs_0 + action_log_probs_1 - new_action_log_probs_1)
        elif new_action_log_probs_0 != None and action_log_probs_0 != None:
            ratio_harl = torch.exp(action_log_probs_0 - new_action_log_probs_0)
        else:
            ratio_harl = 1.0

        if (self.ppo):
            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)

            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                    1.0 + curr_e_clip)
            a_loss = ratio_harl * torch.max(-surr1, -surr2)

            clipped = torch.abs(ratio - 1.0) > curr_e_clip
            clip_frac = torch.mean(clipped.float())
            clip_frac = clip_frac.detach()
        else:
            a_loss = (action_log_probs * advantage * ratio_harl)
    
        info = {
            f'actor_loss_{id}': a_loss,
            f'actor_clip_frac_{id}': clip_frac
        }
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {
            'critic_loss': c_loss
        }
        return info
    
    def update_discs_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.disc_0_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.disc_1_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.disc_2_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.disc_3_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.disc_4_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.disc_5_optimizer.param_groups:
            param_group['lr'] = lr
    
    def update_actor_0_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.actor_0_optimizer.param_groups:
            param_group['lr'] = lr
    
    def update_actor_1_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.actor_1_optimizer.param_groups:
            param_group['lr'] = lr
    
    def update_actor_2_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.actor_2_optimizer.param_groups:
            param_group['lr'] = lr
    
    def update_actor_3_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.actor_3_optimizer.param_groups:
            param_group['lr'] = lr
    
    def update_critic_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr

    
    def _record_train_batch_info(self, batch_dict, train_info):
        return

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/a_loss_0', torch_ext.mean_list(train_info['actor_loss_0']).item(), frame)
        self.writer.add_scalar('losses/a_loss_1', torch_ext.mean_list(train_info['actor_loss_1']).item(), frame)
        self.writer.add_scalar('losses/a_loss_2', torch_ext.mean_list(train_info['actor_loss_2']).item(), frame)
        self.writer.add_scalar('losses/a_loss_3', torch_ext.mean_list(train_info['actor_loss_3']).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(train_info['critic_loss']).item(), frame)
        
        self.writer.add_scalar('losses/bounds_loss_0', torch_ext.mean_list(train_info['b_loss_0']).item(), frame)
        self.writer.add_scalar('losses/bounds_loss_1', torch_ext.mean_list(train_info['b_loss_1']).item(), frame)
        self.writer.add_scalar('losses/bounds_loss_2', torch_ext.mean_list(train_info['b_loss_2']).item(), frame)
        self.writer.add_scalar('losses/bounds_loss_3', torch_ext.mean_list(train_info['b_loss_3']).item(), frame)
        self.writer.add_scalar('losses/entropy_0', torch_ext.mean_list(train_info['entropy_0']).item(), frame)
        self.writer.add_scalar('losses/entropy_1', torch_ext.mean_list(train_info['entropy_1']).item(), frame)
        self.writer.add_scalar('losses/entropy_2', torch_ext.mean_list(train_info['entropy_2']).item(), frame)
        self.writer.add_scalar('losses/entropy_3', torch_ext.mean_list(train_info['entropy_3']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/clip_frac_0', torch_ext.mean_list(train_info['actor_clip_frac_0']).item(), frame)
        self.writer.add_scalar('info/clip_frac_1', torch_ext.mean_list(train_info['actor_clip_frac_1']).item(), frame)
        self.writer.add_scalar('info/clip_frac_2', torch_ext.mean_list(train_info['actor_clip_frac_2']).item(), frame)
        self.writer.add_scalar('info/clip_frac_3', torch_ext.mean_list(train_info['actor_clip_frac_3']).item(), frame)
        self.writer.add_scalar('info/kl_0', torch_ext.mean_list(train_info['kl_0']).item(), frame)
        self.writer.add_scalar('info/kl_1', torch_ext.mean_list(train_info['kl_1']).item(), frame)
        self.writer.add_scalar('info/kl_2', torch_ext.mean_list(train_info['kl_2']).item(), frame)
        self.writer.add_scalar('info/kl_3', torch_ext.mean_list(train_info['kl_3']).item(), frame)
        return
