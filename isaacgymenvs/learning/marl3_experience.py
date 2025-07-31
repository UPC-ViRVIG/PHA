from rl_games.common.experience import ExperienceBuffer
import gym
import numpy as np

class MARL3ExperienceBuffer(ExperienceBuffer):
    def __init__(self, env_info, algo_info, device, aux_tensor_dict=None):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get('agents', 1)
        self.action_space = env_info['action_space']
        
        self.num_actors = algo_info['num_actors']
        self.horizon_length = algo_info['horizon_length']
        self.has_central_value = algo_info['has_central_value']
        self.use_action_masks = algo_info.get('use_action_masks', False)
        batch_size = self.num_actors * self.num_agents
        self.is_discrete = False
        self.is_multi_discrete = False
        self.is_continuous = False
        self.obs_base_shape = (self.horizon_length, self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_shape = ()
            self.actions_num = self.action_space.n
            self.is_discrete = True
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_shape = (len(self.action_space),) 
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        if type(self.action_space) is gym.spaces.Box:
            self.actions_shape = (self.action_space.shape[0],) 
            self.actions_num = self.action_space.shape[0]
            self.is_continuous = True
        self.tensor_dict = {}
        self._init_from_env_info(self.env_info)

        self.aux_tensor_dict = aux_tensor_dict
        if self.aux_tensor_dict is not None:
            self._init_from_aux_dict(self.aux_tensor_dict)
    
    def _init_from_env_info(self, env_info):
        obs_base_shape = self.obs_base_shape
        state_base_shape = self.state_base_shape

        self.tensor_dict['obses'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
        if self.has_central_value:
            assert not self.has_central_value, "central value not supported yet!"
        
        val_space = gym.spaces.Box(low=0, high=1,shape=(env_info.get('value_size',1),))
        self.tensor_dict['rewards'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['values'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['neglogpacs_0'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['neglogpacs_1'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['neglogpacs_2'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['dones'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.uint8), obs_base_shape)

        if self.is_discrete or self.is_multi_discrete:
            assert not self.is_discrete, "discrete not supported yet!"
            assert not self.is_multi_discrete, "multi_discrete not supported yet!"
        if self.use_action_masks:
            assert not self.use_action_masks, "action masks not supported yet!"
        if self.is_continuous:
            self.tensor_dict['actions_0'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['actions_1'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['actions_2'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['mus_0'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['mus_1'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['mus_2'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['sigmas_0'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['sigmas_1'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32), obs_base_shape)
            self.tensor_dict['sigmas_2'] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32), obs_base_shape)