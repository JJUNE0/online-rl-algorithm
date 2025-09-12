from algorithms.base import BaseCritic, BaseActor
from utils.data_handler import fast_clip
from utils.utils import weight_init, AvgL1Norm
import copy
import torch
import torch.nn as nn


class Critic(BaseCritic):
    def __init__(self, state_dim, action_dim, zs_dim=256, hidden_dims=(256, 256), activation_fc_name='elu'):
        super(Critic, self).__init__(activation_fc_name)

        # Q_A
        self.s_input_layer_A = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.emb_input_layer_A = nn.Linear(2 * zs_dim + hidden_dims[0], hidden_dims[0])

        self.emb_hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.emb_hidden_layers_A.append(hidden_layer_A)
        self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        # Q_B
        self.s_input_layer_B = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.emb_input_layer_B = nn.Linear(2 * zs_dim + hidden_dims[0], hidden_dims[0])

        self.emb_hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.emb_hidden_layers_B.append(hidden_layer_B)
        self.output_layer_B = nn.Linear(hidden_dims[-1], 1)

        self.apply(weight_init)

    @staticmethod
    def _format(state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action, zsa, zs):
        s, a = self._format(state, action)
        sa = torch.cat([s, a], dim=1)
        embeddings = torch.cat([zsa, zs], 1)

        q_A = AvgL1Norm(self.s_input_layer_A(sa))
        q_A = torch.cat([q_A, embeddings], dim=1)

        q_A = self.activation_fc(self.emb_input_layer_A(q_A))
        for i, hidden_layer_A in enumerate(self.emb_hidden_layers_A):
            q_A = self.activation_fc(hidden_layer_A(q_A))
        q_A = self.output_layer_A(q_A)

        q_B = AvgL1Norm(self.s_input_layer_B(sa))
        q_B = torch.cat([q_B, embeddings], dim=1)

        q_B = self.activation_fc(self.emb_input_layer_B(q_B))
        for i, hidden_layer_B in enumerate(self.emb_hidden_layers_B):
            q_B = self.activation_fc(hidden_layer_B(q_B))
        q_B = self.output_layer_B(q_B)

        return torch.cat([q_A, q_B], 1)


class Actor(BaseActor):
    def __init__(self, obs_dim, act_dim, max_action, act_noise_scale, zs_dim, hidden_dims, encoder_hidden_dims,
                 noise_scale_annealing, noise_scale_annealing_max_steps, activation_fc_name='relu'):
        super(Actor, self).__init__(activation_fc_name)
        self.max_action = max_action
        self.act_noise_scale_init = act_noise_scale

        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.zs_dim = zs_dim
        self.hidden_dims = hidden_dims
        self.encoder_hidden_dims = encoder_hidden_dims
        self.activation_fc_name = activation_fc_name

        self.mlp = TD7MLP(obs_dim, act_dim, zs_dim, hidden_dims, activation_fc_name)
        self.encoder = TD7Encoder(obs_dim, act_dim, zs_dim, encoder_hidden_dims)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.explore_cnt = 0
        self.noise_scale_annealing = noise_scale_annealing
        self.noise_scale_annealing_max_steps = noise_scale_annealing_max_steps

        self.apply(weight_init)

    def forward(self, state, zs):
        action = self.mlp.forward(state, zs)
        return action

    def exploit(self, state):
        zs = self.fixed_encoder.zs(state)
        action = self.forward(state, zs)
        return action

    def explore(self, state):
        self.explore_cnt += 1
        if self.noise_scale_annealing:
            act_noise_scale = max(self.act_noise_scale_init - (self.act_noise_scale_init - 0.1) * self.explore_cnt / self.noise_scale_annealing_max_steps, 0.1)
        else:
            act_noise_scale = self.act_noise_scale_init

        zs = self.fixed_encoder.zs(state)
        action = self.forward(state, zs)
        action = action + torch.randn_like(action) * act_noise_scale
        return action

    def get_action(self, state, eval):
        state = torch.FloatTensor(state).unsqueeze(0).to('cuda')
        with torch.no_grad():
            if not eval:
                action = self.explore(state).cpu().numpy()[0]
                action = fast_clip(action, -1, 1)
            else:
                action = self.exploit(state).cpu().numpy()[0]

            action = action * self.max_action
        return action, None


class TD7MLP(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim, hidden_dims, activation_fc_name):
        super(TD7MLP, self).__init__()
        self.apply(weight_init)
        if activation_fc_name in ['ELU', 'elu']:
            self.activation_fc = nn.ELU()
        elif activation_fc_name in ['ReLU', 'relu', 'RELU']:
            self.activation_fc = nn.ReLU()
        else:
            raise NotImplementedError

        self.s_input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.zss_input_layer = nn.Linear(zs_dim + hidden_dims[0], hidden_dims[0])
        self.zss_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.zss_hidden_layers.append(hidden_layer)
        self.zss_output_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.apply(weight_init)

    @staticmethod
    def AvgL1Norm(x, eps=1e-8):
        return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)

        return x

    def forward(self, state, zs):
        state = self._format(state)
        state = self.AvgL1Norm(self.s_input_layer(state))
        zss = torch.cat([state, zs], 1)

        zss = self.activation_fc(self.zss_input_layer(zss))
        for i, hidden_layer in enumerate(self.zss_hidden_layers):
            zss = self.activation_fc(hidden_layer(zss))
        zss = self.zss_output_layer(zss)
        action = torch.tanh(zss)
        return action


class TD7Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hidden_dims=(256, 256), activation_fc_name='elu'):
        super(TD7Encoder, self).__init__()
        if activation_fc_name in ['ELU', 'elu']:
            self.activation_fc = nn.ELU()
        elif activation_fc_name in ['ReLU', 'relu', 'RELU']:
            self.activation_fc = nn.ReLU()
        else:
            raise NotImplementedError

        self.s_encoder_input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.s_encoder_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.s_encoder_hidden_layers.append(hidden_layer)
        self.s_encoder_output_layer = nn.Linear(hidden_dims[-1], zs_dim)

        self.zsa_encoder_input_layer = nn.Linear(zs_dim + action_dim, hidden_dims[0])
        self.zsa_encoder_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.zsa_encoder_hidden_layers.append(hidden_layer)
        self.zsa_encoder_output_layer = nn.Linear(hidden_dims[-1], zs_dim)
        self.apply(weight_init)

    @staticmethod
    def _format( state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def zs(self, state):
        state = self._format(state)

        zs = self.activation_fc(self.s_encoder_input_layer(state))
        for i, hidden_layer in enumerate(self.s_encoder_hidden_layers):
            zs = self.activation_fc(hidden_layer(zs))
        zs = AvgL1Norm(self.s_encoder_output_layer(zs))
        return zs

    def zsa(self, zs, action):
        action = self._format(action)
        zsa = torch.cat([zs, action], 1)

        zsa = self.activation_fc(self.zsa_encoder_input_layer(zsa))
        for i, hidden_layer in enumerate(self.zsa_encoder_hidden_layers):
            zsa = self.activation_fc(hidden_layer(zsa))
        zsa = self.zsa_encoder_output_layer(zsa)
        return zsa


# ================================================================================= #
# =================================== ONNX ======================================== #
# ================================================================================= #

class TD7ONNXPolicy(BaseActor):
    def __init__(self, state_dim, action_dim, max_action, zs_dim,  hidden_dims, encoder_hidden_dims, activation_fc_name='relu'):
        super(TD7ONNXPolicy, self).__init__(activation_fc_name)
        self.max_action = max_action
        self.mlp = TD7MLP(state_dim, action_dim, zs_dim, hidden_dims, activation_fc_name)
        self.fixed_encoder = TD7Encoder(state_dim, action_dim, zs_dim, encoder_hidden_dims)

    def forward(self, state):
        state = torch.unsqueeze(state, dim=0)
        zs = self.fixed_encoder.zs(state)
        action = self.mlp.forward(state, zs)
        action = action[0] * self.max_action
        return action