import torch
from algorithms.base import BaseAlgorithm
from buffers.off_policy_buffer import OffPolicyBuffer
from .network import Actor, Critic
from utils.utils import soft_update, quantile_huber_loss_f


class TQC(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'TQC'
        self.type = 'off_policy'
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.actor = Actor(self.obs_dim, self.act_dim, config['actor_hidden_dims'], env.action_bound, config['algorithm']['log_std_bound'],
                                config['activation_fc'])
        self.critic = Critic(self.obs_dim, self.act_dim, config['critic_hidden_dims'], config['algorithm']['n_critics'], config['algorithm']['n_quantiles'],
                                config['activation_fc'])
        
        self.buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['buffer_capacity']))

        self.target_entropy = -torch.prod(torch.Tensor((self.act_dim, ))).to(config['device'])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['algorithm']['temperature_lr'])

        self.quantiles_total = config['algorithm']['n_critics'] * config['algorithm']['n_quantiles']
        self.top_quantiles_to_drop = int(config['algorithm']['n_critics'] * config['algorithm']['n_drop_atoms'])
        self.tau = config['tau']

        super().__init__()


    def train(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, iteration, encoder_optimizer):
        for _ in range(iteration):
            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'], device=self.config['device'])
            # Train the Critic Loss
            critic_optimizer.zero_grad()
            with torch.no_grad():
                next_actions, next_log_pis, _ = policy.sample(next_states)
                next_z = target_critic(next_states, next_actions)  # batch x nets x quantiles
                sorted_z, _ = torch.sort(next_z.reshape(self.config['batch_size'], -1))
                sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]

                target_z = rewards + (1 - dones) * self.config['gamma'] * (sorted_z_part - self.log_alpha.exp() * next_log_pis)

            cur_z = critic(states, actions)
            critic_loss = quantile_huber_loss_f(cur_z, target_z, self.config['device'])

            critic_loss.backward()
            critic_optimizer.step()

            # Train the Actor Loss
            policy_optimizer.zero_grad()
            actions, log_pis, _ = policy.sample(states)
            z = critic(states, actions)
            actor_loss = (self.log_alpha.exp().detach() * log_pis - z.mean(2).mean(1, keepdim=True)).mean()
            actor_loss.backward()
            policy_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Target-Critic (Soft) Update
            soft_update(critic, target_critic, self.tau)

        return {
                    "critic": float(critic_loss.detach().cpu()),
                    "actor": float(actor_loss.detach().cpu()),
                    "alpha": float(alpha_loss.mean().detach().cpu()),
                
                }
    
# ================================================================================= #
# =================================== ONNX ======================================== #
# ================================================================================= #




