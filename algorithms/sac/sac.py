import torch
from algorithms.base import BaseAlgorithm
from buffers.off_policy_buffer import OffPolicyBuffer
from .network import Actor, Critic
from utils.utils import soft_update


class SAC(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'SAC'
        self.type = 'off_policy'
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.actor = Actor(self.obs_dim, self.act_dim, config['actor_hidden_dims'], env.action_bound,
                                config['algorithm']['log_std_bound'], config['activation_fc'])
        self.critic = Critic(self.obs_dim, self.act_dim, config['critic_hidden_dims'], config['activation_fc'])

        self.buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['buffer_capacity']))

        self.target_entropy = -torch.prod(torch.Tensor((self.act_dim, ))).to(config['device'])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['algorithm']['temperature_lr'])
        self.tau = config['tau']

        super().__init__()

    def train(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, iteration, encoder_optimizer):
        for _ in range(iteration):
            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'], device=self.config['device'])

            # Train the Critic Loss
            critic_optimizer.zero_grad()
            with torch.no_grad():
                next_actions, next_log_pis, _ = policy.sample(next_states)
                next_q_values_1, next_q_values_2 = target_critic(next_states, next_actions)
                next_q_values = torch.min(next_q_values_1, next_q_values_2) - self.log_alpha.exp() * next_log_pis
                target_q_values = rewards + (1 - dones) * self.config['gamma'] * next_q_values

            q_values_1, q_values_2 = critic(states, actions)
            critic_loss = ((q_values_1 - target_q_values) ** 2).mean() + ((q_values_2 - target_q_values) ** 2).mean()

            critic_loss.backward()
            critic_optimizer.step()

            # Train the Actor Loss
            policy_optimizer.zero_grad()
            actions, log_pis, _ = policy.sample(states)
            q_values_1, q_values_2 = critic(states, actions)
            q_values = torch.min(q_values_1, q_values_2)

            actor_loss = (self.log_alpha.exp().detach() * log_pis - q_values).mean()
            actor_loss.backward()
            policy_optimizer.step()

            # Train the Entropy Temperature
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Target-Critic (Soft) Update
            soft_update(critic, target_critic, self.tau)

        return {
                    "critic": float(critic_loss.detach().cpu()),
                    "actor": float(actor_loss.detach().cpu()),
                    "q1": float(q_values_1.mean().detach().cpu()),
                    "q2": float(q_values_2.mean().detach().cpu()),
                
                }


