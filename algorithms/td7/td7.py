import torch
import torch.nn.functional as F
from algorithms.base import BaseAlgorithm
from buffers.off_policy_buffer import LAP
from .network import Actor, Critic
from utils.utils import LAP_huber


class TD7(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'TD7'
        self.type = 'off_policy'
        self.learn_step = 0
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.actor = Actor(self.obs_dim, self.act_dim, 1, config['algorithm']['noise_scale'],
                                config['algorithm']['latent_obs_dim'], config['actor_hidden_dims'], config['algorithm']['encoder_hidden_dims'],
                                config['algorithm']['noise_scale_annealing'], config['algorithm']['noise_scale_annealing_max_steps'], config['activation_fc'])
        self.critic = Critic(self.obs_dim, self.act_dim, config['algorithm']['latent_obs_dim'], config['critic_hidden_dims'], 'elu')

        self.buffer = LAP(self.obs_dim, self.act_dim, config['device'],
                                 int(config['buffer_capacity']), config['algorithm']['LAP_normalize_action'],
                                 1, config['algorithm']['LAP_prioritized'])

        self.log_alpha = None
        self.alpha_optimizer = None

        self.training_steps = 0
        self.target_update_steps = config['algorithm']['target_update_steps']
        self.max_action = 1

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0
        super().__init__()

    def train(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, encoder_optimizer, iteration):
        for _ in range(iteration):
            self.training_steps += 1
            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'])

            # Update Encoder
            with torch.no_grad():
                next_zs = policy.encoder.zs(next_states)
            zs = policy.encoder.zs(states)
            pred_zs = policy.encoder.zsa(zs, actions)
            encoder_loss = F.mse_loss(pred_zs, next_zs)

            encoder_optimizer.zero_grad()
            encoder_loss.backward()
            encoder_optimizer.step()

            # Update Critic
            with torch.no_grad():
                fixed_target_zs = policy.fixed_encoder_target.zs(next_states)

                target_act_noise = (torch.randn_like(actions) * self.config['algorithm']['target_noise_scale']).clamp(
                    -self.config['algorithm']['target_noise_clip'], self.config['algorithm']['target_noise_clip']).to(self.config['device'])

                if buffer.do_normalize_action is True:  # Bug fixed from the original code
                    next_target_actions = (target_policy(next_states, fixed_target_zs) + target_act_noise).clamp(-1, 1)
                else:
                    next_target_actions = (target_policy(next_states, fixed_target_zs) + target_act_noise).clamp(-self.max_action, self.max_action)

                fixed_target_zsa = policy.fixed_encoder_target.zsa(fixed_target_zs, next_target_actions)

                Q_target = target_critic(next_states, next_target_actions, fixed_target_zsa, fixed_target_zs).min(1, keepdim=True)[0]
                Q_target = rewards + (1 - dones) * self.config['gamma'] * Q_target.clamp(self.min_target, self.max_target)

                self.max = max(self.max, float(Q_target.max()))
                self.min = min(self.min, float(Q_target.min()))

                fixed_zs = policy.fixed_encoder.zs(states)
                fixed_zsa = policy.fixed_encoder.zsa(fixed_zs, actions)

            Q = critic(states, actions, fixed_zsa, fixed_zs)
            td_loss = (Q - Q_target).abs()
            critic_loss = LAP_huber(td_loss)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update LAP
            priority = td_loss.max(1)[0].clamp(min=1).pow(self.config['algorithm']['LAP_alpha'])
            buffer.update_priority(priority)

            if self.training_steps % self.config['algorithm']['policy_update_delay'] == 0:
                # Update Actor
                actor_actions = policy(states, fixed_zs)
                fixed_zsa = policy.fixed_encoder.zsa(fixed_zs, actor_actions)
                Q = critic(states, actor_actions, fixed_zsa, fixed_zs)
                actor_loss = -Q.mean()

                policy_optimizer.zero_grad()
                actor_loss.backward()
                policy_optimizer.step()

                #  Conditioning for Action Policy Smoothness (CAPS) Part.
                if self.config['use_caps']:
                    if self.config['caps_annealing']:
                        caps_lambda_t = (min(self.training_steps / self.config['caps_annealing_max_steps'], 1)
                                         * self.config['caps_lambda_t'])
                        caps_lambda_s = (min(self.training_steps / self.config['caps_annealing_max_steps'], 1)
                                         * self.config['caps_lambda_s'])
                    else:
                        caps_lambda_t, caps_lambda_s = self.config['caps_lambda_t'], self.config['caps_lambda_s']
                    policy_optimizer.zero_grad()
                    next_actions = policy(next_states, fixed_target_zs)
                    caps_t_loss = caps_lambda_t * ((next_actions - actions) ** 2).mean()

                    noise = self.config['caps_eps'] * torch.normal(torch.zeros_like(states), torch.ones_like(states))
                    noisy_states = states + noise

                    with torch.no_grad():
                        fixed_noisy_zs = policy.fixed_encoder.zs(noisy_states)

                    actions_from_noisy_states = policy(noisy_states, fixed_noisy_zs)
                    caps_s_loss = caps_lambda_s * ((actions_from_noisy_states - actions) ** 2).mean()

                    caps_loss = caps_t_loss + caps_s_loss
                    caps_loss.backward()
                    policy_optimizer.step()

            # Update Iteration
            if self.training_steps % self.target_update_steps == 0:
                target_policy.load_state_dict(policy.state_dict())
                target_critic.load_state_dict(critic.state_dict())
                policy.fixed_encoder_target.load_state_dict(policy.fixed_encoder.state_dict())
                policy.fixed_encoder.load_state_dict(policy.encoder.state_dict())

                buffer.reset_max_priority()

                self.max_target = self.max
                self.min_target = self.min
