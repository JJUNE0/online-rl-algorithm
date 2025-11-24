from .learner import Learner
from .logger import OffPolicyLogger

import torch
import inspect
import time
import datetime
import wandb

from utils.utils import _get_default


import time
import numpy as np
from collections import deque
from utils import wandb_utils

GREEN = "\033[92m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"


class Runner(object):
    def __init__(self, env, eval_env, algorithm, config):
        assert config['random_seed'] > 0
        assert algorithm.type == 'off_policy'
        assert config['update_after'] > config['batch_size']
        assert config['buffer_capacity'] > config['batch_size']

        # Set the environment, algorithm, and configuration.
        self.env = env
        self.eval_env = eval_env
        self.config = config
        self.stop_flag = False
        self.total_steps = 0
        self.total_episodes = 0
        self.best_return = float("-inf")
        self.start_time = time.time()

        # Create the components of the AC architecture.
        self.learner = Learner(algorithm, config)

        # Load the saved model.
        if self.config['load_model']:
            self.load_model()
            
        self.logger = OffPolicyLogger(config)
        
        self.wb = wandb_utils.init_wandb(config)
    
        self.t_env = 0.0
        self.t_policy = 0.0
        self.t_buffer_push = 0.0
        
        self.n_policy_calls = 0            
        self._last_eval_reset_steps = 0     

    def load_model(self):
        """Load model optimizers and buffer from checkpoint."""
        checkpoint = self.config['load_checkpoint_dir'] + '/checkpoint.pt'
        # Load the critic and actor (policy).
        self.learner.actor.load_state_dict(checkpoint['policy_state_dict'])
        self.learner.critic.load_state_dict(checkpoint['critic_state_dict'])
        # Load the optimizers.
        self.learner.actor_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.learner.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if checkpoint['encoder_optimizer_state_dict'] is not None:
            self.learner.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        if checkpoint['log_alpha'] is not None:
            self.learner.algorithm.log_alpha = torch.asarray([checkpoint['log_alpha']], requires_grad=True, device=self.config['device'])
            self.learner.algorithm.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        else:
            self.learner.algorithm.log_alpha = None
            self.learner.algorithm.alpha_optimizer = None
        # Load the buffer.
        self.learner.buffer = checkpoint['buffer']
        return


    def stop(self):
        print("Training Stopped!")
        self.stop_flag = True

    def evaluate(self):
        reward_list = []
        pol_ms_list = []
        
        for epi_count in range(1, self.config['eval_episodes'] + 1):
            epi_reward = 0

            state, _ = self.eval_env.reset()
            
            if self.learner.buffer.type == 'sequence':
                history_len = self.learner.actor.core.max_len
                state_history = deque(maxlen=history_len)
                initial_state_pad = np.zeros_like(state)
                for _ in range(history_len - 1):
                    state_history.append(initial_state_pad)
                state_history.append(state)
            
            terminated = False
            truncated = False
            while not (terminated or truncated):
                t0 = time.perf_counter()
                # TODO 오히려 잘 안됨...
                if self.learner.buffer.type == 'sequence':
                    current_history = np.array(state_history) # shape: (T, S)
                    action, _ = self.learner.actor.get_action(current_history, eval=True)
                else:
                    action, _ = self.learner.actor.get_action(state, eval=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()   # GPU 이벤트 flush
                t1 = time.perf_counter()
                pol_ms_list.append((t1 - t0) * 1000.0)
                
                try:
                    next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                except ValueError:
                    next_state, reward, terminated, truncated, info = self.eval_env.step(action[0])

                
                if self.learner.buffer.type == 'sequence':
                    state_history.append(next_state)
                else:
                    state = next_state
                epi_reward += reward

                if self.stop_flag:
                    return

            reward_list.append(epi_reward)

        avg_return = sum(reward_list) / len(reward_list)
        max_return = max(reward_list)
        min_return = min(reward_list)

        if len(pol_ms_list) > 0:
            pol_ms_avg = float(np.mean(pol_ms_list))
            pol_ms_p95 = float(np.percentile(pol_ms_list, 95))
        else:
            pol_ms_avg, pol_ms_p95 = 0.0, 0.0

        wandb_utils.log_speed(self.wb, self.total_steps, {
                "speed/policy_ms_avg": pol_ms_avg,
                "speed/policy_ms_p95": pol_ms_p95,
            })
        
        return avg_return, max_return, min_return

    def monitor(self):
        avg_return, max_return, min_return = self.evaluate()
        
        pol_calls = max(1, self.n_policy_calls)
        pol_ms = (self.t_policy * 1e3) / pol_calls
        env_steps_since = max(1, self.total_steps - self._last_eval_reset_steps)
        env_ms = (self.t_env * 1e3) / env_steps_since
        push_ms = (self.t_buffer_push * 1e3) / env_steps_since
        
        print(
            f"{GREEN}{BOLD}[Evaluation]{RESET}  | Steps {self.total_steps}  |  Episodes {self.total_episodes}  |  "
            f"Average return {avg_return:.2f}  |  Max return: {max_return:.2f}  |  Min return: {min_return:.2f}"
        )
        print(
            f"{CYAN}{BOLD}[Rollout]{RESET} policy {pol_ms:.3f} ms/call | "
            f"env {env_ms:.3f} ms/step | push {push_ms:.3f} ms/step"
        )
        wandb_utils.log_speed(self.wb, self.total_steps, {
            "speed/rollout_policy_ms_per_call": pol_ms,
            "speed/rollout_env_ms_per_step": env_ms,
            "speed/rollout_push_ms_per_step": push_ms,
            })
        wandb_utils.log_eval(self.wb, self.total_steps, self.total_episodes, avg_return, max_return, min_return)
        
        if self.config['save_model']:
            (policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict, encoder_optimizer_state_dict,
             log_alpha, alpha_optimizer_state_dict, buffer) = self.learner.get_params()
            wall_time = time.time() - self.start_time
            now = datetime.datetime.now()
            cur_time = now.strftime('%Y-%m-%d_%H:%M:%S')
            meta_data = {'algorithm': self.config['algorithm']['name'], 'env_id': self.env.id, 'episodes': self.total_episodes,
                         'average return': float(avg_return), 'timestep': self.total_steps, 'wall time': wall_time,
                         'log_datetime': cur_time, 'critic_hidden_dims': list(self.config['critic_hidden_dims']),
                         'actor_hidden_dims': list(self.config['actor_hidden_dims'])}

            if avg_return > self.best_return:
                self.best_return = avg_return
                best_update_flag = True

                self.logger.log(self.learner.actor, self.learner.critic, policy_optimizer_state_dict, critic_optimizer_state_dict,
                                encoder_optimizer_state_dict, meta_data, log_alpha, alpha_optimizer_state_dict, buffer,
                                best_update_flag=best_update_flag, cur_update_flag=True)

            if self.total_steps % self.config['model_checkpoint_freq'] == 0:
                self.logger.log(policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict,
                                encoder_optimizer_state_dict, meta_data, log_alpha, alpha_optimizer_state_dict, buffer,
                                best_update_flag=False, cur_update_flag=False)
                print(f"Save the model ... checkpoint: {self.logger.checkpoint_no}")

    def run(self):

        while True:
            step = 0
            epi_return = 0
            self.total_episodes += 1

            state, _ = self.env.reset()
            terminated, truncated = False, False
            episode_start = True
            
            while not (terminated or truncated):
                step += 1
                self.total_steps += 1
                
                t0 = time.perf_counter()
                if self.total_steps < self.config['max_random_rollout']:
                    action = self.env.random_action_sample()
                else:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    tp0 = time.perf_counter()
                    action, _ = self.learner.actor.get_action(state, eval=False)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    tp1 = time.perf_counter()
                    self.t_policy += (tp1 - tp0)
                    self.n_policy_calls += 1   
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                te0 = time.perf_counter()
                next_states, reward, terminated, truncated, info = self.env.step(action)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                te1 = time.perf_counter()
                self.t_env += (te1 - te0)
                true_done = 0.0 if truncated else float(terminated or truncated)
                
                tb0 = time.perf_counter()
                if self.learner.buffer.type == 'sequence':
                    self.learner.buffer.push(state, action, reward, next_states, true_done, episode_start)
                else:
                    self.learner.buffer.push(state, action, reward, next_states, true_done)
                tb1 = time.perf_counter()
                self.t_buffer_push += (tb1 - tb0)
                
                episode_start = bool(terminated or truncated)
                epi_return += reward
                state = next_states

                if (self.total_steps % self.config['max_rollout']) == 0 and (self.total_steps > self.config['update_after']):
                    self.learner.learn()
                    wandb_utils.log_train(self.wb, self.total_steps, self.total_episodes,
                                      epi_return=None,
                                      total_losses=getattr(self.learner, "total_losses", None))

                if (self.total_steps % self.config['eval_freq']) == 0:
                    self.monitor()

                if self.total_steps >= self.config['max_steps']:
                    print("Training Done!")
                    self.stop_flag = True

                if self.stop_flag:
                    wandb_utils.finish(self.wb)
                    return




