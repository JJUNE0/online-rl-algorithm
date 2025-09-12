from .learner import Learner
from .logger import OffPolicyLogger

import torch
import inspect
import time
import datetime


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
        self.logger = OffPolicyLogger(config)
        self.start_time = time.time()

        # Create the components of the AC architecture.
        self.learner = Learner(algorithm, config)

        # Load the saved model.
        if self.config['load_model']:
            self.load_model()

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
        for epi_count in range(1, self.config['eval_episodes'] + 1):
            epi_reward = 0
            state, _ = self.eval_env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = self.learner.actor.get_action(state, eval=True)
                try:
                    next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                except ValueError:
                    next_state, reward, terminated, truncated, info = self.eval_env.step(action[0])

                state = next_state
                epi_reward += reward

                if self.stop_flag:
                    return

            reward_list.append(epi_reward)

        avg_return = sum(reward_list) / len(reward_list)
        max_return = max(reward_list)
        min_return = min(reward_list)

        return avg_return, max_return, min_return

    def monitor(self):
        avg_return, max_return, min_return = self.evaluate()
        print("Evaluation  | Steps {}  |  Episodes {}  |  Average return {:.2f}  |  Max return: {:.2f}  |  "
              "Min return: {:.2f}".format(self.total_steps, self.total_episodes, avg_return, max_return, min_return))

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

            while not (terminated or truncated):
                step += 1
                self.total_steps += 1
                #print(self.total_steps)
                
                if self.total_steps < self.config['max_random_rollout']:
                    action = self.env.random_action_sample()
                else:
                    action, _ = self.learner.actor.get_action(state, eval=False)

                next_states, reward, terminated, truncated, info = self.env.step(action)
                true_done = 0.0 if truncated else float(terminated or truncated)
                self.learner.buffer.push(state, action, reward, next_states, true_done)

                epi_return += reward
                state = next_states

                if (self.total_steps % self.config['max_rollout']) == 0 and (self.total_steps > self.config['update_after']):
                    self.learner.learn()

                if (self.total_steps % self.config['eval_freq']) == 0:
                    self.monitor()

                if self.total_steps >= self.config['max_steps']:
                    print("Training Done!")
                    self.stop_flag = True

                if self.stop_flag:
                    return




