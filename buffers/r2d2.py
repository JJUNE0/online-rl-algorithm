import torch
import numpy as np
from buffers.base import BaseBuffer

class R2D2Buffer(BaseBuffer):
    """
    Replay buffer for recurrent policies like R2D2.
    Stores transitions but samples them as sequences.
    """
    def __init__(self, obs_dim, act_dim, capacity, sequence_length):
        super().__init__(capacity)
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")
        
        self.size = 0
        self.position = 0
        self.sequence_length = sequence_length

        # 버퍼에 s, a, r, done 외에 에피소드 시작 여부를 저장합니다.
        self.state_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.action_buffer = np.empty(shape=(self.capacity, act_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        # done은 에피소드의 끝을 나타내므로, episode_starts 플래그가 있으면 시퀀스가 에피소드를 넘나드는 것을 방지하기 좋습니다.
        self.episode_starts_buffer = np.empty(shape=(self.capacity, 1), dtype=np.bool_)

    def clear(self):
        self.size = 0
        self.position = 0

    def push(self, state, action, reward, next_state, done, episode_start):
        # next_state는 state_buffer[t+1]에 저장되므로, 명시적으로 저장할 필요가 없습니다.
        self.size = min(self.size + 1, self.capacity)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.done_buffer[self.position] = done
        self.episode_starts_buffer[self.position] = episode_start

        self.position = (self.position + 1) % self.capacity

    def _get_valid_indices(self, batch_size):
        """에피소드 경계를 넘지 않는 유효한 시퀀스 시작 인덱스를 찾습니다."""
        valid_indices = []
        # 버퍼가 꽉 찼을 때와 아닐 때를 모두 고려
        end_idx = self.size if self.size < self.capacity else self.capacity
        
        for i in range(end_idx - self.sequence_length + 1):
            # 시퀀스 중간에 다른 에피소드가 시작되면 안 됩니다.
            if np.sum(self.episode_starts_buffer[i+1:i+self.sequence_length]) == 0:
                valid_indices.append(i)
        
        if not valid_indices:
            raise ValueError("Not enough data to sample a full sequence. Check sequence_length and buffer population.")

        return np.random.choice(valid_indices, size=batch_size)

    def sample(self, batch_size, device='cuda'):
        """개별 transition이 아닌, sequence를 샘플링합니다."""
        if self.size < self.sequence_length:
            return None # 아직 샘플링할 데이터가 충분하지 않음

        start_indices = self._get_valid_indices(batch_size)

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for idx in start_indices:
            # 상태는 s_t, s_{t+1}, ..., s_{t+L} 이므로 L+1개가 필요합니다.
            # 하지만 학습 시에는 s_t 부터 s_{t+L-1} 까지의 상태에서 한 행동과 보상을 사용하므로,
            # next_state를 위해 마지막 state를 하나 더 가져옵니다.
            indices = np.arange(idx, idx + self.sequence_length + 1) % self.capacity

            states = self.state_buffer[indices]
            actions = self.action_buffer[indices[:-1]] # 행동은 L개
            rewards = self.reward_buffer[indices[:-1]] # 보상도 L개
            dones = self.done_buffer[indices[:-1]]     # done도 L개

            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_dones.append(dones)

        # states: (batch, seq_len+1, obs_dim), others: (batch, seq_len, dim)
        states = torch.FloatTensor(np.array(batch_states)).to(device)
        actions = torch.FloatTensor(np.array(batch_actions)).to(device)
        rewards = torch.FloatTensor(np.array(batch_rewards)).to(device)
        dones = torch.FloatTensor(np.array(batch_dones)).to(device)

        return states, actions, rewards, dones