from algorithms import SAC, TD3, TQC, TD7, DDPG #, HRMTD3, TFHRMTD3, HRMTD3_RNN
from .runner import Runner

import torch

from typing import Any
import numpy as np

from utils.utils import _get_default

class Trainer:
    def __init__(self, env, eval_env, config):
        
        self.env = env
        self.eval_env = eval_env
        self.algorithm = config.algorithm
        self.config = config
        
        seed = _get_default(getattr(config, "random_seed", None), 0)
        print(f"Seed : {seed}") 
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Check CUDA environment
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            config['device'] = 'cpu'
            
        if self.algorithm.name in ['td7', 'TD7']:
            algorithm = TD7(env, config)
        elif self.algorithm.name in ['tqc', 'TQC']:
            algorithm = TQC(env, config)
        elif self.algorithm.name in ['sac', 'SAC']:
            algorithm = SAC(env, config)
        elif self.algorithm.name in ['td3', 'TD3']:
            algorithm = TD3(env, config)
        elif self.algorithm.name in ['ddpg', 'DDPG']:
            algorithm = DDPG(env, config)
        # elif self.algorithm.name in ['hrmtd3']:
        #     algorithm = HRMTD3(env, config)
        #     config['save_model']=False
        # elif self.algorithm.name in ['tfhrmtd3']:
        #     algorithm = TFHRMTD3(env, config)
        #     config['save_model']=False
        # elif self.algorithm.name in ['tfhrmtd3']:
        #     algorithm = TFHRMTD3(env, config)
        #     config['save_model']=False
        # elif self.algorithm.name in ['hrmtd3_rnn' , 'hrmtd3_lstm', 'hrmtd3_gru']:
        #     algorithm = HRMTD3_RNN(env, config)
        #     config['save_model']=False
        else:
            print("Check algorithm.name in configs/algorithm")
            raise NameError
        
        self.runner = Runner(env, eval_env, algorithm, config)
    
    def run(self):
        self.runner.run()
        return
    
    def stop(self):
        self.runner.stop()
        return