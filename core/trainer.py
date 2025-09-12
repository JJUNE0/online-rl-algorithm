from algorithms import SAC, TD3, TQC, TD7
from .runner import Runner

import torch
import os
import datetime
import inspect
import yaml
from typing import Any
import numpy as np
import wandb


def _get_default(val: Any, default: Any):
    return val if val is not None else default

class Trainer:
    def __init__(self, env, eval_env, config):
        
        self.env = env
        self.eval_env = eval_env
        self.algorithm = config.algorithm
        self.config = config
        
        seed = _get_default(getattr(config, "seed", None), 0) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Check CUDA environment
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            config['device'] = 'cpu'
        
        group_name = getattr(self.config, "wandb_group", "default")
        
        if getattr(self.config, "use_wandb", False):
            wandb.init(
                config=vars(self.config),
                project=_get_default(getattr(self.config, "wandb_project", None), "online_rl_pytorch"),
                entity=getattr(self.config, "wandb_team", None),
                group=group_name,
                job_type="train_agent",
                name=f"{group_name}-seed{seed}"
            )

            
        if self.algorithm.name in ['td7', 'TD7']:
            algorithm = TD7(env, config)
        elif self.algorithm.name in ['tqc', 'TQC']:
            algorithm = TQC(env, config)
        elif self.algorithm.name in ['sac', 'SAC']:
            algorithm = SAC(env, config)
        elif self.algorithm.name in ['td3', 'TD3']:
            algorithm = TD3(env, config)
        else:
            print("Choose an algorithm in {'TQC', 'TD7'}.")
            raise NameError
        
        self.runner = Runner(env, eval_env, algorithm, config)
    
    def run(self):
        self.runner.run()
        return
    
    def stop(self):
        self.runner.stop()
        return