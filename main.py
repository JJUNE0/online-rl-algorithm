import os
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

from core.trainer import Trainer
from envs import OpenAIGym #, CraneEnv
#from envs.wrappers.welford_normalize import OnlineNormalizedEnv 


def log_file_directory(args):
    if args.load_model:
        save_path = args.load_checkpoint_dir
        timestamp = Path(save_path.rstrip("/")).name 
        print(f"Resuming training. Checkpoints will be saved to: {save_path}")
        
        config_load_path = os.path.join(save_path, 'config.yaml')
        if os.path.exists(config_load_path):
            saved_config = OmegaConf.load(config_load_path)

            saved_config.load_checkpoint_dir = args.load_checkpoint_dir
            
            args = saved_config
            
            print(f"Configuration loaded from: {config_load_path}")
            print("All other command-line overrides are IGNORED.")
        else:
            print(f"Warning: Configuration file not found at {config_load_path}. Using current settings.")
        
    else:
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f'logs/{args.algorithm.name}/{args.env_id}/{timestamp}/'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        print(f"Starting new training. Checkpoints will be saved to: {save_path}")
        config_save_path = os.path.join(save_path, 'config.yaml')
        OmegaConf.save(config=args, f=config_save_path)
        print(f"Configuration saved to: {config_save_path}")
        
    return save_path

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(args):
    
    if args.save_model:
        log_dir = log_file_directory(args)
        args.save_dir = log_dir
    
    if args.env_type == "gymnasium":
        env = OpenAIGym(env_id=args.env_id)
        eval_env = OpenAIGym(env_id=args.env_id)
    elif args.env_type == "gtsu":
        env =CraneEnv(fmu_filename=args.fmu_filename)    
        eval_env =CraneEnv(fmu_filename=args.fmu_filename)        
    else:
        raise NotImplementedError

    
    trainer = Trainer(env, eval_env, args)
    trainer.run()

if __name__ == "__main__":
    main()
