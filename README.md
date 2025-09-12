# 🤖 Online Reinforcement Learning Research Framework
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This is a research framework for online reinforcement learning algorithms, built on PyTorch. Designed with a modular structure, it allows for easy implementation of new algorithms and facilitates comparative experiments.

## ✨ Features
- 🧩 Modular Design: Code is separated by functionality (algorithms, buffers, core), making it easy to extend and maintain.

- 🔧 Hydra-Powered Configuration: Manage all hyperparameters through .yaml files in the configs directory, ensuring high reproducibility.

- 📊 Weights & Biases Integration: Seamlessly track and visualize your training progress, results, and videos in real-time with wandb.

- 🤖 Gymnasium Support: Test algorithms across a wide range of tasks with support for gymnasium and gymnasium-robotics environments.

## 🛠️ Installation
1. Clone the Repository
 ```bash
  git clone <your-repository-url>
  cd online-rl-algorithm
  ```
2. Create and Activate a Virtual Environment
  ```bash
  conda create -n online_rl python=3.10
  ```
3. Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

## 🚀 Usage
Start training by running [main.py]. Thanks to Hydra, you can override any configuration parameter directly from the command line.

### Basic Usage
This command runs the default experiment defined in configs/config.yaml, which trains the SAC algorithm on the Walker2d-v5 environment.
  ```bash
  python main.py
  ```
### Changing Algorithm and Environment
You can easily specify the algorithm and environment from the command line.

  ```bash
# Train TD3 on the HalfCheetah-v4 environment
python main.py algorithm=td3 env.env_id=HalfCheetah-v4

# Train TQC on the Ant-v4 environment
python main.py algorithm=tqc env.env_id=Ant-v4
```

### Resuming Training
To resume a previous run, set load_model=True and provide the path to the checkpoint directory.

```
python main.py load_model=True load_checkpoint_dir=logs/sac/Walker2d-v5/2025-09-12_10-54-34/
```

All training artifacts—including logs, model checkpoints, and configuration files—are automatically saved to the logs/ directory, organized by algorithm, environment, and timestamp.

## 🦾 Implemented Algorithms
[SAC (Soft Actor-Critic)](https://arxiv.org/pdf/1801.01290)

[TD3 (Twin Delayed Deep Deterministic Policy Gradient)]()

[TD7]()

[TQC (Truncated Quantile Critics)]()

[HRAC]()

## 📁 Project Structure
``` bash
/
├───main.py                 # Entry point for training
├───requirements.txt        # List of project dependencies
├───algorithms/             # Implementations of RL algorithms
├───buffers/                # Replay buffer implementations
├───configs/                # Hydra configuration files (.yaml)
├───core/                   # Core logic (Trainer, Logger, etc.)
├───envs/                   # Environment wrappers (Gymnasium)
├───logs/                   # Directory for logs and model checkpoints
└───utils/                  # Utility functions
```
