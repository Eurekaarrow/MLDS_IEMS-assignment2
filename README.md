# MLDS_IEMS Assignment 2 - DQN

DQN (Double DQN target, Huber loss, replay buffer, target network) for CartPole-v1 and MsPacman-v5.

## Setup
Same environment as assignment 1:
```
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

## CartPole-v1 (gamma 0.95, CPU, ~3 min per seed)
```
python dqn_cartpole.py train --seed 0          # 1000 episodes, keeps the best greedy checkpoint
python dqn_cartpole.py eval  --seed 0          # 500 roll-outs of q_best.pt
python plots.py cartpole --seed 0
python validate_cartpole.py --seed 0           # tests C1-C6 (+ ablations, ~5 min)
```

## MsPacman-v5 (gamma 0.99, one GPU)
```
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
python dqn_pacman.py train --run pacman --episodes 6000
python dqn_pacman.py eval  --run pacman        # 500 roll-outs of q_final.pt
python plots.py pacman
python validate_pacman.py --run pacman         # tests M1-M5 + non-standard plots
```
Checkpoints, replay data and logs are not in the repo.
