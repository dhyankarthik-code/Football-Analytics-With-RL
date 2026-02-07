# RL Integration - Quick Start Guide

## Installation

```bash
# Install RL dependencies
pip install stable-baselines3[extra] gymnasium tensorboard
```

## Training

### 1. Prepare Ground Truth Annotations

Create JSON files with ball positions per frame:

```json
{
  "0": {"ball": [x1, y1, x2, y2]},
  "1": {"ball": [x1, y1, x2, y2]},
  ...
}
```

### 2. Train PPO Agent

```bash
python -m src.rl.trainer \
    --video_dir data/training_videos/ \
    --ground_truth_dir data/ground_truth/ \
    --output_dir models/rl/ \
    --timesteps 1000000 \
    --lr 3e-4
```

### 3. Monitor Training

```bash
tensorboard --logdir models/rl/tensorboard
```

## Testing

```bash
# Run environment tests
pytest tests/test_rl_env.py -v
```

## Inference

```python
from src.rl.agent import RLAgent
from src.rl.environment import FootballTrackingEnv

# Load trained agent
agent = RLAgent("models/rl/ppo_ball_tracking_final.zip", algorithm="PPO")

# Use in pipeline
observation = env._get_observation()
action = agent.predict(observation)
refined_box = agent.refine_ball_box(ball_box, observation)
```

## Architecture

- **State**: Ball bbox + velocity + visual features (256-dim)
- **Action**: Continuous [Δx, Δy, Δw, Δh] adjustments
- **Reward**: IoU + differential IoU + smoothness + track continuity
- **Algorithm**: PPO with MultiInputPolicy
