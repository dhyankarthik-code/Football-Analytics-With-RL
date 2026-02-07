"""
Training script for PPO agent on football ball tracking.
"""
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.rl.environment import FootballTrackingEnv

logger = get_logger("RLTrainer")


def make_env(config, video_path, ground_truth_path):
    """Create environment instance."""
    def _init():
        return FootballTrackingEnv(
            config=config,
            video_path=video_path,
            ground_truth_path=ground_truth_path,
            max_steps=500
        )
    return _init


def train_ppo(
    config_path: str,
    video_dir: str,
    ground_truth_dir: str,
    output_dir: str,
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    device: str = "auto"
):
    """
    Train PPO agent for ball tracking.
    
    Args:
        config_path: Path to config.yaml
        video_dir: Directory containing training videos
        ground_truth_dir: Directory with ground truth annotations
        output_dir: Output directory for models
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Optimization epochs per rollout
        gamma: Discount factor
        device: Device (cuda/cpu/auto)
    """
    # Load config
    config = load_config(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find training videos
    import glob
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    gt_files = glob.glob(os.path.join(ground_truth_dir, "*.json"))
    
    if not video_files:
        raise ValueError(f"No videos found in {video_dir}")
    
    logger.info(f"Found {len(video_files)} training videos")
    
    # Create environment (using first video for now)
    # TODO: Implement multi-video training with curriculum learning
    env = DummyVecEnv([make_env(config, video_files[0], gt_files[0] if gt_files else None)])
    
    # Create PPO model
    logger.info("Initializing PPO model...")
    model = PPO(
        policy="MultiInputPolicy",  # For Dict observation space
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(output_dir, "tensorboard")
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="ppo_ball_tracking"
    )
    
    # Train
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "ppo_ball_tracking_final.zip")
    model.save(final_model_path)
    logger.info(f"Training complete! Model saved to {final_model_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for ball tracking")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--video_dir", type=str, required=True, help="Training videos directory")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Ground truth annotations directory")
    parser.add_argument("--output_dir", type=str, default="models/rl", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    train_ppo(
        config_path=args.config,
        video_dir=args.video_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
