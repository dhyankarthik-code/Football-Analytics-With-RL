"""
Test suite for RL environment.
"""
import pytest
import numpy as np
from src.utils.config import load_config
from src.rl.environment import FootballTrackingEnv
from src.rl.reward import RewardCalculator, calculate_iou


def test_iou_calculation():
    """Test IoU calculation."""
    box_a = np.array([0, 0, 10, 10])
    box_b = np.array([5, 5, 15, 15])
    
    iou = calculate_iou(box_a, box_b)
    
    # Expected IoU: intersection = 5*5 = 25, union = 100 + 100 - 25 = 175
    expected_iou = 25 / 175
    
    assert abs(iou - expected_iou) < 1e-6


def test_reward_calculator():
    """Test reward calculator."""
    calc = RewardCalculator()
    
    pred_box = np.array([10, 10, 50, 50])
    gt_box = np.array([12, 12, 52, 52])
    
    reward, components = calc.compute(pred_box, gt_box)
    
    assert "iou" in components
    assert "diff_iou" in components
    assert "total" in components
    assert isinstance(reward, float)


def test_environment_reset():
    """Test environment reset."""
    config = load_config("configs/config.yaml")
    
    env = FootballTrackingEnv(
        config=config,
        video_path=None,  # No video for testing
        max_steps=10
    )
    
    obs, info = env.reset()
    
    assert "ball_box" in obs
    assert "ball_velocity" in obs
    assert "visual_features" in obs
    assert obs["ball_box"].shape == (4,)
    assert obs["ball_velocity"].shape == (2,)
    assert obs["visual_features"].shape == (256,)


def test_environment_step():
    """Test environment step."""
    config = load_config("configs/config.yaml")
    
    env = FootballTrackingEnv(
        config=config,
        video_path=None,
        max_steps=10
    )
    
    obs, info = env.reset()
    
    # Random action
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "ball_box" in obs


def test_action_space():
    """Test action space bounds."""
    config = load_config("configs/config.yaml")
    
    env = FootballTrackingEnv(config=config)
    
    # Sample actions should be within bounds
    for _ in range(100):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        assert np.all(action >= -0.1)
        assert np.all(action <= 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
