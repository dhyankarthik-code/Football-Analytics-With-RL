"""
Configuration utility for Football Analytics Pro.
Handles loading YAML configs and provides dot-notation access.
"""
import os
import yaml
from box import Box

def load_config(config_path: str = "configs/config.yaml") -> Box:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        Box: Configuration object allowing dot notation access (e.g., config.detection.model_path)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        try:
            config_dict = yaml.safe_load(f)
            # Use Box for dot notation access
            config = Box(config_dict, default_box=True)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

def get_device(config: Box) -> str:
    """
    Determine the optimal device (CPU vs CUDA) based on config and availability.
    """
    import torch
    
    if config.device and config.device != "auto":
        return config.device
        
    if torch.cuda.is_available():
        return "cuda:0"
    
    return "cpu"
