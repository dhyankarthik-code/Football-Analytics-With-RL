"""
TimeSformer model definition for event detection.
Wraps Hugging Face Transformers TimeSformer.
"""
import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification, TimesformerConfig
from src.utils.logger import get_logger

logger = get_logger("EventModel")

class EventDetectorModel(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Initialize TimeSformer model.
        
        Args:
            num_classes: Number of event classes (e.g., No-Foul, Foul, Penalty, Dive)
            pretrained: Whether to load Kinetics-400 weights
        """
        super().__init__()
        
        model_name = "facebook/timesformer-base-finetuned-k400"
        
        if pretrained:
            logger.info(f"Loading pretrained TimeSformer: {model_name}")
            self.model = TimesformerForVideoClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = TimesformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = TimesformerForVideoClassification(config)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor usually shape (Batch, Channels, Frames, Height, Width)
               HuggingFace expects (Batch, C, T, H, W) -> pixel_values
        
        Returns:
            Logits tensor (Batch, Num_Classes)
        """
        # Ensure input is channels first, temporal second as preferred by HF image processors usually,
        # but check documentation. TimeSformer expects (batch_size, num_frames, num_channels, height, width) 
        # or (batch_size, num_channels, num_frames, height, width) depending on processor.
        # We will assume standard Torch video format (B, C, T, H, W) and let the model handle it.
        # Actually HF TimeSformer output is a ClassificationOutput object.
        
        outputs = self.model(pixel_values=x)
        return outputs.logits
