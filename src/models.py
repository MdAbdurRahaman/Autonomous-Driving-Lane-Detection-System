#!/usr/bin/env python3
"""
src/models.py
Defines the deep learning architectures for the Lane Detection System.
Includes Model A: U-Net and Model B: DeepLabV3 with ResNet50.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation

logger = logging.getLogger(__name__)


# ==========================================
# MODEL A: U-NET ARCHITECTURE
# ==========================================

class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for semantic segmentation.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        # Final Output layer (maps to output channel counts, sigmoid applied in loss/predict)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with Skip Connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits


# ==========================================
# MODEL B: DEEPLABV3-RESNET50 ARCHITECTURE
# ==========================================

def get_deeplabv3_resnet50(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """
    Initializes and adapts DeepLabV3-ResNet50 for binary lane detection.
    """
    try:
        # Modern torchvision weights API
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = segmentation.deeplabv3_resnet50(weights=weights)
        logger.info("Initialized DeepLabV3-ResNet50 using Weights API.")
    except (ImportError, AttributeError):
        # Fallback to older torchvision format
        model = segmentation.deeplabv3_resnet50(pretrained=pretrained)
        logger.info("Initialized DeepLabV3-ResNet50 using legacy pretrained flag.")

    # DeepLabV3 structure:
    # model.classifier is DeepLabHead.
    # model.classifier[4] is Conv2d(256, 21, kernel_size=(1, 1))
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    # Adapt auxiliary classifier if it exists
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
        
    return model


# ==========================================
# MODEL BUILDER FACTORY
# ==========================================

def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to build lane detection models.
    
    Args:
        model_name: "unet" or "deeplabv3"
        pretrained: Only applicable to DeepLabV3
        
    Returns:
        PyTorch nn.Module model instance.
    """
    name_lower = model_name.lower().strip()
    if name_lower == "unet":
        logger.info("Building custom U-Net model...")
        return UNet(in_channels=3, out_channels=1)
    elif name_lower == "deeplabv3":
        logger.info(f"Building DeepLabV3 model (pretrained={pretrained})...")
        return get_deeplabv3_resnet50(pretrained=pretrained, num_classes=1)
    else:
        raise ValueError(f"Unknown model architecture: '{model_name}'. Options are 'unet' or 'deeplabv3'.")


if __name__ == "__main__":
    # Quick architecture validation
    test_tensor = torch.randn(1, 3, 256, 512)
    
    unet = build_model("unet")
    unet_out = unet(test_tensor)
    print(f"U-Net Input shape: {test_tensor.shape} | Output shape: {unet_out.shape}")
    
    deeplab = build_model("deeplabv3", pretrained=False)
    # DeepLabV3 outputs a dict with 'out' and optionally 'aux'
    deeplab_out = deeplab(test_tensor)["out"]
    print(f"DeepLabV3 Input shape: {test_tensor.shape} | Output shape: {deeplab_out.shape}")
