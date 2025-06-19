"""
Object-Contextual Representations for Semantic Segmentation
Paper: https://arxiv.org/abs/1909.11065

OCR (Object-Contextual Representations) Head implementation for semantic segmentation.
OCR leverages object-contextual representations to enhance feature representation
by modeling the relationships between pixels and object regions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SpatialGatherModule(nn.Module):
    """Spatial Gather Module for collecting contextual information"""

    def __init__(self):
        super(SpatialGatherModule, self).__init__()

    def forward(self, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Aggregate spatial features using attention probs

        Args:
            features: [B, C, H, W] - Feature maps
            probs: [B, K, H, W] - Attention probabilities

        Returns:
            context: [B, K, C] - Aggregated context features
        """
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)  # [B, K, H*W]
        features = features.view(batch_size, features.size(1), -1)  # [B, C, H*W]
        features = features.permute(0, 2, 1)  # [B, H*W, C]

        # Weighted aggregation: [B, K, H*W] x [B, H*W, C] -> [B, K, C]
        context = torch.matmul(probs, features)

        return context


class SpatialOCRModule(nn.Module):
    """Spatial OCR Module for object-contextual representation learning"""

    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 out_channels: int,
                 scale: int = 1,
                 dropout: float = 0.1):
        super(SpatialOCRModule, self).__init__()

        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.out_channels = out_channels

        # Query, Key, Value projections
        self.query_project = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True)
        )

        self.key_project = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True)
        )

        self.value_project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Output projection
        self.out_project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        self.spatial_gather = SpatialGatherModule()

    def forward(self,
                query_feats: torch.Tensor,
                key_feats: torch.Tensor,
                value_feats: torch.Tensor,
                out_feats: torch.Tensor,
                probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Spatial OCR Module

        Args:
            query_feats: Query features [B, C, H, W]
            key_feats: Key features [B, C, H, W]
            value_feats: Value features [B, C, H, W]
            out_feats: Output features [B, C, H, W]
            probs: Object probabilities [B, K, H, W]

        Returns:
            Enhanced features [B, C, H, W]
        """
        batch_size, h, w = query_feats.size(0), query_feats.size(2), query_feats.size(3)

        # Project features
        query = self.query_project(query_feats)
        key = self.key_project(key_feats)
        value = self.value_project(value_feats)

        # Spatial gather to get object representations
        # context: [B, K, C]
        context = self.spatial_gather(value, probs)

        # Reshape for attention computation
        query = query.view(batch_size, self.key_channels, -1)  # [B, C, H*W]
        query = query.permute(0, 2, 1)  # [B, H*W, C]
        key = context  # [B, K, C]
        value = context  # [B, K, C]

        # Compute attention weights
        # [B, H*W, C] x [B, C, K] -> [B, H*W, K]
        sim_map = torch.matmul(query, key.permute(0, 2, 1))
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # Apply attention to get context-aware features
        # [B, H*W, K] x [B, K, C] -> [B, H*W, C]
        context_feats = torch.matmul(sim_map, value)
        context_feats = context_feats.permute(0, 2, 1).contiguous()  # [B, C, H*W]
        context_feats = context_feats.view(batch_size, self.out_channels, h, w)  # [B, C, H, W]

        # Combine with original features
        out_feats = out_feats + self.out_project(context_feats)

        return out_feats


class ObjectContextBlock(nn.Module):
    """Object Context Block combining soft object regions with OCR module"""

    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 out_channels: int,
                 num_classes: int,
                 dropout: float = 0.1):
        super(ObjectContextBlock, self).__init__()

        self.num_classes = num_classes

        # Soft object region detection
        self.object_context_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Generate soft object regions
        self.object_region_conv = nn.Conv2d(out_channels, num_classes, 1)

        # OCR module
        self.ocr_module = SpatialOCRModule(
            in_channels=out_channels,
            key_channels=key_channels,
            out_channels=out_channels,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input features [B, C, H, W]

        Returns:
            ocr_feats: OCR enhanced features [B, C, H, W]
            object_regions: Soft object regions [B, K, H, W]
        """
        # Generate object context features
        obj_feats = self.object_context_conv(x)

        # Generate soft object regions
        object_regions = self.object_region_conv(obj_feats)
        object_regions = F.softmax(object_regions, dim=1)

        # Apply OCR module
        ocr_feats = self.ocr_module(
            query_feats=obj_feats,
            key_feats=obj_feats,
            value_feats=obj_feats,
            out_feats=obj_feats,
            probs=object_regions
        )

        return ocr_feats, object_regions


class OCRHead(nn.Module):
    """
    Object-Contextual Representations Head for Semantic Segmentation

    Args:
        in_channels: Number of input channels from backbone
        num_classes: Number of segmentation classes
        ocr_mid_channels: Middle channels for OCR module
        ocr_key_channels: Key channels for attention
        dropout_ratio: Dropout ratio
    """

    def __init__(self,
                 in_channels: int = 2048,
                 num_classes: int = 21,
                 ocr_mid_channels: int = 512,
                 ocr_key_channels: int = 256,
                 dropout_ratio: float = 0.1,
                 **kwargs):
        super(OCRHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ocr_mid_channels = ocr_mid_channels
        self.ocr_key_channels = ocr_key_channels

        # Context branch
        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, ocr_mid_channels, 3, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True)
        )

        # Object Context Block
        self.object_context_block = ObjectContextBlock(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            num_classes=num_classes,
            dropout=dropout_ratio
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(ocr_mid_channels, ocr_mid_channels, 3, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(ocr_mid_channels, num_classes, 1)
        )

        # Auxiliary classifier for object regions
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(ocr_mid_channels, ocr_mid_channels, 3, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(ocr_mid_channels, num_classes, 1)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            features: Dictionary containing backbone features
                - high_level_features: [B, C, H, W] main features for segmentation

        Returns:
            Dictionary containing:
                - out: [B, num_classes, H, W] main segmentation output
                - aux: [B, num_classes, H, W] auxiliary output
        """
        # Get high-level features from backbone
        x = features['high_level_features']
        input_size = x.size()[2:]

        # Context features
        context_feats = self.context_conv(x)

        # Object context processing
        ocr_feats, object_regions = self.object_context_block(context_feats)

        # Main segmentation output
        out = self.classifier(ocr_feats)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        # Auxiliary output from context features
        aux = self.aux_classifier(context_feats)
        aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=False)

        return {
            'out': out,
            'aux': aux,
            'object_regions': object_regions
        }

    def get_loss_function(self):
        """Return appropriate loss function for OCR head"""
        def ocr_loss_fn(outputs, targets, aux_weight=0.4):
            """
            OCR loss combining main loss and auxiliary loss

            Args:
                outputs: Model outputs dict
                targets: Ground truth [B, H, W]
                aux_weight: Weight for auxiliary loss

            Returns:
                Combined loss
            """
            main_loss = F.cross_entropy(outputs['out'], targets, ignore_index=255)
            aux_loss = F.cross_entropy(outputs['aux'], targets, ignore_index=255)

            return main_loss + aux_weight * aux_loss

        return ocr_loss_fn


def build_ocr_head(in_channels: int = 2048,
                   num_classes: int = 21,
                   ocr_mid_channels: int = 512,
                   ocr_key_channels: int = 256,
                   dropout_ratio: float = 0.1,
                   **kwargs) -> OCRHead:
    """
    Build OCR Head

    Args:
        in_channels: Input channels from backbone
        num_classes: Number of segmentation classes
        ocr_mid_channels: Middle channels for OCR
        ocr_key_channels: Key channels for attention
        dropout_ratio: Dropout ratio

    Returns:
        OCRHead instance
    """
    return OCRHead(
        in_channels=in_channels,
        num_classes=num_classes,
        ocr_mid_channels=ocr_mid_channels,
        ocr_key_channels=ocr_key_channels,
        dropout_ratio=dropout_ratio,
        **kwargs
    )


if __name__ == "__main__":
    # Test OCR Head
    print("Testing OCR Head...")

    # Create test features
    batch_size = 2
    features = {
        'high_level_features': torch.randn(batch_size, 2048, 32, 32)
    }

    # Create OCR head
    ocr_head = build_ocr_head(
        in_channels=2048,
        num_classes=21,
        ocr_mid_channels=512,
        ocr_key_channels=256
    )

    # Forward pass
    outputs = ocr_head(features)

    print(f"Main output shape: {outputs['out'].shape}")
    print(f"Aux output shape: {outputs['aux'].shape}")
    print(f"Object regions shape: {outputs['object_regions'].shape}")

    # Test loss function
    targets = torch.randint(0, 21, (batch_size, 512, 512))
    loss_fn = ocr_head.get_loss_function()

    # Upsample outputs to match target size
    upsampled_outputs = {
        'out': F.interpolate(outputs['out'], size=(512, 512), mode='bilinear', align_corners=False),
        'aux': F.interpolate(outputs['aux'], size=(512, 512), mode='bilinear', align_corners=False)
    }

    loss = loss_fn(upsampled_outputs, targets)
    print(f"Loss: {loss.item():.4f}")

    print("OCR Head test completed successfully!")
