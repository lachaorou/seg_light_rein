"""
Token Merging for Fast Stable Diffusion and Vision Transformers
Paper: https://arxiv.org/abs/2303.17604

This implementation provides ToMe (Token Merging) for computational efficiency.
ToMe reduces the number of tokens in Vision Transformers by merging similar tokens,
which can significantly speed up inference while maintaining performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
import math


def do_nothing(x, mode=None):
    """Do nothing function for fallback"""
    return x


def mps_gather_workaround(input, dim, index):
    """
    Workaround for MPS gather function
    """
    if input.device.type == "mps":
        # MPS doesn't support gather with different dtypes
        return torch.gather(input.cpu(), dim, index.cpu()).to(input.device)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies bipartite soft matching to merge tokens.

    Args:
        metric: Metric tensor for similarity computation [B, N, C]
        r: Number of tokens to remove
        class_token: Whether to protect class token
        distill_token: Whether to protect distillation token

    Returns:
        merge: Function to merge tokens
        unmerge: Function to unmerge tokens (for gradients)
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce the number of tokens if we have enough tokens
    if r <= 0 or r >= metric.shape[1] - protected:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged tokens
        src_idx = edge_idx[..., :r, :]  # Source tokens (to be merged)
        dst_idx = mps_gather_workaround(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        """Merge tokens based on matching"""
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape

        unm = mps_gather_workaround(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = mps_gather_workaround(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = mps_gather_workaround(dst, dim=-2, index=dst_idx.expand(n, r, c))

        if mode == "mean":
            dst = dst + src
        elif mode == "max":
            dst = torch.maximum(dst, src)
        elif mode == "min":
            dst = torch.minimum(dst, src)
        else:
            dst = dst + src  # default to mean

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """Unmerge tokens for gradient flow"""
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = mps_gather_workaround(dst, dim=-2, index=dst_idx.expand(n, r, c))

        # Combine back
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        # Scatter unmerged tokens
        out.scatter_(dim=-2, index=unm_idx.expand(n, unm_len, c) * 2, src=unm)
        # Scatter source tokens
        out.scatter_(dim=-2, index=src_idx.expand(n, r, c) * 2, src=src)
        # Scatter destination tokens
        out.scatter_(dim=-2, index=dst_idx.expand(n, r, c) * 2 + 1, src=dst)

        return out

    return merge, unmerge


class ToMeBlock(nn.Module):
    """
    Token Merging Block that can be inserted into existing architectures
    """

    def __init__(
        self,
        embed_dim: int,
        merge_ratio: float = 0.5,
        merge_mode: str = "mean",
        use_class_token: bool = False,
        use_distill_token: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.merge_ratio = merge_ratio
        self.merge_mode = merge_mode
        self.use_class_token = use_class_token
        self.use_distill_token = use_distill_token

        # Learnable projection for computing similarity metric
        self.metric_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply token merging to input tensor

        Args:
            x: Input tensor [B, N, C] or [B, C, H, W]

        Returns:
            Merged tensor with reduced number of tokens
        """
        if x.dim() == 4:
            # Convert from [B, C, H, W] to [B, N, C]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            need_reshape = True
        else:
            need_reshape = False
            B, N, C = x.shape
            H = W = int(math.sqrt(N))

        # Compute metric for similarity
        metric = self.metric_proj(x)

        # Calculate number of tokens to remove
        r = int(x.shape[1] * self.merge_ratio)

        # Get merge/unmerge functions
        merge, unmerge = bipartite_soft_matching(
            metric, r, self.use_class_token, self.use_distill_token
        )

        # Apply merging
        x_merged = merge(x, mode=self.merge_mode)

        if need_reshape:
            # Try to reshape back - this is approximate since we reduced tokens
            new_N = x_merged.shape[1]
            new_H = new_W = int(math.sqrt(new_N))
            if new_H * new_W == new_N:
                x_merged = x_merged.reshape(B, new_H, new_W, C).permute(0, 3, 1, 2)
            else:
                # If perfect square not possible, pad/crop to make it work
                target_size = int(math.sqrt(new_N))
                if target_size * target_size < new_N:
                    target_size += 1

                # Pad tokens if needed
                pad_needed = target_size * target_size - new_N
                if pad_needed > 0:
                    padding = torch.zeros(B, pad_needed, C, device=x.device, dtype=x.dtype)
                    x_merged = torch.cat([x_merged, padding], dim=1)

                x_merged = x_merged[:, :target_size*target_size].reshape(B, target_size, target_size, C).permute(0, 3, 1, 2)

        return x_merged


class ToMeWrapper(nn.Module):
    """
    Wrapper to apply ToMe to any feature map or attention module
    """

    def __init__(
        self,
        module: nn.Module,
        embed_dim: int,
        merge_ratio: float = 0.3,
        merge_locations: list = None,
        merge_mode: str = "mean"
    ):
        super().__init__()
        self.module = module
        self.tome_blocks = nn.ModuleDict()
        self.merge_locations = merge_locations or ["before"]

        # Create ToMe blocks for each specified location
        for location in self.merge_locations:
            self.tome_blocks[location] = ToMeBlock(
                embed_dim=embed_dim,
                merge_ratio=merge_ratio,
                merge_mode=merge_mode
            )

    def forward(self, x):
        # Apply ToMe before the module if specified
        if "before" in self.merge_locations:
            x = self.tome_blocks["before"](x)

        # Apply the original module
        x = self.module(x)

        # Apply ToMe after the module if specified
        if "after" in self.merge_locations:
            x = self.tome_blocks["after"](x)

        return x


def apply_tome_to_model(
    model: nn.Module,
    merge_ratio: float = 0.3,
    target_modules: list = None,
    merge_mode: str = "mean"
) -> nn.Module:
    """
    Apply Token Merging to specific modules in a model

    Args:
        model: Target model to modify
        merge_ratio: Ratio of tokens to merge (0.0 - 1.0)
        target_modules: List of module names/types to wrap with ToMe
        merge_mode: How to merge tokens ("mean", "max", "min")

    Returns:
        Modified model with ToMe applied
    """
    if target_modules is None:
        target_modules = ["Conv2d", "MultiheadAttention"]

    def apply_tome_recursive(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this module should be wrapped
            should_wrap = False
            for target_type in target_modules:
                if target_type in str(type(child_module)):
                    should_wrap = True
                    break

            if should_wrap:
                # Try to infer embedding dimension
                embed_dim = 256  # default
                if hasattr(child_module, 'out_channels'):
                    embed_dim = child_module.out_channels
                elif hasattr(child_module, 'embed_dim'):
                    embed_dim = child_module.embed_dim

                # Wrap the module
                wrapped = ToMeWrapper(
                    child_module,
                    embed_dim=embed_dim,
                    merge_ratio=merge_ratio,
                    merge_mode=merge_mode
                )
                setattr(module, child_name, wrapped)
            else:
                # Recursively apply to children
                apply_tome_recursive(child_module, full_name)

    apply_tome_recursive(model)
    return model


def build_tome_module(
    embed_dim: int = 256,
    merge_ratio: float = 0.3,
    merge_mode: str = "mean",
    **kwargs
) -> ToMeBlock:
    """
    Build a Token Merging module

    Args:
        embed_dim: Embedding dimension
        merge_ratio: Ratio of tokens to merge
        merge_mode: Merging strategy

    Returns:
        ToMeBlock instance
    """
    return ToMeBlock(
        embed_dim=embed_dim,
        merge_ratio=merge_ratio,
        merge_mode=merge_mode,
        **kwargs
    )


if __name__ == "__main__":
    # Test ToMe block
    print("Testing Token Merging...")

    # Test with feature maps
    tome_block = ToMeBlock(embed_dim=256, merge_ratio=0.3)

    # Test 4D input (feature maps)
    x = torch.randn(2, 256, 32, 32)
    print(f"Input shape: {x.shape}")

    y = tome_block(x)
    print(f"Output shape: {y.shape}")

    # Test 3D input (sequence)
    x = torch.randn(2, 1024, 256)
    print(f"Input shape (sequence): {x.shape}")

    y = tome_block(x)
    print(f"Output shape (sequence): {y.shape}")

    print("Token Merging test completed successfully!")
