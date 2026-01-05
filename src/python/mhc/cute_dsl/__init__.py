"""CuTe DSL kernels for MHC (Multi-Head Communication) layer.

This module provides Python-native CUDA kernels using the NVIDIA CUTLASS Python DSL.
The implementation follows Quack patterns for speed-of-light performance.

Key components:
- rmsnorm: RMSNorm forward/backward kernels
- sinkhorn: Sinkhorn-Knopp normalization kernels
- stream_ops: Stream aggregation/distribution operations
- mhc_layer: Complete MHC layer composition
"""

from .rmsnorm import rmsnorm, rmsnorm_fwd, RMSNormFunction
from .sinkhorn import sinkhorn_knopp, sinkhorn_knopp_fwd, SinkhornKnoppFunction
from .stream_ops import (
    stream_aggregate,
    stream_distribute_mix_add,
    StreamAggregateFunction,
    StreamDistributeMixAddFunction,
)
from .mhc_layer import MHCLayerDSL, mhc_layer_fused_dsl

__all__ = [
    # RMSNorm
    "rmsnorm",
    "rmsnorm_fwd",
    "RMSNormFunction",
    # Sinkhorn-Knopp
    "sinkhorn_knopp",
    "sinkhorn_knopp_fwd",
    "SinkhornKnoppFunction",
    # Stream Operations
    "stream_aggregate",
    "stream_distribute_mix_add",
    "StreamAggregateFunction",
    "StreamDistributeMixAddFunction",
    # MHC Layer
    "MHCLayerDSL",
    "mhc_layer_fused_dsl",
]
