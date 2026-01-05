from .layer import MHCLayer
from .ops import (
    sinkhorn_knopp,
    rmsnorm,
    mhc_layer_fused,
    mhc_layer_fused_dynamic,
)

# CuTe DSL implementations (optional - requires nvidia-cutlass)
try:
    from .cute_dsl import (
        MHCLayerDSL,
        mhc_layer_fused_dsl,
        rmsnorm as rmsnorm_dsl,
        sinkhorn_knopp as sinkhorn_knopp_dsl,
        stream_aggregate,
        stream_distribute_mix_add,
    )
    CUTE_DSL_AVAILABLE = True
except ImportError:
    MHCLayerDSL = None
    mhc_layer_fused_dsl = None
    rmsnorm_dsl = None
    sinkhorn_knopp_dsl = None
    stream_aggregate = None
    stream_distribute_mix_add = None
    CUTE_DSL_AVAILABLE = False

__all__ = [
    # Original CUDA implementations
    "MHCLayer",
    "sinkhorn_knopp",
    "rmsnorm",
    "mhc_layer_fused",
    "mhc_layer_fused_dynamic",
    # CuTe DSL implementations
    "MHCLayerDSL",
    "mhc_layer_fused_dsl",
    "rmsnorm_dsl",
    "sinkhorn_knopp_dsl",
    "stream_aggregate",
    "stream_distribute_mix_add",
    "CUTE_DSL_AVAILABLE",
]
