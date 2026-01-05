"""Base class for reduction kernels.

Adapted from Quack's reduction_base.py for MHC kernels. Provides:
- ReductionBase class with common infrastructure
- Thread/warp/block configuration selection
- Tiled copy setup
- Reduction buffer allocation
"""

from typing import Type, Tuple, Optional
import math

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Float32, const_expr

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

from . import copy_utils


class ReductionBase:
    """Base class for reduction kernels (RMSNorm, Softmax, etc.).

    This class provides common infrastructure for memory-bound reduction
    kernels following Quack patterns. It handles:
    - Thread configuration based on reduction dimension
    - Tiled copy setup with vectorization
    - Reduction buffer allocation for block/cluster reduction
    - Cluster setup for very large reductions

    Subclasses should implement:
    - _threads_per_row(): Return optimal threads per row
    - kernel(): The actual kernel implementation
    - __call__(): Launch configuration and kernel invocation
    """

    def __init__(
        self,
        dtype,
        N: int,
        stage: int = 1,
        reduction_dtype=None,
    ):
        """Initialize reduction kernel base.

        Args:
            dtype: Input/output element dtype
            N: Reduction dimension size
            stage: Number of pipeline stages (for double-buffering)
            reduction_dtype: Dtype for reduction accumulation (default: Float32)
        """
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype if reduction_dtype else Float32
        self.cluster_n = 1

    def _threads_per_row(self) -> int:
        """Select optimal threads per row based on reduction dimension.

        This determines parallelism within each row. Larger N benefits
        from more threads per row.

        Returns:
            Number of threads processing each row
        """
        N = self.N
        # Threshold-based selection following Quack
        for limit, threads in [
            (64, 8),
            (128, 16),
            (3072, 32),
            (6144, 64),
            (16384, 128),
        ]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self) -> int:
        """Select total thread count.

        Returns:
            Total threads per block
        """
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        """Set cluster size for distributed shared memory reduction.

        For very large N (>16k for bf16, >32k for fp32), we use
        multiple CTAs in a cluster to cooperatively reduce.
        """
        if not CUTLASS_AVAILABLE:
            self.cluster_n = 1
            return

        N = self.N
        if const_expr(self.dtype.width == 16):
            # bf16/fp16 thresholds
            thresholds = [
                (16 * 1024, 1),
                (32 * 1024, 2),
                (64 * 1024, 4),
                (128 * 1024, 8),
            ]
        else:
            # fp32 thresholds
            thresholds = [
                (32 * 1024, 1),
                (64 * 1024, 2),
                (128 * 1024, 4),
                (256 * 1024, 8),
            ]

        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    def _get_tiled_copy(self, vecsize: int = 1):
        """Create tiled copy configuration for this kernel.

        Args:
            vecsize: Vectorization size

        Returns:
            Tuple of (tiled_copy, tiler_mn, threads_per_row)
        """
        if not CUTLASS_AVAILABLE:
            raise RuntimeError("CUTLASS not available")

        assert self.N % vecsize == 0, f"N ({self.N}) must be divisible by vecsize ({vecsize})"

        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % 32 == 0, "num_threads must be multiple of warp size"

        num_blocks_N = (
            (self.N // vecsize) + threads_per_row * self.cluster_n - 1
        ) // (threads_per_row * self.cluster_n)
        tiler_mn = (
            num_threads // threads_per_row,
            vecsize * num_blocks_N * threads_per_row,
        )

        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, num_threads, vecsize
        )

        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout, cluster_n: int):
        """Compute reduction buffer layout based on thread-value layout.

        Args:
            tv_layout: Thread-value layout from tiled copy
            cluster_n: Cluster size

        Returns:
            Layout for reduction buffer in shared memory
        """
        if not CUTLASS_AVAILABLE:
            raise RuntimeError("CUTLASS not available")

        num_warps = cute.size(tv_layout, mode=[0]) // 32  # WARP_SIZE

        # Determine warps per row from TV layout structure
        if cute.rank(tv_layout.shape[0]) == 1:
            warps_per_row = num_warps
        else:
            warps_per_row = max(tv_layout.shape[0][0] // 32, 1)

        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(self, smem, tv_layout):
        """Allocate reduction buffer and optional cluster barrier.

        Args:
            smem: SmemAllocator instance
            tv_layout: Thread-value layout

        Returns:
            Tuple of (reduction_buffer tensor, mbar_ptr or None)
        """
        if not CUTLASS_AVAILABLE:
            raise RuntimeError("CUTLASS not available")

        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )

        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(Int64, num_elems=self.stage)
        else:
            mbar_ptr = None

        return reduction_buffer, mbar_ptr

    def _initialize_cluster(self, tidx, mbar_ptr, num_warps: int):
        """Initialize cluster barriers if using distributed shared memory.

        Args:
            tidx: Thread index
            mbar_ptr: Memory barrier pointer
            num_warps: Number of warps
        """
        if not CUTLASS_AVAILABLE:
            return

        @cute.jit
        def _init_cluster(self_cluster_n, self_stage, tidx, mbar_ptr):
            if const_expr(self_cluster_n > 1):
                if tidx < self_stage:
                    cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.cluster_arrive_relaxed()

        _init_cluster(self.cluster_n, self.stage, tidx, mbar_ptr)

    def get_vectorization_size(self, *tensors) -> int:
        """Calculate optimal vectorization size for given tensors.

        Args:
            *tensors: Tensors to consider (will use their dtypes)

        Returns:
            Optimal vector size
        """
        if not CUTLASS_AVAILABLE:
            return 1

        # Get widths of all tensor dtypes
        widths = [t.element_type.width for t in tensors if t is not None]
        if not widths:
            return 1

        largest_width = max(widths)

        # Maximum 128 bits per load, GCD with N for alignment
        max_vec = 128 // largest_width
        return math.gcd(self.N, max_vec)


class SmemAllocator:
    """Simple shared memory allocator wrapper.

    Provides a convenient interface for allocating shared memory
    tensors and arrays with proper alignment.
    """

    def __init__(self):
        if not CUTLASS_AVAILABLE:
            raise RuntimeError("CUTLASS not available")
        self._allocator = cutlass.utils.SmemAllocator()

    def allocate_tensor(self, dtype, layout, byte_alignment: int = 16):
        """Allocate a tensor in shared memory.

        Args:
            dtype: Element dtype
            layout: Tensor layout
            byte_alignment: Byte alignment

        Returns:
            Shared memory tensor
        """
        return self._allocator.allocate_tensor(
            dtype, layout, byte_alignment=byte_alignment
        )

    def allocate_array(self, dtype, num_elems: int, byte_alignment: int = 8):
        """Allocate an array in shared memory.

        Args:
            dtype: Element dtype
            num_elems: Number of elements
            byte_alignment: Byte alignment

        Returns:
            Pointer to shared memory array
        """
        return self._allocator.allocate_array(
            dtype, num_elems=num_elems, byte_alignment=byte_alignment
        )


def ceil_div(a: int, b: int) -> int:
    """Ceiling division helper.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        ceil(a / b)
    """
    return (a + b - 1) // b
