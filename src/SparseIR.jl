module SparseIR

include("C_API.jl") # libsparseir
using .C_API

import LinearAlgebra
using LinearAlgebra: cond
using QuadGK: quadgk

export Fermionic, Bosonic
export MatsubaraFreq, BosonicFreq, FermionicFreq, pioverbeta
export FiniteTempBasis, FiniteTempBasisSet
export DiscreteLehmannRepresentation
export overlap
export LogisticKernel, RegularizedBoseKernel
export iscentrosymmetric
export AugmentedBasis, TauConst, TauLinear, MatsubaraConst
export TauSampling, MatsubaraSampling, evaluate, fit, evaluate!, fit!,
       sampling_points, npoints
export from_IR, to_IR, npoles, get_poles, default_omega_sampling_points

function _is_column_major_contiguous(A::AbstractArray)
    strides(A) == cumprod((1, size(A)...)[1:(end - 1)])
end

using Libdl
import libsparseir_jll
# Julia から見た backend は「opaque なポインタ」で十分
const SpirGemmBackend = Ptr{Cvoid}

# GLobal に保持しておく（必要に応じて他の ccall に渡す想定）
const _spir_default_backend = Ref{SpirGemmBackend}(C_NULL)

# ===== BLAS 関数ポインタの取得 =====

function _get_blas_gemm_ptrs()
    # Initialize LinearAlgebra.BLAS to ensure libblastrampoline is forwarded to the
    # actual BLAS implementation
    LinearAlgebra.BLAS

    interface = LinearAlgebra.BLAS.USE_BLAS64 ? :ilp64 : :lp64
    dgemm_name = "dgemm_"
    zgemm_name = "zgemm_"
    dgemm_ptr = LinearAlgebra.BLAS.lbt_get_forward(dgemm_name, interface)
    zgemm_ptr = LinearAlgebra.BLAS.lbt_get_forward(zgemm_name, interface)
    if dgemm_ptr == C_NULL || zgemm_ptr == C_NULL
        error("Failed to resolve BLAS symbols for $interface: dgemm_ptr=$dgemm_ptr, zgemm_ptr=$zgemm_ptr")
    end

    return dgemm_ptr, zgemm_ptr
end

# C 側の backend ハンドル型を Ptr{Cvoid} で表現（詳細な struct は Rust 側のみが知っていればよい）
const SpirGemmBackend = Ptr{Cvoid}

function _init_sparseir_blas_backend()
    dgemm_ptr, zgemm_ptr = _get_blas_gemm_ptrs()

    # Use the correct backend based on BLAS integer size
    # ILP64 uses 64-bit integers, LP64 uses 32-bit integers
    if LinearAlgebra.BLAS.USE_BLAS64
        backend = ccall(
            (:spir_gemm_backend_new_from_fblas_ilp64, libsparseir_jll.libsparseir),
            SpirGemmBackend,                  # struct spir_gemm_backend*
            (Ptr{Cvoid}, Ptr{Cvoid}),         # const void *dgemm64, const void *zgemm64
            dgemm_ptr, zgemm_ptr,
        )
    else
        backend = ccall(
            (:spir_gemm_backend_new_from_fblas_lp64, libsparseir_jll.libsparseir),
            SpirGemmBackend,                  # struct spir_gemm_backend*
            (Ptr{Cvoid}, Ptr{Cvoid}),         # const void *dgemm, const void *zgemm
            dgemm_ptr, zgemm_ptr,
        )
    end

    backend == C_NULL && error("Failed to create SparseIR BLAS backend from Julia BLAS")

    _spir_default_backend[] = backend
    return nothing
end

function __init__()
    _init_sparseir_blas_backend()
end

include("freq.jl")
include("abstract.jl")
include("kernel.jl")
include("sve.jl")
include("poly.jl")
include("basis.jl")
include("sampling.jl")
include("dlr.jl")
include("basis_set.jl")
include("augment.jl")

end # module SparseIR
