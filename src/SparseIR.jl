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

"""
    SparseIRError <: Exception

Raised when a call into `libsparseir` fails. Carries a human-readable message
naming the failing C entry point and, where available, the raw numeric status
code returned by that entry point (`status`, `nothing` for failures that are
not reported through a status code, such as a null handle).
"""
struct SparseIRError <: Exception
    msg::String
    status::Union{Nothing,Int32}
end

SparseIRError(msg::AbstractString) = SparseIRError(String(msg), nothing)

function Base.showerror(io::IO, e::SparseIRError)
    print(io, "SparseIRError: ", e.msg)
    if e.status !== nothing
        print(io, " (status=", e.status, ", ", _status_name(e.status), ")")
    end
    return nothing
end

const _STATUS_NAMES = Dict{Int32,String}(
    Int32(C_API.SPIR_COMPUTATION_SUCCESS) => "SPIR_COMPUTATION_SUCCESS",
    Int32(C_API.SPIR_GET_IMPL_FAILED) => "SPIR_GET_IMPL_FAILED",
    Int32(C_API.SPIR_INVALID_DIMENSION) => "SPIR_INVALID_DIMENSION",
    Int32(C_API.SPIR_INPUT_DIMENSION_MISMATCH) => "SPIR_INPUT_DIMENSION_MISMATCH",
    Int32(C_API.SPIR_OUTPUT_DIMENSION_MISMATCH) => "SPIR_OUTPUT_DIMENSION_MISMATCH",
    Int32(C_API.SPIR_NOT_SUPPORTED) => "SPIR_NOT_SUPPORTED",
    Int32(C_API.SPIR_INVALID_ARGUMENT) => "SPIR_INVALID_ARGUMENT",
    Int32(C_API.SPIR_INTERNAL_ERROR) => "SPIR_INTERNAL_ERROR"
)

"""
    _status_name(status)

Name of a known `libsparseir` status code, or a marker for an unrecognized one.
"""
_status_name(status) = get(_STATUS_NAMES, Int32(status), "unrecognized status code")

const _DIMENSION_STATUSES = (
    Int32(C_API.SPIR_INVALID_DIMENSION),
    Int32(C_API.SPIR_INPUT_DIMENSION_MISMATCH),
    Int32(C_API.SPIR_OUTPUT_DIMENSION_MISMATCH)
)

"""
    _check_status(status, operation)

Check a `libsparseir` status code exhaustively: return `nothing` on
`SPIR_COMPUTATION_SUCCESS`, throw `DimensionMismatch` for the dimension-related
codes and [`SparseIRError`](@ref) for **every** other value, including codes
this wrapper does not know about.
"""
function _check_status(status, operation::AbstractString)
    st = Int32(status)
    st == Int32(C_API.SPIR_COMPUTATION_SUCCESS) && return nothing
    if st in _DIMENSION_STATUSES
        throw(DimensionMismatch("$operation failed with status $st ($(_status_name(st)))"))
    end
    throw(SparseIRError("$operation failed", st))
end

"""
    _check_handle(ptr, operation)

Throw [`SparseIRError`](@ref) if a C entry point returned a null handle.
"""
function _check_handle(ptr::Ptr, operation::AbstractString)
    ptr == C_NULL && throw(SparseIRError("$operation returned a null handle"))
    return ptr
end

"""
    _check_all_finite(A, name)

Throw `ArgumentError` naming the first offending index if `A` contains a `NaN`
or an infinity. Called at construction time for every array this wrapper hands
to a C entry point that factorizes, decomposes or solves with it.
"""
function _check_all_finite(A::AbstractArray, name::AbstractString)
    idx = findfirst(!isfinite, A)
    idx === nothing && return nothing
    throw(ArgumentError("$name contains the non-finite value $(A[idx]) at index $(idx); \
                         all entries must be finite"))
end

"""
    _check_unique(points, name)

Throw `ArgumentError` if `points` contains an exact duplicate. Duplicated
sampling points make the sampling matrix rank-deficient (infinite condition
number), which the C API accepts silently and which produces meaningless fit
coefficients.
"""
function _check_unique(points::AbstractVector, name::AbstractString)
    seen = Dict{eltype(points),Int}()
    for (i, p) in enumerate(points)
        j = get(seen, p, 0)
        if j != 0
            throw(ArgumentError("$name contains the duplicate value $p at indices $j and \
                                 $i; sampling points must be pairwise distinct"))
        end
        seen[p] = i
    end
    return nothing
end

import libsparseir_jll
# From Julia, an "opaque pointer" is sufficient to represent the backend
const SpirGemmBackend = Ptr{Cvoid}

# Globally retained (passed to other ccall as needed)
const _spir_default_backend = Ref{SpirGemmBackend}(C_NULL)

# ===== Obtaining BLAS function pointers =====

function _get_blas_gemm_ptrs()
    # Ensure libblastrampoline is forwarded to the actual BLAS implementation
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

# The backend handle type on the C side is represented as Ptr{Cvoid}
# (the detailed struct is only known to the Rust side)
const SpirGemmBackend = Ptr{Cvoid}

function _init_sparseir_blas_backend()
    dgemm_ptr, zgemm_ptr = _get_blas_gemm_ptrs()

    # Use the correct backend based on BLAS integer size
    # ILP64 uses 64-bit integers, LP64 uses 32-bit integers
    if LinearAlgebra.BLAS.USE_BLAS64
        backend = ccall(
            (:spir_gemm_backend_new_from_fblas_ilp64, C_API.libsparseir),
            SpirGemmBackend,                  # struct spir_gemm_backend*
            (Ptr{Cvoid}, Ptr{Cvoid}),         # const void *dgemm64, const void *zgemm64
            dgemm_ptr, zgemm_ptr
        )
    else
        backend = ccall(
            (:spir_gemm_backend_new_from_fblas_lp64, C_API.libsparseir),
            SpirGemmBackend,                  # struct spir_gemm_backend*
            (Ptr{Cvoid}, Ptr{Cvoid}),         # const void *dgemm, const void *zgemm
            dgemm_ptr, zgemm_ptr
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
