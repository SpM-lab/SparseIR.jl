"""
    AbstractSampling

Abstract type for sparse sampling.

Encodes the "basis transformation" of a propagator from the truncated IR
basis coefficients `G_ir[l]` to time/frequency sampled on sparse points
`G(x[i])` together with its inverse, a least squares fit:

         ________________                   ___________________
        |                |    evaluate     |                   |
        |     Basis      |---------------->|     Value on      |
        |  coefficients  |<----------------|  sampling points  |
        |________________|      fit        |___________________|

"""
abstract type AbstractSampling{T,Tmat,F<:SVD} end

function cond(sampling::AbstractSampling)
    return first(sampling.matrix_svd.S) / last(sampling.matrix_svd.S)
end

function Base.show(io::IO, smpl::S) where {S<:AbstractSampling}
    println(io, S)
    print(io, "Sampling points: ")
    return println(io, smpl.sampling_points)
end

"""
    TauSampling <: AbstractSampling

Sparse sampling in imaginary time.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary time.
"""
struct TauSampling{T,Tmat,F<:SVD} <: AbstractSampling{T,Tmat,F}
    sampling_points::Vector{T}
    matrix::Matrix{Tmat}
    matrix_svd::F
end

"""
    TauSampling(basis[, sampling_points])

Construct a `TauSampling` object. If not given, the `sampling_points` are chosen as
the extrema of the highest-order basis function in imaginary time. This turns out
to be close to optimal with respect to conditioning for this size (within a few percent).
"""
function TauSampling(
    basis::AbstractBasis, sampling_points=default_tau_sampling_points(basis)
)
    matrix = eval_matrix(TauSampling, basis, sampling_points)
    sampling = TauSampling(sampling_points, matrix, svd(matrix))

    if iswellconditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end

    return sampling
end

const TauSampling64 = @static if VERSION < v"1.9-"
    TauSampling{Float64,Float64,SVD{Float64,Float64,Matrix{Float64}}}
else
    TauSampling{Float64,Float64,SVD{Float64,Float64,Matrix{Float64},Vector{Float64}}}
end

"""
    MatsubaraSampling <: AbstractSampling

Sparse sampling in Matsubara frequencies.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary frequencies.
"""
struct MatsubaraSampling{T,Tmat,F<:SVD} <: AbstractSampling{T,Tmat,F}
    sampling_points::Vector{T}
    matrix::Matrix{Tmat}
    matrix_svd::F
end

const MatsubaraSampling64F = MatsubaraSampling{FermionicFreq, ComplexF64, SVD{ComplexF64, Float64, Matrix{ComplexF64}}}

const MatsubaraSampling64B = MatsubaraSampling{BosonicFreq, ComplexF64, SVD{ComplexF64, Float64, Matrix{ComplexF64}}}

"""
    MatsubaraSampling(basis[, sampling_points])

Construct a `MatsubaraSampling` object. If not given, the `sampling_points` are chosen as
the (discrete) extrema of the highest-order basis function in Matsubara. This turns out
to be close to optimal with respect to conditioning for this size (within a few percent).
"""
function MatsubaraSampling(
    basis::AbstractBasis, sampling_points=default_matsubara_sampling_points(basis)
)
    matrix = eval_matrix(MatsubaraSampling, basis, sampling_points)
    sampling = MatsubaraSampling(sampling_points, matrix, svd(matrix))

    if iswellconditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end

    return sampling
end

"""
    eval_matrix(T, basis, x)

Return evaluation matrix from coefficients to sampling points. `T <: AbstractSampling`.
"""
eval_matrix(::Type{TauSampling}, basis, x) = permutedims(basis.u(x))
eval_matrix(::Type{MatsubaraSampling}, basis, x) = permutedims(basis.uhat(x))

"""
    evaluate(sampling, al; dim=1)

Evaluate the basis coefficients `al` at the sparse sampling points.
"""
function evaluate(
    smpl::AbstractSampling{S,Tmat}, al::AbstractArray{T,N}; dim=1
) where {S,Tmat,T,N}
    if size(smpl.matrix, 2) ≠ size(al, dim)
        msg = "Number of columns (got $(size(smpl.matrix, 2))) has to match al's size in dim (got $(size(al, dim)))"
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 1), size(al)[(dim + 1):end]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return evaluate!(buffer, smpl, al; dim)
end

"""
    evaluate!(buffer::AbstractArray{T,N}, sampling, al; dim=1) where {T,N}

Like [`evaluate`](@ref), but write the result to `buffer`.
Please use dim = 1 or N to avoid allocating large temporary arrays internally.
"""
function evaluate!(buffer::AbstractArray, smpl::AbstractSampling, al; dim=1)
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 1), size(al)[(dim + 1):end]...)
    if size(buffer) ≠ bufsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $bufsize)"
        throw(DimensionMismatch(msg))
    end
    return matop_along_dim!(buffer, smpl.matrix, al, dim, mul!)
end

"""
    fit(sampling, al::AbstractArray{T,N}; dim=1)

Fit basis coefficients from the sparse sampling points
Please use dim = 1 or N to avoid allocating large temporary arrays internally.
"""
function fit(
    smpl::AbstractSampling{S,Tmat}, al::AbstractArray{T,N}; dim=1
) where {S,Tmat,T,N}
    if size(smpl.matrix, 1) ≠ size(al, dim)
        msg = "Number of rows (got $(size(smpl.matrix, 1))) has to match al's size in dim (got $(size(al, dim)))"
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 2), size(al)[(dim + 1):N]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return fit!(buffer, smpl, al; dim)
end

"""
    workarrlength(smpl::AbstractSampling, al; dim=1)

Return length of workarr for `fit!`.
"""
function workarrlength(smpl::AbstractSampling, al::AbstractArray; dim=1)
    return length(smpl.matrix_svd.S) * (length(al) ÷ size(al, dim))
end

"""
    fit!(buffer, sampling, al::Array{T,N}; dim=1)

Like [`fit`](@ref), but write the result to `buffer`.
Please use dim = 1 or N to avoid allocating large temporary arrays internally.
The length of `workarry` cannot be smaller than the returned value of `workarrlength`.
"""
function fit!(
    buffer::Array{S,N}, smpl::AbstractSampling, al::Array{T,N};
    dim=1, workarr::Vector{S}=Vector{S}(undef, workarrlength(smpl, al; dim)),
) where {S,T,N}
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 2), size(al)[(dim + 1):end]...)
    if size(buffer) ≠ bufsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $bufsize)"
        throw(DimensionMismatch(msg))
    end
    length(workarr) ≥ workarrlength(smpl, al; dim) ||
        throw(ArgumentError("workarr too small"))
    return div_noalloc!(buffer, smpl.matrix_svd, al, workarr, dim)
end

"""
    movedim(arr::AbstractArray, src => dst)

Move `arr`'s dimension at `src` to `dst` while keeping the order of the remaining
dimensions unchanged.
"""
function movedim(arr::AbstractArray{T,N}, dims::Pair) where {T,N}
    src, dst = dims
    src == dst && return arr
    return permutedims(arr, getperm(N, dims))
end

function getperm(N, dims::Pair)
    src, dst = dims
    perm = collect(1:N)
    deleteat!(perm, src)
    insert!(perm, dst, src)
    return perm
end

"""
    matop_along_dim!(buffer, mat, arr::AbstractArray, dim::Integer, op)

Apply the operator `op` to the matrix `mat` and to the array `arr` along the dimension
`dim`, writing the result to `buffer`.
"""
function matop_along_dim!(buffer, mat, arr::AbstractArray{T,N}, dim, op) where {T,N}
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))

    if dim == 1
        matop!(buffer, mat, arr, op, 1)
    elseif dim != N
        # Move the target dim to the first position
        perm = getperm(N, dim => 1)
        arr_perm = permutedims(arr, perm)
        buffer_perm = permutedims(buffer, perm)
        matop!(buffer_perm, mat, arr_perm, op, 1)
        permutedims!(buffer, buffer_perm, getperm(N, 1 => dim))
    else
        # Apply the operator to the last dimension
        matop!(buffer, mat, arr, op, N)
    end
    return buffer
end

"""
    matop!(buffer, mat, arr::AbstractArray, op, dim)

Apply the operator `op` to the matrix `mat` and to the array `arr` along the first
dimension (dim=1) or the last dimension (dim=N).
"""
function matop!(
    buffer::AbstractArray{S,N}, mat, arr::AbstractArray{T,N}, op, dim
) where {S,T,N}
    if dim == 1
        flatarr = reshape(arr, (size(arr, 1), :))
        flatbuffer = reshape(buffer, (size(buffer, 1), :))
        op(flatbuffer, mat, flatarr)
    elseif dim == N
        flatarr = reshape(arr, (:, size(arr, N)))
        flatbuffer = reshape(buffer, (:, size(buffer, N)))
        op(flatbuffer, flatarr, transpose(mat))
    else
        throw(DomainError("Dimension must be 1 or $N"))
    end
    return buffer
end

function div_noalloc!(
    buffer::AbstractArray{S,N}, mat::SVD, arr::AbstractArray{T,N}, workarr, dim
) where {S,T,N}
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))

    if dim == 1
        flatarr = reshape(arr, (size(arr, 1), :))
        flatbuffer = reshape(buffer, (size(buffer, 1), :))
        ldiv_noalloc!(flatbuffer, mat, flatarr, workarr)
    elseif dim != N
        # Move the target dim to the first position
        arr_perm = movedim(arr, dim => 1)
        buffer_perm = movedim(buffer, dim => 1)
        flatarr = reshape(arr_perm, (size(arr_perm, 1), :))
        flatbuffer = reshape(buffer_perm, (size(buffer_perm, 1), :))
        ldiv_noalloc!(flatbuffer, mat, flatarr, workarr)
        buffer .= movedim(buffer_perm, 1 => dim)
    else
        flatarr = reshape(arr, (:, size(arr, N)))
        flatbuffer = reshape(buffer, (:, size(buffer, N)))
        rdiv_noalloc!(flatbuffer, flatarr, mat, workarr)
    end
    return buffer
end

function ldiv_noalloc!(Y::AbstractMatrix, A::SVD, B::AbstractMatrix, workarr)
    # Setup work space
    worksize = (size(A.U, 2), size(B, 2))
    worklength = prod(worksize)
    length(workarr) ≥ worklength ||
        throw(DimensionMismatch("size(workarr)=$(size(workarr)), min worksize=$worklength"))
    workarr_view = reshape(view(workarr, 1:worklength), worksize)

    mul!(workarr_view, A.U', B)
    workarr_view ./= A.S
    return mul!(Y, A.V, workarr_view)
end

function rdiv_noalloc!(Y::AbstractMatrix, A::AbstractMatrix, B::SVD, workarr)
    # Setup work space
    worksize = (size(A, 1), size(B.U, 2))
    worklength = prod(worksize)
    length(workarr) ≥ worklength ||
        throw(DimensionMismatch("size(workarr)=$(size(workarr)), min worksize=$worklength"))
    workarr_view = reshape(view(workarr, 1:worklength), worksize)

    # Note: conj creates a temporary matrix
    mul!(workarr_view, A, conj(B.U))
    workarr_view ./= reshape(B.S, 1, :)
    return mul!(Y, workarr_view, conj(B.Vt))
end
