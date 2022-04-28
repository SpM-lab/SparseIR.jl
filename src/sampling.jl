using LinearAlgebra: svd, SVD

export TauSampling, MatsubaraSampling, evaluate, fit

"""
    AbstractSampling

Abstract class for sparse sampling.

Encodes the "basis transformation" of a propagator from the truncated IR
basis coefficients `G_ir[l]` to time/frequency sampled on sparse points
`G(x[i])` together with its inverse, a least squares fit:

         ________________                   ___________________
        |                |    evaluate     |                   |
        |     Basis      |---------------->|     Value on      |
        |  coefficients  |<----------------|  sampling points  |
        |________________|      fit        |___________________|

"""
abstract type AbstractSampling end

"""
    cond(sampling)

Condition number of the fitting problem.
"""
cond(sampling::AbstractSampling) = first(sampling.matrix.S) / last(sampling.matrix.S)

"""
    TauSampling <: AbstractSampling

Sparse sampling in imaginary time.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary time.
"""
struct TauSampling{T,B<:AbstractBasis,S,Sr} <: AbstractSampling
    sampling_points::Vector{T}
    basis::B
    matrix::SVD{S,Sr,Matrix{S}}
    matrixfull::Matrix{S}
end

"""
    TauSampling(basis, sampling_points)

Construct a `TauSampling` object.
"""
function TauSampling(basis, sampling_points=default_tau_sampling_points(basis))
    matrixfull = eval_matrix(TauSampling, basis, sampling_points)
    sampling = TauSampling(sampling_points, basis, svd(matrixfull), matrixfull)

    if iswellconditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end

    return sampling
end

"""
    MatsubaraSampling <: AbstractSampling

Sparse sampling in Matsubara frequencies.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary frequencies.
"""
struct MatsubaraSampling{T,B<:AbstractBasis,S,Sr} <: AbstractSampling
    sampling_points::Vector{T}
    basis::B
    matrix::SVD{S,Sr,Matrix{S}}
    matrixfull::Matrix{S}
end

"""
    MatsubaraSampling(basis, sampling_points)

Construct a `MatsubaraSampling` object.
"""
function MatsubaraSampling(basis, sampling_points=default_matsubara_sampling_points(basis))
    matrixfull = eval_matrix(MatsubaraSampling, basis, sampling_points)
    sampling = MatsubaraSampling(sampling_points, basis, svd(matrixfull), matrixfull)

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
    evaluate(sampling, al)

Evaluate the basis coefficients at the sparse sampling points.
"""
function evaluate(smpl::AbstractSampling, al; dim::Integer=1)
    return matop_along_dim(smpl.matrixfull, al, dim)
end
# More elegant but slower:
# evaluate(smpl::AbstractSampling, al; dims=1) = mapslices(i -> smpl.matrixfull * i, al; dims)

"""
    fit(sampling, al)

Fit basis coefficients from the sparse sampling points
"""
fit(smpl::AbstractSampling, al; dims=1) = mapslices(i -> smpl.matrix \ i, al; dims)

"""
    movedim(arr::AbstractArray, src => dst)

Move `arr`'s dimension at `src` to `dst` while keeping the order of the remaining 
dimensions unchanged.
"""
function movedim(arr::AbstractArray{T,N}, dims::Pair) where {T,N}
    src, dst = dims
    src == dst && return arr

    perm = collect(1:N)
    deleteat!(perm, src)
    insert!(perm, dst, src)
    return permutedims(arr, perm)
end

"""
    matop_along_dim(op::AbstractMatrix, arr::AbstractArray, dim::Integer)

Apply the matrix operator `op` to the array `arr` along the dimension `dim`.
"""
function matop_along_dim(op::AbstractMatrix, arr::AbstractArray{T,N},
                          dim::Integer) where {T,N}
    # Move the target dim to the first position
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))
    size(arr, dim) == size(op, 2) || error("Dimension mismatch!")

    arr = movedim(arr, dim => 1)
    return movedim(matop(op, arr), 1 => dim)
end

"""
    matop(op::AbstractMatrix, arr::AbstractArray)

Apply `op` to the first dimension of `arr`.
"""
function matop(op::AbstractMatrix, arr::AbstractArray{T,N}) where {T,N}
    size(op, 2) == size(arr, 1) || error("Dimension mismatch!")
    N == 1 && return op * arr

    flatarr = reshape(arr, (size(arr, 1), :))
    return reshape(op * flatarr, (:, size(arr)[2:end]...))
end
