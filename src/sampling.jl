using LinearAlgebra: svd, SVD

export TauSampling, MatsubaraSampling, evaluate, fit
export evaluate_opt

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

# TODO: implement axis
"""
    evaluate(sampling, al)

Evaluate the basis coefficients at the sparse sampling points.
"""
evaluate(smpl::AbstractSampling, al; dims=1) = mapslices(i -> smpl.matrixfull * i, al; dims)

"""
    fit(sampling, al)

Fit basis coefficients from the sparse sampling points
"""
fit(smpl::AbstractSampling, al; dims=1) = mapslices(i -> smpl.matrix \ i, al; dims)

"""
Move the axis at src to dst with keeping the order of the rest axes unchanged.

src=1, dst=2, N=4, perm=[2, 1, 3, 4]
src=2, dst=4, N=4, perm=[1, 3, 4, 2]
"""
function move_axis(arr::AbstractArray{T,N}, src::Int, dst::Int) where {T,N}
    src == dst && return arr

    perm = collect(1:N)
    deleteat!(perm, src)
    insert!(perm, dst, src)
    return permutedims(arr, perm)
end


"""
Apply a matrix operator to an array along a given axis
"""
function matop_along_axis(op::AbstractMatrix{T}, arr::AbstractArray{S,N}, axis::Int64) where {T,S,N}
    # Move the target axis to the first position
    (axis < 0 || axis > N) && throw(DomainError("axis must be in [1,N]"))
    size(arr)[axis] != size(op)[2] && error("Dimension mismatch!")

    arr = move_axis(arr, axis, 1)
    return move_axis(matop(op, arr), 1, axis)
end

"""
Apply op to the first axis of an array
"""
function matop(op::AbstractMatrix{T}, arr::AbstractArray{S,N}) where {T,S,N}
    (size(arr)[1] != size(op)[2]) && error("Dimension mismatch!")
    N == 1 && return op * arr

    rest_dims = size(arr)[2:end]
    arr = reshape(arr, (size(op)[2], prod(rest_dims)))
    return reshape(op * arr, (size(op)[1], rest_dims...))
end


"""
BLAS version of evaluate
"""
evaluate_opt(smpl::AbstractSampling, al; dims::Int=1) = matop_along_axis(smpl.matrixfull, al, dims)