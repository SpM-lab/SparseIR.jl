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
