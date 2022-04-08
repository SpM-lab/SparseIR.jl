import LinearAlgebra: svd, SVD

export TauSampling, MatsubaraSampling, evaluate, fit

abstract type AbstractSampling end

cond(sampling::AbstractSampling) = first(sampling.matrix.S) / last(sampling.matrix.S)

struct TauSampling{T<:Real,B<:AbstractBasis} <: AbstractSampling
    sampling_points::Vector{T}
    basis::B
    matrix::SVD
end

struct MatsubaraSampling{T<:Real,B<:AbstractBasis} <: AbstractSampling
    sampling_points::Vector{T}
    basis::B
    matrix::SVD
end

function TauSampling(basis, sampling_points=default_tau_sampling_points(basis))
    matrix = svd(eval_matrix(TauSampling, basis, sampling_points))
    sampling = TauSampling(sampling_points, basis, matrix)

    if is_well_conditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end

    return sampling
end

function MatsubaraSampling(basis, sampling_points=default_matsubara_sampling_points(basis))
    matrix = svd(eval_matrix(MatsubaraSampling, basis, sampling_points))
    sampling = MatsubaraSampling(sampling_points, basis, matrix)

    if is_well_conditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end

    return sampling
end

eval_matrix(::Type{TauSampling}, basis, x) = permutedims(basis.u(x))
eval_matrix(::Type{MatsubaraSampling}, basis, x) = permutedims(basis.uhat(x))

# TODO implement axis
evaluate(smpl::AbstractSampling, al, axis=nothing) = Matrix(smpl.matrix) * al
fit(smpl::AbstractSampling, al, axis=nothing) = smpl.matrix \ al

# struct DecomposedMatrix{T <: Number}
#     a::Matrix{T}
#     uH::Matrix{T}
#     s::Vector{T}
#     v::Matrix{T}
# end

# function get_svd_result(a, ε=Inf)
#     F = svd(a)
#     u, s, vH = F.U, F.S, F.Vt
#     if isinf(ε)
#         return u, s, vH
#     else
#         wher = s / first(s) .≤ ε
#         return u[:, wher], s[wher], vH[wher, :]
#     end
# end

# function DecomposedMatrix(a, svd_result=get_svd_result(a))
#     ndims(a) == 2 || error("a must be a matrix")
#     u, s, vH = svd_result

#     return DecomposedMatrix(a, u', s, v')
# end