struct MatsubaraPoleBasis <: AbstractBasis
    β::Float64
    poles::Vector{Float64}
end

function (basis::MatsubaraPoleBasis)(n::Vector{T}) where {T<:Integer}
    iv = (im * π / getbeta(basis)) .* n
    return 1 ./ (transpose(iv) .- basis.poles)
end

struct TauPoleBasis <: AbstractBasis
    β::Float64
    poles::Vector{Float64}
    statistics::Statistics
    wmax::Float64
end

getwmax(basis::TauPoleBasis) = basis.wmax

function TauPoleBasis(beta::Real, statistics::Statistics, poles::Vector{<:AbstractFloat})
    return TauPoleBasis(beta, poles, statistics, maximum(abs, poles))
end

function (basis::TauPoleBasis)(tau::Vector{<:AbstractFloat})
    all(τ -> 0 ≤ τ ≤ getbeta(basis), tau) ||
        throw(DomainError(tau, "tau must be in [0, beta]!"))

    x = (2 / getbeta(basis)) .* tau .- 1
    y = basis.poles ./ getwmax(basis)
    Λ = getbeta(basis) * getwmax(basis)
    if basis.statistics == fermion
        res = -LogisticKernel(Λ).(x, transpose(y))
    else
        K = RegularizedBoseKernel(Λ)
        res = -K(x, transpose(y)) ./ transpose(y)
    end
    return transpose(res)
end

"""
Sparse pole representation
"""
struct SparsePoleRepresentation{T<:AbstractFloat} <: AbstractBasis
    basis::AbstractBasis
    poles::Vector{T}
    u::TauPoleBasis
    uhat::MatsubaraPoleBasis
    statistics::Statistics
    fitmat::Matrix{Float64}
    matrix::SVD
end

function SparsePoleRepresentation(
    basis::AbstractBasis, poles=default_omega_sampling_points(basis)
)
    y_sampling_points = poles ./ getwmax(basis)
    u = TauPoleBasis(getbeta(basis), basis.statistics, poles)
    uhat = MatsubaraPoleBasis(getbeta(basis), poles)
    weight = weight_func(basis.kernel, basis.statistics)(y_sampling_points)
    fitmat = -basis.s .* basis.v(poles) .* transpose(weight)
    matrix = svd(fitmat)
    return SparsePoleRepresentation(basis, poles, u, uhat, basis.statistics, fitmat, matrix)
end

getbeta(obj::SparsePoleRepresentation) = getbeta(obj.basis)

function Base.getproperty(obj::SparsePoleRepresentation, d::Symbol)
    if d === :v
        return nothing
    else
        return getfield(obj, d)
    end
end

function default_tau_sampling_points(obj::SparsePoleRepresentation)
    return default_tau_sampling_points(obj.basis)
end

function default_matsubara_sampling_points(obj::SparsePoleRepresentation)
    return default_matsubara_sampling_points(obj.basis)
end

iswellconditioned(::SparsePoleRepresentation) = false

"""
From IR to SPR

gl:
    Expansion coefficients in IR
"""
function from_IR(spr::SparsePoleRepresentation, gl::AbstractArray, dims=1)
    return mapslices(i -> spr.matrix \ i, gl; dims)
end

"""
From SPR to IR

g_spr:
    Expansion coefficients in SPR
"""
function to_IR(spr::SparsePoleRepresentation, g_spr::AbstractArray, dims=1)
    return mapslices(i -> spr.fitmat * i, g_spr; dims)
end
