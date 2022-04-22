export SparsePoleRepresentation, to_IR, from_IR

using LinearAlgebra: svd, SVD
struct MatsubaraPoleBasis
    beta::Float64
    poles::Vector{Float64}
end

"""
Evaluate basis functions at given frequency n
"""
function (basis::MatsubaraPoleBasis)(n::Vector{T}) where {T<:Integer}
    iv = (im * π / basis.beta) .* n
    return 1 ./ (iv[newaxis, :] .- basis.poles[:, newaxis])
end

struct TauPoleBasis
    beta::Float64
    poles::Vector{Float64}
    statistics::Statistics
    wmax::Float64
end

function TauPoleBasis(beta::Real, statistics::Statistics,
                      poles::Vector{T}) where {T<:AbstractFloat}
    return TauPoleBasis(beta, poles, statistics, maximum(abs.(poles)))
end

"""
Evaluate basis functions at tau
"""
function (basis::TauPoleBasis)(tau::Vector{T}) where {T<:AbstractFloat}
    all(0 .≤ tau .≤ basis.beta) || error("tau must be in [0, beta]!")

    x = (2 / basis.beta) .* tau .- 1
    y = basis.poles ./ basis.wmax
    Λ = basis.beta * basis.wmax
    if basis.statistics == fermion
        res = -LogisticKernel(Λ).(x[:, newaxis], y[newaxis, :])
    else
        K = RegularizedBoseKernel(Λ)
        res = -K(x[:, newaxis], y[newaxis, :]) ./ y[newaxis, :]
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

function SparsePoleRepresentation(basis::AbstractBasis,
                                  sampling_points::Union{Vector{T},Nothing}=nothing) where {T<:AbstractFloat}
    poles = isnothing(sampling_points) ? default_omega_sampling_points(basis) :
            sampling_points
    y_sampling_points = poles ./ wmax(basis)
    u = TauPoleBasis(basis.beta, basis.statistics, poles)
    uhat = MatsubaraPoleBasis(basis.beta, poles)
    weight = weight_func(basis.kernel, basis.statistics)(y_sampling_points)
    fitmat = -1 .* basis.s[:,newaxis] .* basis.v(poles) .* weight[newaxis,:]
    matrix = svd(fitmat)
    return SparsePoleRepresentation(basis, poles, u, uhat, basis.statistics, fitmat, matrix)
end

function Base.getproperty(obj::SparsePoleRepresentation, d::Symbol)
    if d === :size
        return length(getfield(obj, :poles))
    elseif d === :v
        return nothing
    elseif d === :wmax
        return getfield(obj, :u).wmax
    elseif d === :beta || d === :β
        return getfield(obj, :basis).beta
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

iswellconditioned(obj::SparsePoleRepresentation{T}) where {T<:AbstractFloat} = false

"""
From IR to SPR

gl:
    Expansion coefficients in IR
"""
function from_IR(spr::SparsePoleRepresentation,
                 gl::Array{T,N}, dims::Int=1) where {T,N}
    return mapslices(i -> spr.matrix \ i, gl; dims)
end

"""
From SPR to IR

g_spr:
    Expansion coefficients in SPR
"""
function to_IR(spr::SparsePoleRepresentation,
               g_spr::Array{T,N}, dims::Int=1) where {T,N}
    return mapslices(i -> spr.fitmat * i, g_spr; dims)
end