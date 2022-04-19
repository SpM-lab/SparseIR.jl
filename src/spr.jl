export SparsePoleRepresentation

import LinearAlgebra: svd, SVD
struct MatsubaraPoleBasis
    beta::Float64
    poles::Vector{Float64}
end

"""
Evaluate basis functions at given frequency n
"""
function (basis::MatsubaraPoleBasis)(n::Vector{T}) where {T<:Integer}
    iv = (im * π / basis.beta) .* n
    return 1 ./ (iv[newaxis, :] - basis.poles[:, newaxis])
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
    !all(0 .≤ tau .≤ basis.beta) || error("tau must be in [0, beta]!")

    x = (2 / basis.beta) .* tau .- 1
    y = basis.poles ./ basis.wmax
    Λ = basis.beta * basis.wmax
    if basis.statistics == fermion
        res = -LogisticKernel(lambda_)(x[:, newaxis], y[newaxis, :])
    else
        K = RegularizedBoseKernel(Λ)
        res = -K(x[:, newaxis], y[newaxis, :]) ./ y[newaxis, :]
    end
    return transpose(res)
end

"""
Sparse pole representation
"""
struct SparsePoleRepresentation{T} <: AbstractBasis
    basis::AbstractBasis
    poles::Vector{T}
    u::TauPoleBasis
    uhat::MatsubaraPoleBasis
    statistics::Statistics
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
    println(length(poles))
    println(size(basis.s))
    println(size(basis.v(poles)))
    println(size(weight), typeof(weight))
    fit_mat = -1 .* basis.s[:,newaxis] .* basis.v(poles) .* weight[newaxis,:]
    println(typeof(fit_mat), size(fit_mat))
    matrix = svd(fit_mat)
    return SparsePoleRepresentation(basis, poles, u, uhat, basis.statistics, matrix)
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

is_well_conditioned(obj::SparsePoleRepresentation) = false

#===
TODO: implement to_IR/from_IR once axis is implemented in evaluate/fit in sampling.jl
"""
From IR to SPR

gl:
    Expansion coefficients in IR
"""
function from_IR(spr::SparsePoleRepresentation,
                 gl::Array{T,N}, axis::Int64=1)::Array{ComplexF64,N} where {T,N}
    return spr.o.from_IR(gl, axis - 1)
end

"""
From SPR to IR

g_spr:
    Expansion coefficients in SPR
"""
function to_IR(spr::SparsePoleRepresentation,
               g_spr::Array{T,N}, axis::Int64=1)::Array{ComplexF64,N} where {T,N}
    return spr.o.to_IR(g_spr, axis - 1)
end
===#