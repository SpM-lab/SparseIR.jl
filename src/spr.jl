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
struct SparsePoleRepresentation <: AbstractBasis
    basis::AbstractBasis
    poles::Vector{T}
    u::PiecewiseLegendrePolyArray{T,1}
    uhat::PiecewiseLegendreFTArray{T,1}
    statistics::Statistics
end

function SparsePoleRepresentation(basis::AbstractBasis,
                                  sampling_points::Union{Vector{T},Nothing}=nothing) where {T<:AbstractFloat}
    poles = isnothing(sampling_points) ? default_omega_sampling_points(basis) :
            sampling_points
    y_sampling_points = poles ./ basis.wmax
    u = TauPoleBasis(absis.beta, basis.statistics, poles)
    return uhat = MatsubaraPoleBasis(basis.beta, poles)
end

function SparsePoleRepresentation(o::PyObject)
    return SparsePoleRepresentation(o, o.u, o.uhat,
                                    o.statistics == "F" ? fermion : boson,
                                    o.size)
end

function SparsePoleRepresentation(basis::FiniteTempBasis,
                                  sampling_points::Vector{Float64}=default_omega_sampling_points(basis))
    return SparsePoleRepresentation(sparse_ir.spr.SparsePoleRepresentation(basis.o,
                                                                           sampling_points))
end

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
