export LegendreBasis

@doc raw"""Legendre basis

In the original paper [L. Boehnke et al., PRB 84, 075145 (2011)],
they used:

    G(\tau) = \sum_{l=0} \sqrt{2l+1} P_l[x(\tau)] G_l/beta,

where P_l[x] is the $l$-th Legendre polynomial.

In this class, the basis functions are defined by

    U_l(\tau) \equiv c_l (\sqrt{2l+1}/beta) * P_l[x(\tau)],

where c_l are additional l-depenent constant factors.
By default, we take c_l = 1, which reduces to the original definition.
"""
struct LegendreBasis{T<:AbstractFloat} <: AbstractBasis
    statistics::Statistics
    β::Float64
    cl::Vector{T}
    u::PiecewiseLegendrePolyArray{T}
    uhat::PiecewiseLegendreFTArray{T}
end

function LegendreBasis(statistics::Statistics, beta::Float64, size::Int64; cl::Vector{Float64}=ones(Float64, size))
    beta > 0 || error("inverse temperature beta must be positive! $(beta)")
    size > 0 || error("size of basis must be positive")

    # u
    knots = Float64[0, beta]
    data = zeros(Float64, size, length(knots)-1, size)
    symm = (-1).^collect(0:(size-1))
    for l in 1:size
        data[l, 1, l] = sqrt(((l-1)+0.5)/beta) * cl[l]
    end
    u = PiecewiseLegendrePolyArray(data, knots, symm=symm)

    # uhat
    uhat_base = PiecewiseLegendrePolyArray(sqrt(beta) .* data, Float64[-1,1], symm=symm)
    even_odd = Dict(fermion => :odd, boson => :even)[statistics]
    uhat = hat.(uhat_base, even_odd, 0:size-1)

    return LegendreBasis(statistics, beta, cl, u, uhat)
end

function Base.getproperty(obj::LegendreBasis, d::Symbol)
    if d === :size
        return length(getfield(obj, :u))
    elseif d === :v
        return nothing
    elseif d === :beta # backward compatibility
        return getfield(obj, :β)
    else
        return getfield(obj, d)
    end
end

iswellconditioned(basis::LegendreBasis) = true

function default_tau_sampling_points(basis::LegendreBasis)
    x = gauss(length(basis.u))[1]
    return (basis.β/2) .* (x .+ 1)
end

struct _ConstTerm{T<:Number}
    value::T
end



"""
Return value for given frequencies
"""
function (ct::_ConstTerm)(n::Vector{T}) where {T <: Integer}
    return fill(ct.value, (1, length(n)))
end

function (ct::_ConstTerm)(n::T) where {T <: Integer}
    return ct([n])
end


"""
Constant term in matsubara-frequency domain
"""
struct MatsubaraConstBasis{T<:AbstractFloat} <: AbstractBasis
    statistics::Statistics
    β::Float64
    uhat::_ConstTerm{T}
end

function MatsubaraConstBasis(statistics::Statistics, beta::Float64, value=1) where {T<:AbstractFloat}
    beta > 0 || error("inverse temperature beta must be positive")
    MatsubaraConstBasis(statistics, beta, _ConstTerm(value))
end


function Base.getproperty(obj::MatsubaraConstBasis, d::Symbol)
    if d == :size
        return 1
    else
        return getfield(obj, d)
    end
end

function Base.propertynames(::MatsubaraConstBasis, private::Bool=false)
    return (:size, fieldnames(MatsubaraConstBasis, private)...)
end