"""
    CompositeBasisFunction

Union of several basis functions in the imaginary-time/real-frequency domain
domains
"""
struct CompositeBasisFunction
    polys::Vector{Any}
end

"""
    (::CompositeBasisFunction)(x::Real)

Evaluate basis function at position `x`
"""
function (obj::CompositeBasisFunction)(x::Real)
    return vcat(p(x) for p in obj.polys)
end

function (obj::CompositeBasisFunction)(x::Vector{T}) where {T<:Real}
    return reduce(vcat, p(x) for p in obj.polys)
end

"""
    CompositeBasisFunctionFT

Union of several basis functions in the imaginary-frequency domain
"""
struct CompositeBasisFunctionFT
    polys::Vector{Any}
end

"""
Evaluate basis function at frequency n
"""
function (obj::CompositeBasisFunctionFT)(n::Union{MatsubaraFreq,
                                                  AbstractVector{MatsubaraFreq}})
    return hcat(p(n) for p in obj.polys)
end

(obj::CompositeBasisFunctionFT)(n::Integer)                 = obj(MatsubaraFreq(n))
(obj::CompositeBasisFunctionFT)(n::AbstractVector{Integer}) = obj(MatsubaraFreq.(n))

struct CompositeBasis <: AbstractBasis
    beta  :: Float64
    bases :: Vector{AbstractBasis}
    u     :: Union{CompositeBasisFunction,Nothing}
    v     :: Union{CompositeBasisFunction,Nothing}
    uhat  :: Union{CompositeBasisFunctionFT,Nothing}
end

iswellconditioned(basis::CompositeBasis) = false

function collect_polys(::Type{T}, polys) where {T}
    if any(isnothing, polys)
        return nothing
    else
        return T(polys)
    end
end

function CompositeBasis(bases::Vector{AbstractBasis})
    u = CompositeBasisFunction([b.u for b in bases])
    v = collect_polys(CompositeBasisFunction, [b.v for b in bases])
    uhat = collect_polys(CompositeBasisFunctionFT, [b.uhat for b in bases])
    return CompositeBasis(getbeta(first(bases)), bases, u, v, uhat)
end

function default_tau_sampling_points(basis::CompositeBasis)
    return sort!(unique!(mapreduce(default_tau_sampling_points, vcat, basis.bases)))
end

function default_matsubara_sampling_points(basis::CompositeBasis; mitigate=true)
    return sort!(unique!(mapreduce(b -> default_matsubara_sampling_points(b; mitigate),
                                   vcat, basis.bases)))
end

significance(self::CompositeBasis) = vcat(map(significance, self.bases)...)
