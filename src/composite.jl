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
(obj::CompositeBasisFunction)(x::Real) = mapreduce(p -> p(x), vcat, obj.polys)
(obj::CompositeBasisFunction)(x::AbstractVector{<:Real}) = mapreduce(p -> p(x), vcat, obj.polys)

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
(obj::CompositeBasisFunctionFT)(x::MatsubaraFreq) = mapreduce(p -> p(x), vcat, obj.polys)
(obj::CompositeBasisFunctionFT)(x::AbstractVector{<:MatsubaraFreq}) = mapreduce(p -> p(x), vcat, obj.polys)

(obj::CompositeBasisFunctionFT)(n::Integer) = obj(MatsubaraFreq(n))
(obj::CompositeBasisFunctionFT)(n::AbstractVector{<:Integer}) = obj(MatsubaraFreq.(n))

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

significance(self::CompositeBasis) = mapreduce(significance, vcat, self.bases)

getstatistics(self::CompositeBasis) = only(unique!(getstatistics.(self.bases)))
