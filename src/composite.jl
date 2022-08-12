"""
    CompositeBasisFunction <: AbstractCompositeBasisFunction

Union of several basis functions in the imaginary-time/real-frequency domain
domains.
"""
struct CompositeBasisFunction{T<:Tuple} <: AbstractCompositeBasisFunction
    polys::T
end
CompositeBasisFunction(polys) = CompositeBasisFunction(Tuple(polys))

"""
    (::CompositeBasisFunction)(x::Real)

Evaluate basis function at position `x`
"""
(obj::CompositeBasisFunction)(x::Real) = mapreduce(p -> p(x), vcat, obj.polys)
(obj::CompositeBasisFunction)(x::AbstractVector{<:Real}) =
    mapreduce(p -> p(x), vcat, obj.polys)

"""
    CompositeBasisFunctionFT <: AbstractCompositeBasisFunction

Union of several basis functions in the imaginary-frequency domain
"""
struct CompositeBasisFunctionFT{T<:Tuple} <: AbstractCompositeBasisFunction
    polys::T
end
CompositeBasisFunctionFT(polys) = CompositeBasisFunctionFT(Tuple(polys))

"""
    (::CompositeBasisFunctionFT)(n::MatsubaraFreq)

Evaluate basis function at frequency `n`.
"""
(obj::CompositeBasisFunctionFT)(x::MatsubaraFreq) = mapreduce(p -> p(x), vcat, obj.polys)
(obj::CompositeBasisFunctionFT)(x::AbstractVector{<:MatsubaraFreq}) =
    mapreduce(p -> p(x), vcat, obj.polys)

(obj::CompositeBasisFunctionFT)(n::Integer) = obj(MatsubaraFreq(n))
(obj::CompositeBasisFunctionFT)(n::AbstractVector{<:Integer}) = obj(MatsubaraFreq.(n))

"""
    CompositeBasis <: AbstractBasis

Union of several basis sets.
"""
struct CompositeBasis{S<:Statistics,B<:Tuple{Vararg{<:AbstractBasis}},
                      CU<:CompositeBasisFunction, CUHAT<:CompositeBasisFunctionFT} <: AbstractBasis
    β          :: Float64
    statistics :: S
    bases      :: B
    u          :: CU
    uhat       :: CUHAT
end

CompositeBasis(bases) =
    CompositeBasis(only(unique(β(b) for b in bases)), only(unique(statistics(b) for b in bases)), 
                   Tuple(bases), CompositeBasisFunction(b.u for b in bases), 
                   CompositeBasisFunctionFT(b.uhat for b in bases))

iswellconditioned(basis::CompositeBasis) = false

# FIXME: this yields bad sampling points
function default_tau_sampling_points(basis::CompositeBasis)
    return sort!(unique!(mapreduce(default_tau_sampling_points, vcat, basis.bases)))
end

# FIXME: this yields bad sampling points
function default_matsubara_sampling_points(basis::CompositeBasis; mitigate=true)
    return sort!(unique!(mapreduce(b -> default_matsubara_sampling_points(b; mitigate),
                                   vcat, basis.bases)))
end

significance(self::CompositeBasis) = mapreduce(significance, vcat, self.bases)
