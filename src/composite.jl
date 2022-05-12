"""
Union of several basis functions in the imaginary-time/real-frequency domain
domains
"""
struct CompositeBasisFunction
    polys::Vector{Any}
end

"""
Evaluate basis function at position x
"""
function (obj::CompositeBasisFunction)(x::Real)
    return vcat((p(x) for p in obj.polys))
end

function (obj::CompositeBasisFunction)(x::Vector{T}) where {T<:Real}
    return vcat((p(x) for p in obj.polys)...)
end

"""
Union of several basis functions in the imaginary-frequency domain
domains
"""
struct CompositeBasisFunctionFT
    polys::Vector{Any}
end

"""
Evaluate basis function at frequency n
"""
function (obj::CompositeBasisFunctionFT)(n::Union{Int,Vector{Int}})
    return hcat((p(n) for p in obj.polys))
end

struct CompositeBasis <: AbstractBasis
    beta::Float64
    bases::Vector{AbstractBasis}
    u::Union{CompositeBasisFunction,Nothing}
    v::Union{CompositeBasisFunction,Nothing}
    uhat::Union{CompositeBasisFunctionFT,Nothing}
end

iswellconditioned(basis::CompositeBasis) = false

function _collect_polys(::Type{T}, polys) where {T}
    if any((p === nothing for p in polys))
        return nothing
    else
        return T([p for p in polys])
    end
end

function CompositeBasis(bases::Vector{AbstractBasis})
    u = CompositeBasisFunction([b.u for b in bases])
    v = _collect_polys(CompositeBasisFunction, [b.v for b in bases])
    uhat = _collect_polys(CompositeBasisFunctionFT, [b.uhat for b in bases])
    return CompositeBasis(getbeta(bases[1]), bases, u, v, uhat)
end

function default_tau_sampling_points(basis::CompositeBasis)
    return sort(unique(vcat((default_tau_sampling_points(b) for b in basis.bases)...)))
end

function default_matsubara_sampling_points(basis::CompositeBasis; mitigate=true)
    return sort(
        unique(
            vcat(
                (
                    default_matsubara_sampling_points(b; mitigate=mitigate) for
                    b in basis.bases
                )...,
            ),
        ),
    )
end
