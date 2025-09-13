"""
    PiecewiseLegendrePoly <: Function

Piecewise Legendre polynomial.

Models a function on the interval ``[xmin, xmax]`` as a set of segments on the
intervals ``S[i] = [a[i], a[i+1]]``, where on each interval the function
is expanded in scaled Legendre polynomials.
"""
mutable struct PiecewiseLegendrePoly
    ptr::Ptr{spir_funcs}
    xmin::Float64
    xmax::Float64
    function PiecewiseLegendrePoly(funcs::Ptr{spir_funcs}, xmin::Float64, xmax::Float64)
        result = new(funcs, xmin, xmax)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

"""
    PiecewiseLegendrePolyVector

Contains a `Vector{PiecewiseLegendrePoly}`.
"""
mutable struct PiecewiseLegendrePolyVector
    ptr::Ptr{spir_funcs}
    xmin::Float64
    xmax::Float64
    function PiecewiseLegendrePolyVector(
            funcs::Ptr{spir_funcs}, xmin::Float64, xmax::Float64)
        result = new(funcs, xmin, xmax)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

"""
    PiecewiseLegendreFTVector

Fourier transform of piecewise Legendre polynomials.

For a given frequency index `n`, the Fourier transform of the Legendre
function is defined as:

        p̂(n) == ∫ dx exp(im * π * n * x / (xmax - xmin)) p(x)
"""
mutable struct PiecewiseLegendreFTVector
    ptr::Ptr{spir_funcs}
    xmin::Float64
    xmax::Float64

    function PiecewiseLegendreFTVector(funcs::Ptr{spir_funcs})
        xmin = -1.0
        xmax = 1.0
        result = new(funcs, xmin, xmax)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

function (polys::PiecewiseLegendrePoly)(x::Real)
    sz = Ref{Int32}(-1)
    spir_funcs_get_size(polys.ptr, sz) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get funcs size")
    ret = Vector{Float64}(undef, Int(sz[]))
    spir_funcs_eval(polys.ptr, x, ret) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to evaluate funcs")
    return only(ret)
end

function (polys::PiecewiseLegendrePolyVector)(x::Real)
    sz = Ref{Int32}(-1)
    spir_funcs_get_size(polys.ptr, sz) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get funcs size")
    ret = Vector{Float64}(undef, Int(sz[]))
    spir_funcs_eval(polys.ptr, x, ret) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to evaluate funcs")
    return ret
end

function (polys::PiecewiseLegendrePolyVector)(x::AbstractVector)
    hcat(polys.(x)...)
end

function (polys::PiecewiseLegendreFTVector)(freq::MatsubaraFreq)
    n = freq.n
    sz = Ref{Int32}(-1)
    spir_funcs_get_size(polys.ptr, sz) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get funcs size")
    ret = Vector{ComplexF64}(undef, Int(sz[]))
    spir_funcs_eval_matsu(polys.ptr, n, ret) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to evaluate funcs")
    return ret
end

function (polys::PiecewiseLegendreFTVector)(x::AbstractVector)
    hcat(polys.(x)...)
end

function Base.getindex(funcs::Ptr{spir_funcs}, i::Int)
    status = Ref{Int32}(-100)
    indices = Vector{Int32}(undef, 1)
    indices[1] = i - 1 # Julia indices are 1-based, C indices are 0-based
    ret = spir_funcs_get_slice(funcs, 1, indices, status)
    status[] == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get basis function for index $i: $(status[])")
    return ret
end

function Base.getindex(polys::PiecewiseLegendrePolyVector, i::Int)
    return PiecewiseLegendrePoly(polys.ptr[i], polys.xmin, polys.xmax)
end

# Base.getindex(funcs::Ptr{spir_funcs}, I) = [funcs[i] for i in I]
function Base.getindex(polys::PiecewiseLegendrePolyVector, I)
    PiecewiseLegendrePoly[polys[i]
                          for i in I]
end

function Base.length(funcs::Ptr{spir_funcs})
    sz = Ref{Int32}(-1)
    spir_funcs_get_size(funcs, sz) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get funcs size")
    return Int(sz[])
end

function Base.length(polys::PiecewiseLegendrePolyVector)
    return length(polys.ptr)
end

Base.firstindex(funcs::Ptr{spir_funcs}) = 1
Base.lastindex(funcs::Ptr{spir_funcs}) = length(funcs)

Base.firstindex(polys::PiecewiseLegendrePolyVector) = firstindex(polys.ptr)
Base.lastindex(polys::PiecewiseLegendrePolyVector) = lastindex(polys.ptr)

function roots(poly::PiecewiseLegendrePoly)
    nroots_ref = Ref{Int32}(-1)
    spir_funcs_get_n_roots(poly.ptr, nroots_ref)
    nroots = nroots_ref[]

    out = Vector{Float64}(undef, nroots)

    SparseIR.C_API.spir_funcs_get_roots(
        poly.ptr, out
    )
    return out
end

function roots(poly::PiecewiseLegendrePolyVector)
    nroots_ref = Ref{Int32}(-1)
    spir_funcs_get_n_roots(poly.ptr, nroots_ref)
    nroots = nroots_ref[]

    out = Vector{Float64}(undef, nroots)

    SparseIR.C_API.spir_funcs_get_roots(
        poly.ptr, out
    )
    return out
end

function overlap(poly::PiecewiseLegendrePolyVector, f::F) where {F}
    xmin = poly.xmin
    xmax = poly.xmax
    pts = filter(x -> xmin ≤ x ≤ xmax, roots(poly))
    q,
    _ = quadgk(
        x -> poly(x) * f(x),
        unique!(sort!(vcat(pts, [xmin, xmax])));
        rtol=eps(), order=10, maxevals=10^4)
    q
end

function overlap(poly::PiecewiseLegendrePoly, f::F) where {F}
    xmin = poly.xmin
    xmax = poly.xmax
    pts = filter(x -> xmin ≤ x ≤ xmax, roots(poly))
    q,
    _ = quadgk(
        x -> poly(x) * f(x),
        unique!(sort!(vcat(pts, [xmin, xmax])));
        rtol=eps(), order=10, maxevals=10^4)
    return q
end

function xmin(poly::PiecewiseLegendrePoly)
    return poly.xmin
end

function xmax(poly::PiecewiseLegendrePoly)
    return poly.xmax
end

function xmin(poly::PiecewiseLegendrePolyVector)
    return poly.xmin
end

function xmax(poly::PiecewiseLegendrePolyVector)
    return poly.xmax
end

function xmin(poly::PiecewiseLegendreFTVector)
    return poly.xmin
end

function xmax(poly::PiecewiseLegendreFTVector)
    return poly.xmax
end
