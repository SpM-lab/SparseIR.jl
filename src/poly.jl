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
    period::Float64 # 0.0 for a non-periodic function, the period for a periodic function
    default_overlap_range::Tuple{Float64, Float64} # Default range for overlap calculations
    function PiecewiseLegendrePoly(
            funcs::Ptr{spir_funcs}, xmin::Float64, xmax::Float64, period::Float64,
            default_overlap_range::Union{Tuple{Float64, Float64}, Nothing}=nothing)
        default_range = default_overlap_range === nothing ? (xmin, xmax) : default_overlap_range
        result = new(funcs, xmin, xmax, period, default_range)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

Base.size(polys::PiecewiseLegendrePoly) = ()

"""
    PiecewiseLegendrePolyVector

Contains a `Vector{PiecewiseLegendrePoly}`.
"""
mutable struct PiecewiseLegendrePolyVector
    ptr::Ptr{spir_funcs}
    xmin::Float64
    xmax::Float64
    period::Float64 # 0.0 for a non-periodic function, the period for a periodic function
    default_overlap_range::Tuple{Float64, Float64} # Default range for overlap calculations
    function PiecewiseLegendrePolyVector(
            funcs::Ptr{spir_funcs}, xmin::Float64, xmax::Float64, period::Float64,
            default_overlap_range::Union{Tuple{Float64, Float64}, Nothing}=nothing)
        default_range = default_overlap_range === nothing ? (xmin, xmax) : default_overlap_range
        result = new(funcs, xmin, xmax, period, default_range)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

function Base.size(ptr::Ptr{spir_funcs})
    sz = Ref{Int32}(-1)
    spir_funcs_get_size(ptr, sz) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get funcs size")
    return Int(sz[])
end

Base.size(polys::PiecewiseLegendrePolyVector) = size(polys.ptr)

"""
    PiecewiseLegendreFTVector

Fourier transform of piecewise Legendre polynomials.

For a given frequency index `n`, the Fourier transform of the Legendre
function is defined as:

        p̂(n) == ∫ dx exp(im * π * n * x / (xmax - xmin)) p(x)
"""
mutable struct PiecewiseLegendreFTVector
    ptr::Ptr{spir_funcs}

    function PiecewiseLegendreFTVector(funcs::Ptr{spir_funcs})
        result = new(funcs)
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

Base.size(polys::PiecewiseLegendreFTVector) = size(polys.ptr)

function (polys::PiecewiseLegendreFTVector)(x::AbstractVector)
    n = length(x)
    sz = Ref{Int32}(-1)
    spir_funcs_get_size(polys.ptr, sz) == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get funcs size")
    n_basis = Int(sz[])
    isempty(x) && return Matrix{ComplexF64}(undef, n_basis, 0)
    result = Matrix{ComplexF64}(undef, n_basis, n)
    col = Vector{ComplexF64}(undef, n_basis)
    for i in 1:n
        spir_funcs_eval_matsu(polys.ptr, x[i].n, col) == SPIR_COMPUTATION_SUCCESS ||
            error("Failed to evaluate funcs")
        result[:, i] = col
    end
    return result
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

function Base.getindex(funcs::Ptr{spir_funcs}, indices::Vector{Int})
    all(indices .>= 1) || error("Indices must be at least 1")
    all(indices .<= size(funcs)) ||
        error("Indices must be less than or equal to the size of the functions")
    status = Ref{Int32}(-100)
    indices_i32 = Vector{Int32}(undef, length(indices))
    indices_i32 .= indices .- 1 # Julia indices are 1-based, C indices are 0-based
    ret = spir_funcs_get_slice(funcs, length(indices), indices_i32, status)
    status[] == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get basis function for index $indices: $(status[])")
    return ret
end

function Base.getindex(polys::PiecewiseLegendrePolyVector, i::Int)
    return PiecewiseLegendrePoly(polys.ptr[i], polys.xmin, polys.xmax, polys.period, polys.default_overlap_range)
end

function Base.getindex(polys::PiecewiseLegendrePolyVector,
        I)::Union{PiecewiseLegendrePoly,PiecewiseLegendrePolyVector}
    indices = collect(1:size(polys))[I]
    if indices isa Int
        return PiecewiseLegendrePoly(polys.ptr[indices], polys.xmin, polys.xmax, polys.period, polys.default_overlap_range)
    elseif length(indices) == 1
        return PiecewiseLegendrePoly(polys.ptr[indices[1]], polys.xmin, polys.xmax, polys.period, polys.default_overlap_range)
    else
        return PiecewiseLegendrePolyVector(polys.ptr[indices], polys.xmin, polys.xmax, polys.period, polys.default_overlap_range)
    end
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

function knots(poly::PiecewiseLegendrePoly)
    nknots_ref = Ref{Int32}(-1)
    spir_funcs_get_n_knots(poly.ptr, nknots_ref)
    nknots = nknots_ref[]

    out = Vector{Float64}(undef, nknots)

    SparseIR.C_API.spir_funcs_get_knots(
        poly.ptr, out
    )
    return out
end

function knots(poly::PiecewiseLegendrePolyVector)
    nknots_ref = Ref{Int32}(-1)
    spir_funcs_get_n_knots(poly.ptr, nknots_ref)
    nknots = nknots_ref[]

    out = Vector{Float64}(undef, nknots)

    SparseIR.C_API.spir_funcs_get_knots(
        poly.ptr, out
    )
    return out
end

"""
    cover_domain(knots::Vector{Float64}, xmin::Float64, xmax::Float64, period::Float64, poly_xmin::Float64, poly_xmax::Float64)

Generate knots that cover the integration domain, handling periodic functions.

This function extends the basic knots to cover the entire integration domain,
taking into account periodicity if applicable.
"""
function cover_domain(knots::Vector{Float64}, xmin::Float64, xmax::Float64,
        period::Float64, poly_xmin::Float64, poly_xmax::Float64)
    if xmin > xmax
        error("xmin must be less than xmax")
    end

    # Add integration boundaries
    knots_vec = unique(vcat(knots, [xmin, xmax]))

    # Handle periodic functions
    if period != 0.0
        extended_knots = collect(knots_vec)

        # Extend in positive direction
        i = 1
        while true
            offset = i * period
            new_knots = knots_vec .+ offset
            if any(new_knots .> poly_xmax)
                break
            end
            append!(extended_knots, new_knots)
            i += 1
        end

        # Extend in negative direction
        i = 1
        while true
            offset = -i * period
            new_knots = knots_vec .+ offset
            if any(new_knots .< poly_xmin)
                break
            end
            append!(extended_knots, new_knots)
            i += 1
        end

        knots_vec = unique(extended_knots)
    end

    # Trim knots to the integration interval
    knots_vec = knots_vec[(knots_vec .>= xmin) .& (knots_vec .<= xmax)]
    knots_vec = sort(knots_vec)

    return knots_vec
end

"""
    overlap(poly::PiecewiseLegendrePoly, f;
        rtol=eps(), return_error=false, maxevals=10^4, points=Float64[])

Evaluate overlap integral of `poly` with arbitrary function `f` using default range.

Given the function `f`, evaluate the integral

    ∫ dx f(x) poly(x)

using adaptive Gauss-Legendre quadrature with the default integration range.

`points` is a sequence of break points in the integration interval where local
difficulties of the integrand may occur (e.g. singularities, discontinuities).
"""
function overlap(
        poly::PiecewiseLegendrePoly, f::F;
        rtol=eps(), return_error=false, maxevals=10^4, points=Float64[]
) where {F}
    xmin, xmax = poly.default_overlap_range
    return overlap(poly, f, xmin, xmax; rtol, return_error, maxevals, points)
end

"""
    overlap(poly::PiecewiseLegendrePoly, f, xmin::Float64, xmax::Float64;
        rtol=eps(), return_error=false, maxevals=10^4, points=Float64[])

Evaluate overlap integral of `poly` with arbitrary function `f`.

Given the function `f`, evaluate the integral

    ∫ dx f(x) poly(x)

using adaptive Gauss-Legendre quadrature.

`points` is a sequence of break points in the integration interval where local
difficulties of the integrand may occur (e.g. singularities, discontinuities).
"""
function overlap(
        poly::PiecewiseLegendrePoly, f::F, xmin::Float64, xmax::Float64;
        rtol=eps(), return_error=false, maxevals=10^4, points=Float64[]
) where {F}
    if xmin > xmax
        error("xmin must be less than xmax")
    end

    # Check bounds for all functions (both periodic and non-periodic)
    if xmin < poly.xmin
        error("xmin ($xmin) must be greater than or equal to the lower bound of the polynomial domain ($(poly.xmin))")
    end
    if xmax > poly.xmax
        error("xmax ($xmax) must be less than or equal to the upper bound of the polynomial domain ($(poly.xmax))")
    end

    knots_ = sort([xmin, xmax, points..., knots(poly)...])
    knots_ = cover_domain(knots_, xmin, xmax, poly.period, poly.xmin, poly.xmax)

    int_result, int_error = quadgk(x -> poly(x) * f(x), knots_...;
        rtol, order=10, maxevals)
    if return_error
        return int_result, int_error
    else
        return int_result
    end
end

"""
    overlap(polys::PiecewiseLegendrePolyVector, f;
        rtol=eps(), return_error=false, maxevals=10^4, points=Float64[])

Evaluate overlap integral of `polys` with arbitrary function `f` using default range.

Given the function `f`, evaluate the integral

    ∫ dx f(x) polys[i](x)

for each polynomial in the vector using adaptive Gauss-Legendre quadrature with the default integration range.
"""
function overlap(
        polys::PiecewiseLegendrePolyVector, f::F;
        rtol=eps(), return_error=false, maxevals=10^4, points=Float64[]
) where {F}
    xmin, xmax = polys.default_overlap_range
    return overlap(polys, f, xmin, xmax; rtol, return_error, maxevals, points)
end

function overlap(
        polys::PiecewiseLegendrePolyVector, f::F, xmin::Float64, xmax::Float64;
        rtol=eps(), return_error=false, maxevals=10^4, points=Float64[]
) where {F}
    result_ = [overlap(polys[i], f, xmin, xmax; rtol, return_error, maxevals, points)
               for i in 1:size(polys)]
    result_shape = (size(polys), size(first(result_))...)
    return reshape(vcat(result_...), result_shape)
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

#function overlap(polys::PiecewiseLegendrePolyVector, f::F; rtol=eps(),
#return_error=false) where {F}
#return overlap.(polys, f; rtol, return_error)
#end
