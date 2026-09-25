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
    default_overlap_range::Tuple{Float64,Float64} # Default range for overlap calculations
    function PiecewiseLegendrePoly(
            funcs::Ptr{spir_funcs}, xmin::Float64, xmax::Float64, period::Float64,
            default_overlap_range::Union{Tuple{Float64,Float64},Nothing}=nothing)
        default_range = default_overlap_range === nothing ? (xmin, xmax) :
                        default_overlap_range
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
    default_overlap_range::Tuple{Float64,Float64} # Default range for overlap calculations
    function PiecewiseLegendrePolyVector(
            funcs::Ptr{spir_funcs}, xmin::Float64, xmax::Float64, period::Float64,
            default_overlap_range::Union{Tuple{Float64,Float64},Nothing}=nothing)
        default_range = default_overlap_range === nothing ? (xmin, xmax) :
                        default_overlap_range
        result = new(funcs, xmin, xmax, period, default_range)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

function Base.size(ptr::Ptr{spir_funcs})
    sz = Ref{Int32}(-1)
    _check_status(spir_funcs_get_size(ptr, sz), "spir_funcs_get_size")
    return Int(sz[])
end

Base.size(polys::PiecewiseLegendrePolyVector) = size(polys.ptr)

"""
    PiecewiseLegendreFTVector

Fourier transforms of a set of piecewise Legendre polynomials, evaluated at
Matsubara frequencies.

For a reduced frequency `n`, the transform of the basis function `u_l` is

    û_l(n) == ∫₀^β dτ exp(iπnτ/β) u_l(τ).

The object knows the statistics of its basis: it accepts `MatsubaraFreq`s of
that statistics or integers of the matching parity (odd for fermions, even for
bosons). A frequency of the other statistics throws `ArgumentError`, an integer
of the wrong parity `DomainError`. `polys[i]` returns a single
[`PiecewiseLegendreFT`](@ref), `polys[range]` another vector.
"""
mutable struct PiecewiseLegendreFTVector
    ptr::Ptr{spir_funcs}
    zeta::Int

    function PiecewiseLegendreFTVector(funcs::Ptr{spir_funcs}, zeta::Integer)
        result = new(funcs, zeta)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

"""
    PiecewiseLegendreFT

A single function of a [`PiecewiseLegendreFTVector`](@ref); calling it returns
a `ComplexF64`.
"""
mutable struct PiecewiseLegendreFT
    ptr::Ptr{spir_funcs}
    zeta::Int

    function PiecewiseLegendreFT(funcs::Ptr{spir_funcs}, zeta::Integer)
        result = new(funcs, zeta)
        finalizer(r -> spir_funcs_release(r.ptr), result)
        return result
    end
end

zeta(polys::Union{PiecewiseLegendreFT,PiecewiseLegendreFTVector}) = polys.zeta

# The C library panics on a point outside the domain (SpM-lab/sparse-ir-rs#266),
# so every point is checked here first.
function _check_domain(x::Real, xmin::Real, xmax::Real)
    isfinite(x) && xmin ≤ x ≤ xmax && return nothing
    throw(DomainError(x, "evaluation point must be finite and lie in [$xmin, $xmax]"))
end

# Reduced Matsubara index for the C library, checked against the statistics.
function _matsubara_index(zeta::Int, freq::MatsubaraFreq)
    SparseIR.zeta(freq) == zeta || throw(ArgumentError(
        "the frequency $(Int(freq))π/β is $(nameof(typeof(statistics(freq)))), \
         but the functions are $(nameof(typeof(Statistics(zeta))))"))
    return Int(freq)
end
function _matsubara_index(zeta::Int, n::Integer)
    S = typeof(Statistics(zeta))
    return Int(MatsubaraFreq{S}(n))       # DomainError for the wrong parity
end

function (polys::PiecewiseLegendrePoly)(x::Real)
    _check_domain(x, polys.xmin, polys.xmax)
    ret = Vector{Float64}(undef, length(polys.ptr))
    _check_status(spir_funcs_eval(polys.ptr, x, ret), "spir_funcs_eval")
    return only(ret)
end

function (polys::PiecewiseLegendrePolyVector)(x::Real)
    _check_domain(x, polys.xmin, polys.xmax)
    ret = Vector{Float64}(undef, length(polys))
    _check_status(spir_funcs_eval(polys.ptr, x, ret), "spir_funcs_eval")
    return ret
end

"""
    (polys::PiecewiseLegendrePolyVector)(x::AbstractVector)

`length(polys) × length(x)` matrix of the functions at the points `x`.
"""
function (polys::PiecewiseLegendrePolyVector)(x::AbstractVector{<:Real})
    for xi in x
        _check_domain(xi, polys.xmin, polys.xmax)
    end
    result = Matrix{Float64}(undef, length(polys), length(x))
    for (j, xj) in enumerate(x)
        result[:, j] = polys(xj)
    end
    return result
end

"""
    (poly::PiecewiseLegendrePoly)(x::AbstractVector)

Values of the single function `poly` at the points `x`.
"""
function (poly::PiecewiseLegendrePoly)(x::AbstractVector{<:Real})
    for xi in x
        _check_domain(xi, poly.xmin, poly.xmax)
    end
    return Float64[poly(xi) for xi in x]
end

function (polys::PiecewiseLegendreFTVector)(freq::Union{MatsubaraFreq,Integer})
    n = _matsubara_index(polys.zeta, freq)
    ret = Vector{ComplexF64}(undef, length(polys))
    _check_status(spir_funcs_eval_matsu(polys.ptr, n, ret), "spir_funcs_eval_matsu")
    return ret
end

function (poly::PiecewiseLegendreFT)(freq::Union{MatsubaraFreq,Integer})
    n = _matsubara_index(poly.zeta, freq)
    ret = Vector{ComplexF64}(undef, length(poly.ptr))
    _check_status(spir_funcs_eval_matsu(poly.ptr, n, ret), "spir_funcs_eval_matsu")
    return only(ret)
end

function (poly::PiecewiseLegendreFT)(x::AbstractVector)
    ns = Int64[_matsubara_index(poly.zeta, xi) for xi in x]
    return ComplexF64[poly(n) for n in ns]
end

Base.size(polys::PiecewiseLegendreFTVector) = size(polys.ptr)
Base.length(polys::PiecewiseLegendreFTVector) = length(polys.ptr)
Base.firstindex(::PiecewiseLegendreFTVector) = 1
Base.lastindex(polys::PiecewiseLegendreFTVector) = length(polys)

"""
    (polys::PiecewiseLegendreFTVector)(x::AbstractVector)

`length(polys) × length(x)` matrix of the functions at the frequencies `x`
(`MatsubaraFreq`s or integers, see [`PiecewiseLegendreFTVector`](@ref)).
"""
function (polys::PiecewiseLegendreFTVector)(x::AbstractVector)
    ns = Int64[_matsubara_index(polys.zeta, xi) for xi in x]
    n_basis = length(polys)
    result = Matrix{ComplexF64}(undef, n_basis, length(ns))
    col = Vector{ComplexF64}(undef, n_basis)
    for (i, n) in enumerate(ns)
        _check_status(spir_funcs_eval_matsu(polys.ptr, n, col), "spir_funcs_eval_matsu")
        result[:, i] = col
    end
    return result
end

function Base.getindex(polys::PiecewiseLegendreFTVector, i::Integer)
    1 ≤ i ≤ length(polys) || throw(BoundsError(polys, i))
    return PiecewiseLegendreFT(polys.ptr[Int(i)], polys.zeta)
end

function Base.getindex(polys::PiecewiseLegendreFTVector,
        I::Union{AbstractRange{<:Integer},AbstractVector{<:Integer}})
    indices = collect(1:length(polys))[I]
    return PiecewiseLegendreFTVector(polys.ptr[indices], polys.zeta)
end

function Base.getindex(funcs::Ptr{spir_funcs}, i::Int)
    status = Ref{Int32}(-100)
    indices = Vector{Int32}(undef, 1)
    indices[1] = i - 1 # Julia indices are 1-based, C indices are 0-based
    ret = spir_funcs_get_slice(funcs, 1, indices, status)
    _check_status(status[], "spir_funcs_get_slice")
    return _check_handle(ret, "spir_funcs_get_slice")
end

function Base.getindex(funcs::Ptr{spir_funcs}, indices::Vector{Int})
    n = length(funcs)
    for i in indices
        1 ≤ i ≤ n || throw(BoundsError(1:n, i))
    end
    status = Ref{Int32}(-100)
    indices_i32 = Vector{Int32}(undef, length(indices))
    indices_i32 .= indices .- 1 # Julia indices are 1-based, C indices are 0-based
    ret = spir_funcs_get_slice(funcs, length(indices), indices_i32, status)
    _check_status(status[], "spir_funcs_get_slice")
    return _check_handle(ret, "spir_funcs_get_slice")
end

function Base.getindex(polys::PiecewiseLegendrePolyVector, i::Int)
    1 ≤ i ≤ length(polys) || throw(BoundsError(polys, i))
    return PiecewiseLegendrePoly(
        polys.ptr[i], polys.xmin, polys.xmax, polys.period, polys.default_overlap_range)
end

function Base.getindex(polys::PiecewiseLegendrePolyVector,
        I)::Union{PiecewiseLegendrePoly,PiecewiseLegendrePolyVector}
    indices = collect(1:size(polys))[I]
    if indices isa Int
        return PiecewiseLegendrePoly(polys.ptr[indices], polys.xmin, polys.xmax,
            polys.period, polys.default_overlap_range)
    elseif length(indices) == 1
        return PiecewiseLegendrePoly(polys.ptr[indices[1]], polys.xmin, polys.xmax,
            polys.period, polys.default_overlap_range)
    else
        return PiecewiseLegendrePolyVector(polys.ptr[indices], polys.xmin, polys.xmax,
            polys.period, polys.default_overlap_range)
    end
end

function Base.length(funcs::Ptr{spir_funcs})
    sz = Ref{Int32}(-1)
    _check_status(spir_funcs_get_size(funcs, sz), "spir_funcs_get_size")
    return Int(sz[])
end

function Base.length(polys::PiecewiseLegendrePolyVector)
    return length(polys.ptr)
end

Base.firstindex(funcs::Ptr{spir_funcs}) = 1
Base.lastindex(funcs::Ptr{spir_funcs}) = length(funcs)

Base.firstindex(polys::PiecewiseLegendrePolyVector) = firstindex(polys.ptr)
Base.lastindex(polys::PiecewiseLegendrePolyVector) = lastindex(polys.ptr)

function knots(poly::Union{PiecewiseLegendrePoly,PiecewiseLegendrePolyVector})
    nknots_ref = Ref{Int32}(-1)
    _check_status(spir_funcs_get_n_knots(poly.ptr, nknots_ref), "spir_funcs_get_n_knots")
    out = Vector{Float64}(undef, nknots_ref[])
    _check_status(
        SparseIR.C_API.spir_funcs_get_knots(poly.ptr, out), "spir_funcs_get_knots")
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
    xmin ≤ xmax || throw(ArgumentError("xmin = $xmin must not exceed xmax = $xmax"))

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
    xmin ≤ xmax || throw(ArgumentError("xmin = $xmin must not exceed xmax = $xmax"))

    # Check bounds for all functions (both periodic and non-periodic)
    xmin ≥ poly.xmin || throw(DomainError(xmin,
        "xmin must not be below the lower end $(poly.xmin) of the domain"))
    xmax ≤ poly.xmax || throw(DomainError(xmax,
        "xmax must not exceed the upper end $(poly.xmax) of the domain"))

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
    if return_error
        # Each element is a `(value, error)` tuple; `size` is undefined for a
        # tuple, so the two halves have to be collected and shaped separately
        # instead of being reshaped together.
        results = [overlap(polys[i], f, xmin, xmax;
                       rtol, return_error=true, maxevals, points)
                   for i in 1:size(polys)]
        values = first.(results)
        errors = last.(results)
        # `quadgk` reports a single scalar error estimate per integral, so the
        # error array is shaped independently of the value array.
        value_shape = (size(polys), size(first(values))...)
        error_shape = (size(polys), size(first(errors))...)
        return reshape(vcat(values...), value_shape),
        reshape(vcat(errors...), error_shape)
    end
    result_ = [overlap(polys[i], f, xmin, xmax;
                   rtol, return_error=false, maxevals, points)
               for i in 1:size(polys)]
    result_shape = (size(polys), size(first(result_))...)
    return reshape(vcat(result_...), result_shape)
end

xmin(poly::PiecewiseLegendrePoly) = poly.xmin
xmax(poly::PiecewiseLegendrePoly) = poly.xmax
xmin(poly::PiecewiseLegendrePolyVector) = poly.xmin
xmax(poly::PiecewiseLegendrePolyVector) = poly.xmax

# NOTE: `PiecewiseLegendreFTVector` lives on the Matsubara axis and carries no
# `xmin`/`xmax`; there used to be `xmin`/`xmax` methods for it that unavoidably
# threw `FieldError`, and they have been removed rather than left as a trap.

"""
    deriv(poly::PiecewiseLegendrePoly, n=1)
    deriv(polys::PiecewiseLegendrePolyVector, n=1)

Return the `n`-th derivative of `poly`/`polys` as a new object of the same type,
computed by `libsparseir`. `n` may be given as an `Integer` or as a `Val`, and
must be non-negative; `n == 0` returns a copy.
"""
function deriv(poly::PiecewiseLegendrePoly, n::Integer=1)
    return PiecewiseLegendrePoly(_funcs_deriv(poly.ptr, n), poly.xmin, poly.xmax,
        poly.period, poly.default_overlap_range)
end

function deriv(polys::PiecewiseLegendrePolyVector, n::Integer=1)
    return PiecewiseLegendrePolyVector(_funcs_deriv(polys.ptr, n), polys.xmin,
        polys.xmax, polys.period, polys.default_overlap_range)
end

deriv(poly::PiecewiseLegendrePoly, ::Val{n}) where {n} = deriv(poly, n)
deriv(polys::PiecewiseLegendrePolyVector, ::Val{n}) where {n} = deriv(polys, n)

function _funcs_deriv(ptr::Ptr{spir_funcs}, n::Integer)
    n >= 0 || throw(DomainError(n, "derivative order must be non-negative"))
    status = Ref{Int32}(-100)
    out = GC.@preserve ptr C_API.spir_funcs_deriv(ptr, Cint(n), status)
    _check_status(status[], "spir_funcs_deriv")
    _check_handle(out, "spir_funcs_deriv")
    return out
end
