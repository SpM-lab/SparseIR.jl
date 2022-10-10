"""
    PiecewiseLegendrePoly <: Function

Piecewise Legendre polynomial.

Models a function on the interval ``[xmin, xmax]`` as a set of segments on the
intervals ``S[i] = [a[i], a[i+1]]``, where on each interval the function
is expanded in scaled Legendre polynomials.
"""
struct PiecewiseLegendrePoly{T} <: Function
    polyorder :: Int
    xmin      :: T
    xmax      :: T

    knots :: Vector{T}
    Δx    :: Vector{T}
    data  :: Matrix{T}
    symm  :: Int
    l     :: Int

    xm     :: Vector{T}
    inv_xs :: Vector{T}
    norm   :: Vector{T}

    function PiecewiseLegendrePoly(polyorder::Integer, xmin::AbstractFloat,
                                   xmax::AbstractFloat, knots::AbstractVector,
                                   Δx::AbstractVector, data::AbstractMatrix, symm::Integer,
                                   l::Integer, xm::AbstractVector, inv_xs::AbstractVector,
                                   norm::AbstractVector)
        !any(isnan, data) || error("data contains NaN")
        issorted(knots) || error("knots must be monotonically increasing")
        Δx ≈ diff(knots) || error("Δx must work with knots")
        return new{eltype(knots)}(polyorder, xmin, xmax, knots, Δx, data, symm, l, xm,
                                  inv_xs, norm)
    end
end

function PiecewiseLegendrePoly(data, p::PiecewiseLegendrePoly; symm=0)
    return PiecewiseLegendrePoly(p.polyorder, p.xmin, p.xmax, p.knots, p.Δx, data, symm,
                                 p.l, p.xm, p.inv_xs, p.norm)
end

function PiecewiseLegendrePoly(data::Matrix, knots::Vector, l::Integer;
                               Δx=diff(knots), symm=0)
    polyorder, nsegments = size(data)
    size(knots) == (nsegments + 1,) || error("Invalid knots array")
    xm     = @views @. (knots[begin:(end - 1)] + knots[(begin + 1):end]) / 2
    inv_xs = 2 ./ Δx
    norm   = sqrt.(inv_xs)
    return PiecewiseLegendrePoly(polyorder, first(knots), last(knots), knots,
                                 Δx, data, symm, l, xm, inv_xs, norm)
end

Base.size(::PiecewiseLegendrePoly) = ()

function Base.show(io::IO, p::PiecewiseLegendrePoly)
    print(io, "PiecewiseLegendrePoly on [$(p.xmin), $(p.xmax)], order=$(p.polyorder)")
end

@inline function (poly::PiecewiseLegendrePoly)(x::Number)
    i, x̃ = split(poly, x)
    return legval(x̃, view(poly.data, :, i)) * poly.norm[i]
end
@inline (poly::PiecewiseLegendrePoly)(xs::AbstractVector) = poly.(xs)

"""
    overlap(poly::PiecewiseLegendrePoly, f; 
        rtol=eps(T), return_error=false, maxevals=10^4, points=T[])

Evaluate overlap integral of `poly` with arbitrary function `f`.

Given the function `f`, evaluate the integral

    ∫ dx f(x) poly(x)

using adaptive Gauss-Legendre quadrature.

`points` is a sequence of break points in the integration interval where local
difficulties of the integrand may occur (e.g. singularities, discontinuities).
"""
function overlap(poly::PiecewiseLegendrePoly{T}, f;
                 rtol=eps(T), return_error=false, maxevals=10^4, points=T[]) where {T}
    int_result, int_error = quadgk(x -> poly(x) * f(x),
                                   unique!(sort!([poly.knots; points]))...;
                                   rtol, order=10, maxevals)
    if return_error
        return int_result, int_error
    else
        return int_result
    end
end

"""
    deriv(poly)

Get polynomial for the derivative.
"""
function deriv(poly::PiecewiseLegendrePoly, n=1)
    ddata = legder(poly.data, n)

    scale = poly.inv_xs .^ n
    ddata .*= transpose(scale)
    return PiecewiseLegendrePoly(ddata, poly; symm=(-1)^n * poly.symm)
end

"""
    roots(poly)

Find all roots of the piecewise polynomial `poly`.
"""
function roots(poly::PiecewiseLegendrePoly; tol=1e-10, alpha=Val(2))
    grid = poly.knots
    grid = refine_grid(grid, alpha)
    return find_all(poly, grid)
end

@inline function check_domain(poly::PiecewiseLegendrePoly, x)
    poly.xmin ≤ x ≤ poly.xmax || throw(DomainError(x, "x is outside the domain"))
    return true
end

"""
    split(poly, x)

Split segment.

Find segment of poly's domain that covers `x`.
"""
@inline function split(poly, x::Number)
    @boundscheck check_domain(poly, x)

    i = max(searchsortedlast(poly.knots, x; lt=≤), 1)
    x̃ = x - poly.xm[i]
    x̃ *= poly.inv_xs[i]
    return i, x̃
end

function Base.:*(poly::PiecewiseLegendrePoly, factor::Number)
    return PiecewiseLegendrePoly(poly.data * factor, poly.knots, poly.l;
                                 Δx=poly.Δx, symm=poly.symm)
end
Base.:*(factor::Number, poly::PiecewiseLegendrePoly) = poly * factor
function Base.:+(p1::PiecewiseLegendrePoly, p2::PiecewiseLegendrePoly)
    p1.knots == p2.knots || error("knots must be the same")
    return PiecewiseLegendrePoly(p1.data + p2.data, p1.knots, -1;
                                 Δx=p1.Δx, symm=p1.symm == p2.symm ? p1.symm : 0)
end
function Base.:-(poly::PiecewiseLegendrePoly)
    return PiecewiseLegendrePoly(-poly.data, poly.knots, -1;
                                 Δx=poly.Δx, symm=poly.symm)
end
Base.:-(p1::PiecewiseLegendrePoly, p2::PiecewiseLegendrePoly) = p1 + (-p2)

#################################
## PiecewiseLegendrePolyVector ##
#################################

"""
    PiecewiseLegendrePolyVector{T}

Alias for `Vector{PiecewiseLegendrePoly{T}}`.
"""
const PiecewiseLegendrePolyVector{T} = Vector{PiecewiseLegendrePoly{T}}

function Base.show(io::IO, polys::PiecewiseLegendrePolyVector)
    print(io, "$(length(polys))-element PiecewiseLegendrePolyVector ")
    print(io, "on [$(polys.xmin), $(polys.xmax)]")
end

function PiecewiseLegendrePolyVector(data::AbstractArray{T,3}, knots::Vector{T};
                                     symm=zeros(Int, size(data, 3))) where {T<:Real}
    return [PiecewiseLegendrePoly(data[:, :, i], knots, i - 1; symm=symm[i])
            for i in axes(data, 3)]
end

function PiecewiseLegendrePolyVector(polys::PiecewiseLegendrePolyVector,
                                     knots::AbstractVector; Δx=diff(knots), symm=0)
    length(polys) == length(symm) ||
        throw(DimensionMismatch("Sizes of polys and symm don't match"))

    return map(zip(polys, symm)) do (poly, sym)
        PiecewiseLegendrePoly(poly.data, knots, poly.l; Δx, symm=sym)
    end
end

function PiecewiseLegendrePolyVector(data::AbstractArray{T,3},
                                     polys::PiecewiseLegendrePolyVector) where {T}
    size(data, 3) == length(polys) ||
        throw(DimensionMismatch("Sizes of data and polys don't match"))

    polys_new = deepcopy(polys)
    for i in eachindex(polys)
        polys_new[i].data .= data[:, :, i]
    end
    return polys_new
end

(polys::PiecewiseLegendrePolyVector)(x) = [poly(x) for poly in polys]
function (polys::PiecewiseLegendrePolyVector)(x::AbstractArray)
    return reshape(mapreduce(polys, vcat, x), (size(polys)..., size(x)...))
end

function Base.getproperty(polys::PiecewiseLegendrePolyVector, sym::Symbol)
    if sym ∈ (:xmin, :xmax, :knots, :Δx, :polyorder, :xm, :inv_xs, :norm)
        return getproperty(first(polys), sym)
    elseif sym === :symm
        return map(poly -> poly.symm, polys)
    elseif sym === :data
        init = Array{Float64, 3}(undef, size(first(polys).data)..., 0)
        return mapreduce(poly -> poly.data, (x...) -> cat(x...; dims=3), polys; init)
    else
        return getfield(polys, sym)
    end
end

# Backward compatibility
function overlap(polys::PiecewiseLegendrePolyVector{T}, f;
                 rtol=eps(T), return_error=false) where {T}
    return overlap.(polys, f; rtol, return_error)
end

#########################
## PiecewiseLegendreFT ##
#########################

"""
    PowerModel

Model from a high-frequency series expansion::

    A(iω) == sum(A[n] / (iω)^(n+1) for n in 1:N)

where ``iω == i * π/2 * wn`` is a reduced imaginary frequency, i.e.,
``wn`` is an odd/even number for fermionic/bosonic frequencies.
"""
struct PowerModel{T<:AbstractFloat}
    moments::Vector{T}
end

const DEFAULT_GRID = [range(0; length=2^6);
                      trunc.(Int, exp2.(range(6, 25; length=32 * (25 - 6) + 1)))]

"""
    PiecewiseLegendreFT <: Function

Fourier transform of a piecewise Legendre polynomial.

For a given frequency index `n`, the Fourier transform of the Legendre
function is defined as:

        p̂(n) == ∫ dx exp(im * π * n * x / (xmax - xmin)) p(x)

The polynomial is continued either periodically (`freq=:even`), in which
case `n` must be even, or antiperiodically (`freq=:odd`), in which case
`n` must be odd.
"""
struct PiecewiseLegendreFT{T,S<:Statistics} <: Function
    poly       :: PiecewiseLegendrePoly{T}
    statistics :: S
    n_asymp    :: T
    model      :: PowerModel{T}
end

function PiecewiseLegendreFT(poly::PiecewiseLegendrePoly{T}, stat::Statistics;
                             n_asymp=Inf) where {T}
    (poly.xmin, poly.xmax) == (-1, 1) || error("Only interval [-1, 1] is supported")
    model = power_model(stat, poly)
    return PiecewiseLegendreFT(poly, stat, T(n_asymp), model)
end

const PiecewiseLegendreFTVector{T,S} = Vector{PiecewiseLegendreFT{T,S}}

function PiecewiseLegendreFTVector(polys::PiecewiseLegendrePolyVector,
                                   stat::Statistics; n_asymp=Inf)
    return [PiecewiseLegendreFT(poly, stat; n_asymp) for poly in polys]
end

function Base.getproperty(polyFTs::PiecewiseLegendreFTVector, sym::Symbol)
    if sym ∈ (:stat, :n_asymp)
        return getproperty(first(polyFTs), sym)
    elseif sym === :poly
        return map(poly -> poly.poly, polyFTs)
    else
        return getfield(polyFTs, sym)
    end
end

statistics(polyFT::PiecewiseLegendreFT)        = polyFT.statistics
statistics(polyFTs::PiecewiseLegendreFTVector) = statistics(first(polyFTs))
zeta(polyFT::PiecewiseLegendreFT)              = zeta(statistics(polyFT))
zeta(polyFTs::PiecewiseLegendreFTVector)       = zeta(first(polyFTs))

"""
    (polyFT::PiecewiseLegendreFT)(ω)

Obtain Fourier transform of polynomial for given `MatsubaraFreq` `ω`.
"""
function (polyFT::Union{PiecewiseLegendreFT{T,S},
                        PiecewiseLegendreFTVector{T,S}})(ω::MatsubaraFreq{S}) where {T,S}
    n = Int(ω)
    if abs(n) < polyFT.n_asymp
        return compute_unl_inner(polyFT.poly, n)
    else
        return giw(polyFT, n)
    end
end

(polyFT::PiecewiseLegendreFT)(n::Integer)       = polyFT(MatsubaraFreq(n))
(polyFT::PiecewiseLegendreFTVector)(n::Integer) = polyFT(MatsubaraFreq(n))
(polyFT::PiecewiseLegendreFT)(n::AbstractArray) = polyFT.(n)
function (polyFTs::PiecewiseLegendreFTVector)(n::AbstractArray)
    return reshape(mapreduce(polyFTs, vcat, n), (size(polyFTs)..., size(n)...))
end

"""
    giw(polyFT, wn)

Return model Green's function for reduced frequencies
"""
function giw(polyFT, wn::Integer)
    iw = im * π / 2 * wn
    iszero(wn) && return zero(inv_iw)
    inv_iw = 1 / iw
    return inv_iw * evalpoly(inv_iw, moments(polyFT))
end

moments(polyFT::PiecewiseLegendreFT) = polyFT.model.moments
function moments(polyFTs::PiecewiseLegendreFTVector)
    n = length(first(polyFTs).model.moments)
    return [[p.model.moments[i] for p in polyFTs] for i in 1:n]
end

"""
    find_extrema(polyFT::PiecewiseLegendreFT; part=nothing, grid=DEFAULT_GRID)

Obtain extrema of Fourier-transformed polynomial.
"""
function find_extrema(û::PiecewiseLegendreFT; part=nothing, grid=DEFAULT_GRID)
    f  = func_for_part(û, part)
    x₀ = discrete_extrema(f, grid)
    x₀ .= 2x₀ .+ zeta(statistics(û))
    x₀ = symmetrize_matsubara(x₀)
    return MatsubaraFreq.(statistics(û), x₀)
end

function sign_changes(û::PiecewiseLegendreFT; part=nothing, grid=DEFAULT_GRID)
    f = func_for_part(û, part)
    x₀ = find_all(f, grid)
    x₀ .= 2x₀ .+ zeta(statistics(û))
    x₀ = symmetrize_matsubara(x₀)
    return MatsubaraFreq.(statistics(û), x₀)
end

@inline function func_for_part(polyFT::PiecewiseLegendreFT, part=nothing)
    if isnothing(part)
        parity = polyFT.poly.symm
        if parity == 1
            part = statistics(polyFT) isa Bosonic ? real : imag
        elseif parity == -1
            part = statistics(polyFT) isa Bosonic ? imag : real
        else
            error("Cannot detect parity")
        end
    end
    return function(n)
        stat = statistics(polyFT)
        part(polyFT(MatsubaraFreq{typeof(stat)}(2n + zeta(stat))))
    end
end

@inline function symmetrize_matsubara(x₀)
    issorted(x₀) || error("set of Matsubara points not ordered")
    first(x₀) ≥ 0 || error("points must be non-negative")
    if iszero(first(x₀))
        x₀ = [-reverse(x₀); x₀[2:end]]
    else
        x₀ = [-reverse(x₀); x₀]
    end
    return x₀
end

function derivs(ppoly, x)
    res = [ppoly(x)]
    for _ in 2:(ppoly.polyorder)
        ppoly = deriv(ppoly)
        push!(res, ppoly(x))
    end
    return res
end

function power_moments(stat, deriv_x1, l)
    statsign = zeta(stat) == 1 ? -1 : 1
    mmax     = length(deriv_x1)
    coeff_lm = @. ((-1)^(1:mmax) + statsign * (-1)^l) * deriv_x1
    return -statsign / √2 * coeff_lm
end

function power_model(stat, poly)
    deriv_x1 = derivs(poly, 1.0)
    moments  = power_moments(stat, deriv_x1, poly.l)
    return PowerModel(moments)
end

########################
### Helper Functions ###
########################

"""
    compute_unl_inner(poly, wn)

Compute piecewise Legendre to Matsubara transform.
"""
@inline function compute_unl_inner(poly::PiecewiseLegendrePoly, wn)
    t_pin = pqn(poly, wn)
    return dot(poly.data, transpose(t_pin))
end
@inline function compute_unl_inner(polys::PiecewiseLegendrePolyVector, wn)
    t_pin = pqn(polys, wn)
    return [dot(poly.data, transpose(t_pin)) for poly in polys]
end

@inline function pqn(poly, wn)
    p        = transpose(range(0; length=poly.polyorder))
    wred     = π / 2 * wn
    phase_wi = phase_stable(poly, wn)
    return @. get_tnl(p, wred * poly.Δx / 2) * phase_wi / (√2 * poly.norm)
end

"""
    get_tnl(l, w)

Fourier integral of the `l`-th Legendre polynomial::

    Tₗ(ω) == ∫ dx exp(iωx) Pₗ(x)
"""
@inline function get_tnl(l, w)
    result = 2im^l * sphericalbesselj(l, abs(w))
    return w < 0 ? conj(result) : result
end

# Works like numpy.choices
@inline choose(a, choices) = [choices[a[i]][i] for i in eachindex(a)]

"""
    shift_xmid(knots, Δx)

Return midpoint relative to the nearest integer plus a shift.

Return the midpoints `xmid` of the segments, as pair `(diff, shift)`,
where shift is in `(0, 1, -1)` and `diff` is a float such that
`xmid == shift + diff` to floating point accuracy.
"""
@inline function shift_xmid(knots, Δx)
    Δx_half = Δx ./ 2
    xmid_m1 = cumsum(Δx) - Δx_half
    xmid_p1 = -reverse(cumsum(reverse(Δx))) + Δx_half
    xmid_0  = knots[2:end] - Δx_half

    shift = round.(Int, xmid_0)
    diff  = choose(shift .+ 2, (xmid_m1, xmid_0, xmid_p1))
    return diff, shift
end

"""
    phase_stable(poly, wn)

Phase factor for the piecewise Legendre to Matsubara transform.

Compute the following phase factor in a stable way:

    exp.(iπ/2 * wn * cumsum(poly.Δx))
"""
@inline function phase_stable(poly, wn)
    xmid_diff, extra_shift = shift_xmid(poly.knots, poly.Δx)

    if wn isa Integer
        shift_arg = wn * xmid_diff
    else
        delta_wn, wn = modf(wn)
        wn           = trunc(Int, wn)
        shift_arg    = wn * xmid_diff
        @. shift_arg += delta_wn * (extra_shift + xmid_diff)
    end

    phase_shifted = @. cispi(shift_arg / 2)
    corr          = @. im^mod(wn * (extra_shift + 1), 4)
    return corr .* phase_shifted
end
