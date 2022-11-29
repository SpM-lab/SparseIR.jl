"""
    PiecewiseLegendrePoly <: Function

Piecewise Legendre polynomial.

Models a function on the interval ``[xmin, xmax]`` as a set of segments on the
intervals ``S[i] = [a[i], a[i+1]]``, where on each interval the function
is expanded in scaled Legendre polynomials.
"""
struct PiecewiseLegendrePoly <: Function
    polyorder :: Int
    xmin      :: Float64
    xmax      :: Float64

    knots :: Vector{Float64}
    Δx    :: Vector{Float64}
    data  :: Matrix{Float64}
    symm  :: Int
    l     :: Int

    xm     :: Vector{Float64}
    inv_xs :: Vector{Float64}
    norm   :: Vector{Float64}

    function PiecewiseLegendrePoly(polyorder::Integer, xmin::Real,
                                   xmax::Real, knots::AbstractVector,
                                   Δx::AbstractVector, data::AbstractMatrix, symm::Integer,
                                   l::Integer, xm::AbstractVector, inv_xs::AbstractVector,
                                   norm::AbstractVector)
        !any(isnan, data) || error("data contains NaN")
        issorted(knots) || error("knots must be monotonically increasing")
        # Δx ≈ diff(knots) || error("Δx must work with knots")
        @inbounds for i in eachindex(Δx)
            Δx[i] ≈ knots[i+1] - knots[i] || error("Δx must work with knots")
        end
        return new(polyorder, xmin, xmax, knots, Δx, data, symm, l, xm, inv_xs, norm)
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

function (poly::PiecewiseLegendrePoly)(x::Real)
    i, x̃ = split(poly, x)
    return legval(x̃, view(poly.data, :, i)) * poly.norm[i]
end
(poly::PiecewiseLegendrePoly)(xs::AbstractVector) = poly.(xs)

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
function overlap(poly::PiecewiseLegendrePoly, f::F;
                 rtol=eps(), return_error=false, maxevals=10^4, points=Float64[]) where {F}
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
    deriv(poly[, ::Val{n}=Val(1)])

Get polynomial for the `n`th derivative.
"""
function deriv(poly::PiecewiseLegendrePoly, ::Val{n}=Val(1)) where {n}
    ddata = legder(poly.data, n)

    # scale = reshape(poly.inv_xs, (1, :)) .^ n
    # ddata .*= scale
    # @show size(ddata) size(poly.inv_xs)
    @views @inbounds for i in axes(ddata, 2)
        ddata[:, i] .*= poly.inv_xs[i] ^ n
    end
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

Base.checkbounds(::Type{Bool}, poly::PiecewiseLegendrePoly, x::Real) =
    poly.xmin ≤ x ≤ poly.xmax

Base.checkbounds(poly::PiecewiseLegendrePoly, x::Real) =
    checkbounds(Bool, poly, x) || throw(BoundsError(poly, x))

"""
    split(poly, x)

Split segment.

Find segment of poly's domain that covers `x`.
"""
function split(poly, x::Real)
    @boundscheck checkbounds(poly, x)

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
    PiecewiseLegendrePolyVector

Alias for `Vector{PiecewiseLegendrePoly}`.
"""
const PiecewiseLegendrePolyVector = Vector{PiecewiseLegendrePoly}

function Base.show(io::IO, polys::PiecewiseLegendrePolyVector)
    print(io, "$(length(polys))-element PiecewiseLegendrePolyVector ")
    print(io, "on [$(polys.xmin), $(polys.xmax)]")
end

function Vector{PiecewiseLegendrePoly}(data::AbstractArray{T,3}, knots::Vector{T};
                                       symm=zeros(Int, size(data, 3))) where {T<:Real}
    return [PiecewiseLegendrePoly(data[:, :, i], knots, i - 1; symm=symm[i])
            for i in axes(data, 3)]
end

function Vector{PiecewiseLegendrePoly}(polys::PiecewiseLegendrePolyVector,
                                       knots::AbstractVector; Δx=diff(knots), symm=0)
    length(polys) == length(symm) ||
        throw(DimensionMismatch("Sizes of polys and symm don't match"))

    return map(zip(polys, symm)) do (poly, sym)
        PiecewiseLegendrePoly(poly.data, knots, poly.l; Δx, symm=sym)
    end
end

function Vector{PiecewiseLegendrePoly}(data::AbstractArray{T,3},
                                       polys::PiecewiseLegendrePolyVector) where {T}
    size(data, 3) == length(polys) ||
        throw(DimensionMismatch("Sizes of data and polys don't match"))

    polys_new = deepcopy(polys)
    @inbounds for i in eachindex(polys)
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
        data = Array{Float64, 3}(undef, size(first(polys).data)..., length(polys))
        for i in eachindex(polys)
            data[:, :, i] .= polys[i].data
        end
        return data
    else
        return getfield(polys, sym)
    end
end

# Backward compatibility
function overlap(polys::PiecewiseLegendrePolyVector, f::F;
                 rtol=eps(), return_error=false) where {F}
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
struct PiecewiseLegendreFT{S<:Statistics} <: Function
    poly       :: PiecewiseLegendrePoly
    statistics :: S
    n_asymp    :: Float64
    model      :: PowerModel{Float64}
end

function PiecewiseLegendreFT(poly::PiecewiseLegendrePoly, stat::Statistics; n_asymp=Inf)
    (poly.xmin, poly.xmax) == (-1, 1) || error("Only interval [-1, 1] is supported")
    model = power_model(stat, poly)
    PiecewiseLegendreFT(poly, stat, Float64(n_asymp), model)
end

const PiecewiseLegendreFTVector{S} = Vector{PiecewiseLegendreFT{S}}

PiecewiseLegendreFTVector(polys::PiecewiseLegendrePolyVector,
                                   stat::Statistics; n_asymp=Inf) =
    [PiecewiseLegendreFT(poly, stat; n_asymp) for poly in polys]

function Base.getproperty(polyFTs::PiecewiseLegendreFTVector, sym::Symbol)
    if sym === :stat || sym === :n_asymp
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
function (polyFT::Union{PiecewiseLegendreFT{S},
                        PiecewiseLegendreFTVector{S}})(ω::MatsubaraFreq{S}) where {S}
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
(polyFTs::PiecewiseLegendreFTVector)(n::AbstractArray) =
    reshape(mapreduce(polyFTs, vcat, n), (size(polyFTs)..., size(n)...))

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
function find_extrema(û::PiecewiseLegendreFT; part=nothing, grid=DEFAULT_GRID, positive_only=false)
    f  = func_for_part(û, part)
    x₀ = discrete_extrema(f, grid)
    x₀ .= 2x₀ .+ zeta(statistics(û))
    positive_only || symmetrize_matsubara!(x₀)
    return MatsubaraFreq.(statistics(û), x₀)
end

function sign_changes(û::PiecewiseLegendreFT; part=nothing, grid=DEFAULT_GRID, positive_only=false)
    f = func_for_part(û, part)
    x₀ = find_all(f, grid)
    x₀ .= 2x₀ .+ zeta(statistics(û))
    positive_only || symmetrize_matsubara!(x₀)
    return MatsubaraFreq.(statistics(û), x₀)
end

function func_for_part(polyFT::PiecewiseLegendreFT{S}, part=nothing) where {S}
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
    let polyFT=polyFT
        n -> part(polyFT(MatsubaraFreq{S}(2n + zeta(polyFT))))::Float64
    end
end

function symmetrize_matsubara!(x₀)
    issorted(x₀) || error("set of Matsubara points not ordered")
    first(x₀) ≥ 0 || error("points must be non-negative")
    revx₀ = reverse(x₀)
    iszero(first(x₀)) && popfirst!(x₀)
    prepend!(x₀, -revx₀)
end

function derivs(ppoly, x)
    res = [ppoly(x)]
    for _ in 2:(ppoly.polyorder)
        ppoly = deriv(ppoly)
        push!(res, ppoly(x))
    end
    return res
end

function power_moments!(stat, deriv_x1, l)
    statsign = zeta(stat) == 1 ? -1 : 1
    @inbounds for m in 1:length(deriv_x1)
        deriv_x1[m] *= -(statsign * (-1)^m + (-1)^l) / sqrt(2)
    end
    deriv_x1
end

function power_model(stat, poly)
    deriv_x1 = derivs(poly, 1.0)
    moments = power_moments!(stat, deriv_x1, poly.l)
    return PowerModel(moments)
end

########################
### Helper Functions ###
########################

"""
    compute_unl_inner(poly, wn)

Compute piecewise Legendre to Matsubara transform.
"""
function compute_unl_inner(poly::PiecewiseLegendrePoly, wn)
    t_pin = pqn(poly, wn)
    return dot(poly.data, t_pin)
end
function compute_unl_inner(polys::PiecewiseLegendrePolyVector, wn)
    t_pin = pqn(polys, wn)
    return [dot(poly.data, t_pin) for poly in polys]
end

function pqn(poly, wn)
    p = reshape(range(0; length=poly.polyorder), (1, :))
    wred = π / 2 * wn
    phase_wi = phase_stable(poly, wn)
    transpose(@. get_tnl(p, wred * poly.Δx / 2) * phase_wi / (sqrt(2) * poly.norm))
end

"""
    get_tnl(l, w)

Fourier integral of the `l`-th Legendre polynomial::

    Tₗ(ω) == ∫ dx exp(iωx) Pₗ(x)
"""
function get_tnl(l, w)
    result = 2im^l * sphericalbesselj(l, abs(w))
    return w < 0 ? conj(result) : result
end

# Works like numpy.choose
choose(a, choices) = [choices[a[i]][i] for i in eachindex(a)]

"""
    shift_xmid(knots, Δx)

Return midpoint relative to the nearest integer plus a shift.

Return the midpoints `xmid` of the segments, as pair `(diff, shift)`,
where shift is in `(0, 1, -1)` and `diff` is a float such that
`xmid == shift + diff` to floating point accuracy.
"""
function shift_xmid(knots, Δx)
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
function phase_stable(poly, wn)
    xmid_diff, extra_shift = shift_xmid(poly.knots, poly.Δx)

    if wn isa Integer
        shift_arg = wn * xmid_diff
    else
        delta_wn, wn = modf(wn)
        wn = trunc(Int, wn)
        shift_arg = wn * xmid_diff
        @. shift_arg += delta_wn * (extra_shift + xmid_diff)
    end

    phase_shifted = @. cispi(shift_arg / 2)
    corr = @. im^mod(wn * (extra_shift + 1), 4)
    return corr .* phase_shifted
end
