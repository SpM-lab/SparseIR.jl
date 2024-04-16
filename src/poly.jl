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
    norms  :: Vector{Float64}

    function PiecewiseLegendrePoly(polyorder::Integer, xmin::Real,
            xmax::Real, knots::AbstractVector,
            Δx::AbstractVector, data::AbstractMatrix, symm::Integer,
            l::Integer, xm::AbstractVector, inv_xs::AbstractVector,
            norms::AbstractVector)
        !any(isnan, data) || error("data contains NaN")
        issorted(knots) || error("knots must be monotonically increasing")
        @inbounds for i in eachindex(Δx)
            Δx[i] ≈ knots[i + 1] - knots[i] || error("Δx must work with knots")
        end
        return new(polyorder, xmin, xmax, knots, Δx, data, symm, l, xm, inv_xs, norms)
    end
end

function PiecewiseLegendrePoly(data, p::PiecewiseLegendrePoly; symm=symm(p))
    return PiecewiseLegendrePoly(
        polyorder(p), xmin(p), xmax(p), copy(knots(p)), copy(Δx(p)),
        data, symm, p.l, copy(p.xm), copy(p.inv_xs), copy(norms(p)))
end

function PiecewiseLegendrePoly(data::Matrix, knots::Vector, l::Integer;
        Δx=diff(knots), symm=0)
    polyorder, nsegments = size(data)
    length(knots) == nsegments + 1 || error("Invalid knots array")
    xm = @views @. (knots[begin:(end - 1)] + knots[(begin + 1):end]) / 2
    inv_xs = 2 ./ Δx
    norms = sqrt.(inv_xs)
    return PiecewiseLegendrePoly(polyorder, first(knots), last(knots), knots,
        Δx, data, symm, l, xm, inv_xs, norms)
end

Base.size(::PiecewiseLegendrePoly) = ()

xmin(p::PiecewiseLegendrePoly) = p.xmin
xmax(p::PiecewiseLegendrePoly) = p.xmax
knots(p::PiecewiseLegendrePoly) = p.knots
Δx(p::PiecewiseLegendrePoly) = p.Δx
symm(p::PiecewiseLegendrePoly) = p.symm
data(p::PiecewiseLegendrePoly) = p.data
norms(p::PiecewiseLegendrePoly) = p.norms
polyorder(p::PiecewiseLegendrePoly) = p.polyorder

function Base.show(io::IO, ::MIME"text/plain", p::PiecewiseLegendrePoly)
    print(io, "PiecewiseLegendrePoly on [$(xmin(p)), $(xmax(p))], order=$(polyorder(p))")
end

@inline function (poly::PiecewiseLegendrePoly)(x::Real)
    i, x̃ = split(poly, x)
    return @inbounds legval(x̃, @view data(poly)[:, i]) * norms(poly)[i]
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
        unique!(sort!([knots(poly); points]))...;
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

    @views @inbounds for i in axes(ddata, 2)
        ddata[:, i] .*= poly.inv_xs[i]^n
    end
    return PiecewiseLegendrePoly(ddata, poly; symm=(-1)^n * symm(poly))
end

"""
    roots(poly)

Find all roots of the piecewise polynomial `poly`.
"""
function roots(poly::PiecewiseLegendrePoly; tol=1e-10, alpha=Val(2))
    grid = knots(poly)
    grid = refine_grid(grid, alpha)
    return find_all(poly, grid)
end

function Base.checkbounds(::Type{Bool}, poly::PiecewiseLegendrePoly, x::Real)
    xmin(poly) ≤ x ≤ xmax(poly)
end

function Base.checkbounds(poly::PiecewiseLegendrePoly, x::Real)
    checkbounds(Bool, poly, x) ||
        throw(DomainError(x, "The domain is [$(xmin(poly)), $(xmax(poly))]"))
end

"""
    split(poly, x)

Split segment.

Find segment of poly's domain that covers `x`.
"""
@inline function split(poly, x::Real)
    @boundscheck checkbounds(poly, x)

    i = max(searchsortedlast(knots(poly), x; lt=≤), 1)
    x̃ = x - poly.xm[i]
    x̃ *= poly.inv_xs[i]
    return i, x̃
end

function Base.:*(poly::PiecewiseLegendrePoly, factor::Number)
    return PiecewiseLegendrePoly(data(poly) * factor, knots(poly), poly.l;
        Δx=Δx(poly), symm=symm(poly))
end
Base.:*(factor::Number, poly::PiecewiseLegendrePoly) = poly * factor
function Base.:+(p1::PiecewiseLegendrePoly, p2::PiecewiseLegendrePoly)
    knots(p1) == knots(p2) || error("knots must be the same")
    return PiecewiseLegendrePoly(data(p1) + data(p2), knots(p1), -1;
        Δx=Δx(p1), symm=symm(p1) == symm(p2) ? symm(p1) : 0)
end
function Base.:-(poly::PiecewiseLegendrePoly)
    return PiecewiseLegendrePoly(-data(poly), knots(poly), -1;
        Δx=Δx(poly), symm=symm(poly))
end
Base.:-(p1::PiecewiseLegendrePoly, p2::PiecewiseLegendrePoly) = p1 + (-p2)

#################################
## PiecewiseLegendrePolyVector ##
#################################

"""
    PiecewiseLegendrePolyVector

Contains a `Vector{PiecewiseLegendrePoly}`.
"""
struct PiecewiseLegendrePolyVector <: AbstractVector{PiecewiseLegendrePoly}
    polyvec::Vector{PiecewiseLegendrePoly}
end

Base.size(polys::PiecewiseLegendrePolyVector) = size(polys.polyvec)

Base.IndexStyle(::Type{PiecewiseLegendrePolyVector}) = IndexLinear()

Base.getindex(polys::PiecewiseLegendrePolyVector, i) = getindex(polys.polyvec, i)
function Base.getindex(polys::PiecewiseLegendrePolyVector, i::AbstractVector)
    PiecewiseLegendrePolyVector(getindex(polys.polyvec, i))
end

Base.setindex!(polys::PiecewiseLegendrePolyVector, p, i) = setindex!(polys.polyvec, p, i)

function Base.similar(polys::PiecewiseLegendrePolyVector)
    PiecewiseLegendrePolyVector(similar(polys.polyvec))
end

function Base.show(io::IO, ::MIME"text/plain", polys::PiecewiseLegendrePolyVector)
    print(io, "$(length(polys))-element PiecewiseLegendrePolyVector ")
    print(io, "on [$(xmin(polys)), $(xmax(polys))]")
end

function PiecewiseLegendrePolyVector(data::AbstractArray{T,3}, knots::Vector{T};
        symm=zeros(Int, size(data, 3))) where {T<:Real}
    PiecewiseLegendrePolyVector(map(axes(data, 3)) do i
        PiecewiseLegendrePoly(data[:, :, i], knots, i - 1; symm=symm[i])
    end)
end

function PiecewiseLegendrePolyVector(polys::PiecewiseLegendrePolyVector,
        knots::AbstractVector; Δx=diff(knots), symm=0)
    length(polys) == length(symm) ||
        throw(DimensionMismatch("Sizes of polys and symm don't match"))

    PiecewiseLegendrePolyVector(map(zip(polys, symm)) do (poly, sym)
        PiecewiseLegendrePoly(poly.data, knots, poly.l; Δx, symm=sym)
    end)
end

function PiecewiseLegendrePolyVector(data::AbstractArray{T,3},
        polys::PiecewiseLegendrePolyVector) where {T}
    size(data, 3) == length(polys) ||
        throw(DimensionMismatch("Sizes of data and polys don't match"))

    PiecewiseLegendrePolyVector(map(eachindex(polys)) do i
        PiecewiseLegendrePoly(data[:, :, i], polys[i])
    end)
end

for name in (:xmin, :xmax, :knots, :Δx, :polyorder, :norms)
    eval(:($name(polys::PiecewiseLegendrePolyVector) = $name(first(polys.polyvec))))
end

symm(polys::PiecewiseLegendrePolyVector) = map(symm, polys)

function data(polys::PiecewiseLegendrePolyVector)
    data = Array{Float64,3}(undef, size(first(polys).data)..., length(polys))
    @inbounds for i in eachindex(polys)
        data[:, :, i] .= polys[i].data
    end
    return data
end

(polys::PiecewiseLegendrePolyVector)(x) = [poly(x) for poly in polys]
function (polys::PiecewiseLegendrePolyVector)(x::AbstractArray)
    reshape(mapreduce(polys, vcat, x; init=Float64[]), (length(polys), size(x)...))
end

# Backward compatibility
function overlap(polys::PiecewiseLegendrePolyVector, f::F; rtol=eps(),
        return_error=false) where {F}
    overlap.(polys, f; rtol, return_error)
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
    poly    :: PiecewiseLegendrePoly
    n_asymp :: Float64
    model   :: PowerModel{Float64}
end

function PiecewiseLegendreFT(poly::PiecewiseLegendrePoly, stat::Statistics; n_asymp=Inf)
    (xmin(poly), xmax(poly)) == (-1, 1) || error("Only interval [-1, 1] is supported")
    model = power_model(stat, poly)
    PiecewiseLegendreFT{typeof(stat)}(poly, Float64(n_asymp), model)
end

n_asymp(polyFT::PiecewiseLegendreFT) = polyFT.n_asymp
statistics(::PiecewiseLegendreFT{S}) where {S} = S()
zeta(polyFT::PiecewiseLegendreFT) = zeta(statistics(polyFT))
poly(polyFT::PiecewiseLegendreFT) = polyFT.poly

struct PiecewiseLegendreFTVector{S} <: AbstractVector{PiecewiseLegendreFT{S}}
    polyvec::Vector{PiecewiseLegendreFT{S}}
end

Base.size(polys::PiecewiseLegendreFTVector) = size(polys.polyvec)

Base.IndexStyle(::Type{PiecewiseLegendreFTVector}) = IndexLinear()

Base.getindex(polys::PiecewiseLegendreFTVector, i) = getindex(polys.polyvec, i)
function Base.getindex(polys::PiecewiseLegendreFTVector, i::AbstractVector)
    PiecewiseLegendreFTVector(getindex(polys.polyvec, i))
end

Base.setindex!(polys::PiecewiseLegendreFTVector, p, i) = setindex!(polys.polyvec, p, i)

function Base.similar(polys::PiecewiseLegendreFTVector)
    PiecewiseLegendreFTVector(similar(polys.polyvec))
end

function PiecewiseLegendreFTVector(polys::PiecewiseLegendrePolyVector,
        stat::Statistics; n_asymp=Inf)
    PiecewiseLegendreFTVector(map(polys) do poly
        PiecewiseLegendreFT(poly, stat; n_asymp)
    end)
end

n_asymp(polyFTs::PiecewiseLegendreFTVector)    = n_asymp(first(polyFTs))
statistics(polyFTs::PiecewiseLegendreFTVector) = statistics(first(polyFTs))
zeta(polyFTs::PiecewiseLegendreFTVector)       = zeta(first(polyFTs))
poly(polyFTs::PiecewiseLegendreFTVector)       = PiecewiseLegendrePolyVector(map(poly, polyFTs))

"""
    (polyFT::PiecewiseLegendreFT)(ω)

Obtain Fourier transform of polynomial for given `MatsubaraFreq` `ω`.
"""
function (polyFT::Union{PiecewiseLegendreFT{S},
        PiecewiseLegendreFTVector{S}})(ω::MatsubaraFreq{S}) where {S}
    n = Int(ω)
    return if abs(n) < n_asymp(polyFT)
        compute_unl_inner(poly(polyFT), n)
    else
        giw(polyFT, n)
    end
end

(polyFT::PiecewiseLegendreFT)(n::Integer) = polyFT(MatsubaraFreq(n))
(polyFT::PiecewiseLegendreFTVector)(n::Integer) = polyFT(MatsubaraFreq(n))
(polyFT::PiecewiseLegendreFT)(n::AbstractArray) = polyFT.(n)
function (polyFTs::PiecewiseLegendreFTVector)(n::AbstractArray)
    reshape(mapreduce(polyFTs, vcat, n)::Vector{ComplexF64}, (length(polyFTs), size(n)...))
end

"""
    giw(polyFT, wn)

Return model Green's function for reduced frequencies
"""
function giw(polyFT, wn::Integer)
    iw = im * π / 2 * wn
    iszero(wn) && return zero(iw)
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
function find_extrema(û::PiecewiseLegendreFT; part=nothing, grid=DEFAULT_GRID,
        positive_only=false)
    f  = func_for_part(û, part)
    x₀ = discrete_extrema(f, grid)
    x₀ .= 2x₀ .+ zeta(statistics(û))
    positive_only || symmetrize_matsubara!(x₀)
    return MatsubaraFreq.(statistics(û), x₀)
end

function sign_changes(û::PiecewiseLegendreFT; part=nothing, grid=DEFAULT_GRID,
        positive_only=false)
    f = func_for_part(û, part)
    x₀ = find_all(f, grid)
    x₀ .= 2x₀ .+ zeta(statistics(û))
    positive_only || symmetrize_matsubara!(x₀)
    return MatsubaraFreq.(statistics(û), x₀)
end

function func_for_part(polyFT::PiecewiseLegendreFT{S}, part=nothing) where {S}
    if isnothing(part)
        parity = symm(poly(polyFT))
        if parity == 1
            part = statistics(polyFT) isa Bosonic ? real : imag
        elseif parity == -1
            part = statistics(polyFT) isa Bosonic ? imag : real
        else
            error("Cannot detect parity")
        end
    end
    let polyFT = polyFT
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
    for _ in 2:(polyorder(ppoly))
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
    wred = π / 4 * wn
    phase_wi = phase_stable(poly, wn)
    res = zero(ComplexF64)
    @inbounds for order in axes(data(poly), 1), j in axes(data(poly), 2)
        res += data(poly)[order, j] * get_tnl(order - 1, wred * Δx(poly)[j]) * phase_wi[j] /
               norms(poly)[j]
    end
    return res / sqrt(2)
end
function compute_unl_inner(polys::PiecewiseLegendrePolyVector, wn)
    p = reshape(range(0; length=polyorder(polys)), (1, :))
    wred = π / 4 * wn
    phase_wi = phase_stable(polys, wn)
    t_pin = permutedims(get_tnl.(p, wred .* Δx(polys)) .* phase_wi ./
                        (sqrt(2) .* norms(polys)))
    return [dot(poly.data, t_pin) for poly in polys]
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

"""
    shift_xmid(knots, Δx)

Return midpoint relative to the nearest integer plus a shift.

Return the midpoints `xmid` of the segments, as pair `(diff, shift)`,
where shift is in `(0, 1, -1)` and `diff` is a float such that
`xmid == shift + diff` to floating point accuracy.
"""
function shift_xmid(knots, Δx)
    Δx_half = Δx ./ 2

    xmid_m1 = cumsum(Δx)
    xmid_m1 .-= Δx_half

    xmid_p1 = reverse!(cumsum(reverse(Δx)))
    xmid_p1 .*= -1
    xmid_p1 .+= Δx_half

    xmid_0 = @inbounds knots[2:end]
    xmid_0 .-= Δx_half

    shift = round.(Int, xmid_0)
    diff = @inbounds [(xmid_m1, xmid_0, xmid_p1)[shift[i] + 2][i] for i in eachindex(shift)]
    return diff, shift
end

"""
    phase_stable(poly, wn)

Phase factor for the piecewise Legendre to Matsubara transform.

Compute the following phase factor in a stable way:

    exp.(iπ/2 * wn * cumsum(Δx(poly)))
"""
function phase_stable(poly, wn::Integer)
    xmid_diff, extra_shift = shift_xmid(knots(poly), Δx(poly))
    @. im^mod(wn * (extra_shift + 1), 4) * cispi(wn * xmid_diff / 2)
end

function phase_stable(poly, wn)
    xmid_diff, extra_shift = shift_xmid(knots(poly), Δx(poly))

    delta_wn, wn = modf(wn)
    wn = trunc(Int, wn)
    shift_arg = wn * xmid_diff
    @. shift_arg += delta_wn * (extra_shift + xmid_diff)

    return @. im^mod(wn * (extra_shift + 1), 4) * cispi(shift_arg / 2)
end
