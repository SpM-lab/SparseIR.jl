import AssociatedLegendrePolynomials: Plm
import IntervalRootFinding: roots as roots_irf, Interval, isunique, interval, mid, Newton,
                            Krawczyk
import QuadGK: quadgk
include("_bessels.jl")

export PiecewiseLegendrePoly, PiecewiseLegendrePolyArray, roots, hat, overlap, deriv

struct PiecewiseLegendrePoly{T<:AbstractFloat} <: Function
    nsegments::Int
    polyorder::Int
    xmin::T
    xmax::T

    knots::Vector{T}
    dx::Vector{T}
    data::Matrix{T}
    symm::Int

    # internal
    xm::Vector{T}
    inv_xs::Vector{T}
    norm::Vector{T}

    function PiecewiseLegendrePoly(nsegments, polyorder, xmin, xmax, knots, dx, data,
                                   symm, xm, inv_xs, norm)
        !any(isnan, data) || error("data contains NaN")
        size(knots) == (nsegments + 1,) || error("Invalid knots array")
        issorted(knots) || error("knots must be monotonically increasing")
        dx ≈ diff(knots) || error("dx must work with knots")
        return new{eltype(knots)}(nsegments, polyorder, xmin, xmax, knots, dx, data, symm,
                                  xm, inv_xs, norm)
    end
end

function PiecewiseLegendrePoly(data, p::PiecewiseLegendrePoly; symm)
    return PiecewiseLegendrePoly(p.nsegments, p.polyorder, p.xmin, p.xmax, p.knots, p.dx,
                                 data, symm, p.xm, p.inv_xs, p.norm)
end

function PiecewiseLegendrePoly(data::Matrix, knots::Vector; dx=diff(knots), symm=0)
    polyorder, nsegments = size(data)
    xm = @views (knots[1:(end - 1)] + knots[2:end]) / 2
    inv_xs = 2 ./ dx
    norm = sqrt.(inv_xs)

    return PiecewiseLegendrePoly(nsegments, polyorder, first(knots), last(knots), knots,
                                 dx, data, symm, xm, inv_xs, norm)
end

Base.size(::PiecewiseLegendrePoly) = ()

function (poly::PiecewiseLegendrePoly)(x)
    i, x̃ = split(poly, x)
    return legval(x̃, poly.data[:, i]) * poly.norm[i]
end

function overlap(poly::PiecewiseLegendrePoly, f; rtol=2.3e-16, return_error=false)
    int_result, int_error = quadgk(x -> poly(x) * f(x), poly.xmin, poly.xmax; rtol)
    if return_error
        return int_result, int_error
    else
        return int_result
    end
end

function deriv(poly::PiecewiseLegendrePoly, n=1)
    ddata = legder(poly.data, n)

    scale = poly.inv_xs .^ n
    ddata .*= reshape(scale, (1, :))
    return PiecewiseLegendrePoly(ddata, poly; symm=(-1)^n * poly.symm)
end

function roots(poly::PiecewiseLegendrePoly{T}) where {T}
    m = (poly.xmin + poly.xmax) / 2
    xmin = abs(poly.symm) == 1 ? m : poly.xmin
    xmax = poly.xmax

    rts = roots_irf(poly, Interval(xmin, xmax))
    filter!(isunique, rts)
    rts = map(mid ∘ interval, rts)

    if abs(poly.symm) == 1
        append!(rts, 2m .- rts)
        poly.symm == -1 && push!(rts, m)
    end
    sort!(rts)
    return rts
end

function check_domain(poly, x)
    poly.xmin ≤ x ≤ poly.xmax || throw(DomainError("x is outside the domain"))
    return true
end

function split(poly, x)
    @boundscheck check_domain(poly, x)

    i = max(searchsortedlast(poly.knots, x; lt=≤), 1)
    x̃ = x - poly.xm[i]
    x̃ *= poly.inv_xs[i]
    return i, x̃
end

function scale(poly::PiecewiseLegendrePoly, factor)
    return PiecewiseLegendrePoly(poly.data * factor, poly.knots; dx=poly.dx, symm=poly.symm)
end#poly(x) * poly.norm

###########################
## PiecewiseLegendrePolyArray ##
###########################

"""
    PiecewiseLegendrePolyArray{T, N}

Alias for `Array{PiecewiseLegendrePoly{T}, N}`.
"""
const PiecewiseLegendrePolyArray{T,N} = Array{PiecewiseLegendrePoly{T},N}

function PiecewiseLegendrePolyArray(data::Array{T,N},
                                    knots::Vector{T}) where {T<:AbstractFloat,N}
    polys = PiecewiseLegendrePolyArray{T,N - 2}(undef, size(data)[3:end]...)
    for i in eachindex(polys)#CartesianIndices(axes(data)[3:end])
        polys[i] = PiecewiseLegendrePoly(data[:, :, i], knots)
    end
    return polys
end

function PiecewiseLegendrePolyArray(polys::PiecewiseLegendrePolyArray, knots; dx, symm)# where {T<:AbstractFloat,N<:Integer}
    size(polys) == size(symm) || error("Sizes of polys and symm don't match")
    polys_new = similar(polys)
    for i in eachindex(polys)
        polys_new[i] = PiecewiseLegendrePoly(polys[i].data, knots; dx, symm=symm[i])
    end
    return polys_new
end

function PiecewiseLegendrePolyArray(data, polys::PiecewiseLegendrePolyArray)
    size(data)[3:end] == size(polys) || error("Sizes of data and polys don't match")
    polys = similar(polys)
    for i in eachindex(polys)#CartesianIndices(axes(data)[3:end])
        polys[i] = PiecewiseLegendrePoly(data[:, :, i], knots; symm=u.symm[i])
    end
    return polys
end

(polys::PiecewiseLegendrePolyArray)(x) = map(poly -> poly(x), polys)
function (polys::PiecewiseLegendrePolyArray)(x::Array)
    return reshape(reduce(vcat, polys.(x)), (size(polys)..., size(x)...))
end

function Base.getproperty(polys::PiecewiseLegendrePolyArray, sym::Symbol)
    if sym ∈ (:xmin, :xmax, :knots, :dx, :polyorder, :nsegments, :xm, :inv_xs, :norm)
        return getproperty(first(polys), sym)
    elseif sym == :symm
        return map(poly -> poly.symm, polys)
    else
        error("Unknown property $sym")
    end
end

#########################
## PiecewiseLegendreFT ##
#########################

struct PowerModel{T<:AbstractFloat}
    moments::Vector{T}
end

const DEFAULT_GRID = [range(0; length=2^6);
                      trunc.(Int, exp2.(range(6, 25; length=16 * (25 - 6) + 1)))]
struct PiecewiseLegendreFT{T<:AbstractFloat}
    poly::PiecewiseLegendrePoly{T}
    freq::Symbol
    ζ::Union{Int,Nothing}
    n_asymp::AbstractFloat

    # internal
    model::Union{PowerModel,Nothing}
end

const PiecewiseLegendreFTArray{T,N} = Array{PiecewiseLegendreFT{T},N}

(polys::PiecewiseLegendreFTArray)(n) = map(poly -> poly(n), polys)
function (polys::PiecewiseLegendreFTArray)(n::Array)
    return reshape(reduce(vcat, polys.(n)), (size(polys)..., size(n)...))
end

function PiecewiseLegendreFT(poly, freq=:even, n_asymp=nothing)
    (poly.xmin, poly.xmax) == (-1, 1) || error("Only interval [-1, 1] is supported")
    ζ = Dict(:any => nothing, :even => 0, :odd => 1)[freq] # TODO: type stability
    if isnothing(n_asymp)
        n_asymp = Inf
        model = nothing
    else
        model = power_model(freq, poly)
    end
    return PiecewiseLegendreFT(poly, freq, ζ, float(n_asymp), model)
end

function (polyFT::PiecewiseLegendreFT)(n)
    n = check_reduced_matsubara(n, polyFT.ζ)
    result = _compute_unl_inner(polyFT.poly, n)

    # TODO this doesn't work right because the derivatives needed for the powermodel are wrong at higher orders
    if abs(n) ≥ polyFT.n_asymp
        result = transpose(giw(polyFT.model, n))
    end

    return result
end

function giw(model::PowerModel, wn)
    check_reduced_matsubara(wn)
    T_result = promote_type(typeof(im), typeof(wn), eltype(model.moments))
    result = zero(T_result)
    inv_iw = (iszero(wn) ? identity : inv)(im * π / 2 * wn)
    for mom in reverse(model.moments)
        result += mom
        result *= inv_iw
    end
    # result = evalpoly(inv_iw, vec(model.moments)) / inv_iw
    return result
end

Base.firstindex(::PiecewiseLegendreFT) = 1

function hat(poly::PiecewiseLegendrePoly, freq; n_asymp=nothing)
    return PiecewiseLegendreFT(poly, freq, n_asymp)
end

function Base.extrema(polyFT::PiecewiseLegendreFT, part=nothing, grid=DEFAULT_GRID)
    f = _func_for_part(polyFT, part)
    x₀ = _discrete_extrema(f, grid)
    x₀ .= 2x₀ .+ polyFT.ζ

    return _symmetrize_matsubara(x₀)
end

function _func_for_part(polyFT::PiecewiseLegendreFT, part=nothing)
    if isnothing(part)
        parity = polyFT.poly.symm
        if parity == 1
            part = iszero(polyFT.ζ) ? real : imag
        elseif parity == -1
            part = iszero(polyFT.ζ) ? imag : real
        else
            error("Cannot detect parity")
        end
    end
    return n -> part(polyFT(2n + polyFT.ζ))
end

function _discrete_extrema(f::Function, xgrid)
    fx = f.(xgrid)
    absfx = abs.(fx)

    # Forward differences: derivativesignchange[i] now means that the secant changes sign
    # fx[i+1]. This means that the extremum is STRICTLY between x[i] and
    # x[i+2]
    gx = diff(fx)
    sgx = signbit.(gx)
    derivativesignchange = sgx[1:(end - 1)] .!= sgx[2:end]
    derivativesignchange_a = [derivativesignchange; false; false]
    derivativesignchange_b = [false; false; derivativesignchange]

    a = xgrid[derivativesignchange_a]
    b = xgrid[derivativesignchange_b]
    absf_a = absfx[derivativesignchange_a]
    absf_b = absfx[derivativesignchange_b]
    res = _bisect_discr_extremum.(f, a, b, absf_a, absf_b)

    # We consider the outer point to be extremua if there is a decrease
    # in magnitude or a sign change inwards
    sfx = signbit.(fx)
    if absfx[begin] > absfx[begin + 1] || sfx[begin] != sfx[begin + 1]
        pushfirst!(res, first(xgrid))
    end
    if absfx[end] > absfx[end - 1] || sfx[end] != sfx[end - 1]
        push!(res, last(xgrid))
    end

    return res
end

function _bisect_discr_extremum(f, a, b, absf_a, absf_b)
    d = b - a
    d ≤ 1 && return absf_a > absf_b ? a : b
    d == 2 && return a + 1

    m = (a + b) ÷ 2
    n = m + 1
    absf_m = abs(f(m))
    absf_n = abs(f(n))
    if absf_m > absf_n
        return _bisect_discr_extremum(f, a, n, absf_a, absf_n)
    else
        return _bisect_discr_extremum(f, m, b, absf_m, absf_b)
    end
end

function _symmetrize_matsubara(x₀)
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

function power_moments(stat, deriv_x1)
    statsign = (stat == :odd) ? -1 : 1
    mmax = length(deriv_x1)
    coeff_lm = @. ((-1)^(1:mmax) + statsign) * deriv_x1
    return -statsign / √2 * coeff_lm
end

function power_model(stat, poly)
    deriv_x1 = derivs(poly, 1)
    moments = power_moments(stat, deriv_x1)
    return PowerModel(moments)
end

check_reduced_matsubara(n::Integer) = n
check_reduced_matsubara(n::Integer, ζ) = (n & 1 == ζ) ? n : error("n must be even")

########################
### Helper Functions ###
########################

function refine_grid(knots, α)
    knots_new = eltype(knots)[]
    for i in eachindex(knots)[begin:(end - 1)]
        interknots = range(knots[i], knots[i + 1]; length=α + 1)[begin:(end - 1)]
        append!(knots_new, interknots)
    end
    append!(last(knots))
    return knots_new
end

function _compute_unl_inner(poly, wn)
    data_sc = poly.data ./ reshape(√2 * poly.norm, (1, :))

    p = range(0; length=poly.polyorder)
    wred = π / 2 * wn
    phase_wi = _phase_stable(poly, wn)
    t_pin = _get_tnl.(p', wred * poly.dx / 2) .* phase_wi

    return sum(transpose(t_pin) .* data_sc)
end

function _get_tnl(l, w)
    result = 2im^l * sphericalbesselj(l, abs(w))
    return (w < 0 ? conj : identity)(result)
end

choose(a, choices) = [choices[a[i]][i] for i in eachindex(a)]

function shift_xmid(knots, dx)
    dx_half = dx ./ 2
    xmid_m1 = cumsum(dx) .- dx_half
    xmid_p1 = -reverse(cumsum(reverse(dx))) + dx_half
    xmid_0 = knots[2:end] - dx_half

    shift = round.(Int, xmid_0)
    diff = choose(shift .+ 2, (xmid_m1, xmid_0, xmid_p1))
    return diff, shift
end

function _phase_stable(poly, wn)
    xmid_diff, extra_shift = shift_xmid(poly.knots, poly.dx)

    if wn isa Integer
        shift_arg = wn * xmid_diff
    else
        delta_wn, wn = modf(wn)
        wn = trunc(Int, wn)
        shift_arg = wn * xmid_diff
        shift_arg .+= delta_wn * (extra_shift + xmid_diff)
    end

    phase_shifted = exp.(im * π / 2 * shift_arg)
    corr = im .^ mod.(wn * (extra_shift .+ 1), 4)
    return corr .* phase_shifted
end

#######################
### HERE BE DRAGONS ###
#######################

legendreP(l, x) = Plm(l, 0, x)

function legval(x, c)
    x = clamp(x, -1, 1)
    return sum(c .* legendreP(range(0; length=length(c)), x))
end

"""
    legder

Adapted from https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/polynomial/legendre.py#L612-L701
"""
legder(cc::AbstractMatrix, cnt=1; dims=1) = mapslices(c -> legder(c, cnt), cc; dims)

function legder(c::AbstractVector{T}, cnt=1) where {T}
    cnt ≥ 0 || error("The order of derivation must be non-negative")
    c = copy(c)
    cnt == 0 && return c
    n = length(c)
    if n ≤ cnt
        c = [zero(T)]
    else
        for _ in 1:cnt
            n -= 1
            der = Vector{T}(undef, n)
            for j in n:-1:2
                der[j] = (2j - 1) * c[j + 1]
                c[j - 1] += c[j + 1]
            end
            der[1] = c[2]
            c = der
        end
    end
    return c
end