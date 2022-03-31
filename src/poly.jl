import AssociatedLegendrePolynomials: Plm
import IntervalRootFinding: roots as roots_irf, Interval, isunique, interval, mid
import SpecialFunctions: sphericalbesselj # TODO: if Bessels.jl has been released, use that instead
import QuadGK: quadgk
import Memoization: @memoize

export PiecewiseLegendrePolyArray, roots, hat, overlap

struct PiecewiseLegendrePoly{T<:Real}
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
    ddata = poly.data
    for _ in 1:n
        ddata = legder.(eachcol(ddata))
        ddata = reduce(hcat, ddata)
    end

    scale = poly.inv_xs .^ n
    ddata .*= scale'
    return PiecewiseLegendrePoly(ddata, poly; symm=(-1)^n * poly.symm)
end

function roots(poly::PiecewiseLegendrePoly)
    rts = roots_irf(x -> poly(x), x -> deriv(poly)(x), Interval(poly.xmin, poly.xmax))
    all(isunique, rts) || error("Failed to determine roots uniquely")
    return map(mid ∘ interval, rts)
end

in_domain(poly, x) = poly.xmin <= x <= poly.xmax

function split(poly, x)
    # in_domain(poly, x) || throw(DomainError("x must be in [$(poly.xmin), $(poly.xmax)]"))

    i = clamp(searchsortedlast(poly.knots, x; lt=<), 1, poly.nsegments)
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

function PiecewiseLegendrePolyArray(data::Array{T,N}, knots::Vector{T}) where {T<:Real,N}
    polys = PiecewiseLegendrePolyArray{T,N - 2}(undef, size(data)[3:end]...)
    for i in eachindex(polys)#CartesianIndices(axes(data)[3:end])
        polys[i] = PiecewiseLegendrePoly(data[:, :, i], knots)
    end
    return polys
end

function PiecewiseLegendrePolyArray(polys::PiecewiseLegendrePolyArray, knots; dx, symm)# where {T<:Real,N<:Integer}
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

struct PowerModel
    moments::Matrix{Float64}
    nmom::Int
    nl::Int
end

const DEFAULT_GRID = [range(0; length=2^6);
                      trunc.(Int, 2 .^ range(6, 25; length=16 * (25 - 6) + 1))]
struct PiecewiseLegendreFT{T<:Real}
    poly::PiecewiseLegendrePoly{T}
    freq::Symbol
    ζ::Union{Int,Nothing}
    n_asymp::Number

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
    return PiecewiseLegendreFT(poly, freq, ζ, n_asymp, model)
end

function (polyFT::PiecewiseLegendreFT)(n)
    n = check_reduced_matsubara(n, polyFT.ζ)
    result = compute_unl_inner(polyFT.poly, n)

    if abs(n) >= polyFT.n_asymp
        result = transpose(giw(polyFT.model))
    end

    return result
end

Base.firstindex(::PiecewiseLegendreFT) = 1

function hat(poly::PiecewiseLegendrePoly, freq; n_asymp=nothing)
    return PiecewiseLegendreFT(poly, freq, n_asymp)
end

function derivs(ppoly, x)
    res = [ppoly(x)]
    for _ in range(1, ppoly.polyorder - 1)
        ppoly = deriv(ppoly)
        push!(res, ppoly(x))
    end
    return res
end

PowerModel(moments) = PowerModel(moments, size(moments)...)

function giw_ravel(wn)
    # TODO
    return error("not implemented")
end

function power_moments(stat, deriv_x1)
    statsign = Dict(:odd => -1, :even => 1)[stat]
    mmax, lmax = size(deriv_x1)
    m = range(0; length=mmax)
    l = range(0; length=lmax)
    coeff_lm = @. ((-1)^(m + 1) + statsign * (-1)^(l')) * deriv_x1
    return -statsign / √2 * coeff_lm
end

function power_model(stat, poly)
    deriv_x1 = derivs(poly, 1)
    ndims(deriv_x1) == 1 && (deriv_x1 = deriv_x1[:, :])
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

function compute_unl_inner(poly, wn)
    data_sc = poly.data ./ reshape(poly.norm, (1, :))

    p = range(0; length=poly.polyorder)
    wred = π / 2 * wn
    phase_wi = phase_stable(poly, wn)
    t_pin = get_tnl.(p', wred * poly.dx / 2) .* phase_wi

    return sum(transpose(t_pin) .* data_sc)
end

function get_tnl(l, w)
    result = 2 * im^l * sphericalbesselj(l, abs(w))
    return w < 0 ? conj(result) : result
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

function phase_stable(poly, wn)
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

@memoize legendreP(l, x) = Plm(l, 0, x)

function legval(x, c)
    x = clamp(x, -1, 1)
    return sum(c .* legendreP(range(0; length=length(c)), x))
end

function legder(c)
    n = length(c)
    if n <= 1
        return zeros(1)
    else # n > 1
        n -= 1
        der = Array{eltype(c)}(undef, n)
        for j in n:-1:3
            der[j] = (2j - 1) * c[j + 1]
            c[j - 1] += c[j]
        end
        if n > 1
            der[2] = 3c[3]
        end
        der[1] = c[2]
        return der
    end
end