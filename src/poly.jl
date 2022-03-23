import AssociatedLegendrePolynomials: Plm
import LinearAlgebra: dot

struct PiecewiseLegendrePoly
    nsegments::Any
    polyorder::Any
    xmin::Any
    xmax::Any

    knots::Any
    dx::Any
    data::Any
    symm::Any

    # internal
    xm::Any
    inv_xs::Any
    norm::Any
end

function PiecewiseLegendrePoly(data, knots; dx=nothing, symm=nothing)
    !any(isnan, data) || error("data contains NaN")
    if knots isa PiecewiseLegendrePoly
        isnothing(dx) && !isnothing(symm) || error("wrong arguments")
        knots.data = data
        knots.symm = symm
        return knots
    end

    ndims(data) >= 2 || error("data must be at least 2-dimensional")
    polyorder, nsegments = first(size(data), 2)
    size(knots) == (nsegments + 1,) || error("Invalid knots array")
    issorted(knots) || error("knots must be monotonically increasing")
    if isnothing(symm)
        # TODO: infer symmetry from data
        symm = zeros(size(data)[3:end])
    else
        size(symm) == size(data)[3:end] || error("shape mismatch")
    end
    if isnothing(dx)
        dx = diff(knots)
    else
        dx ≈ diff(knots) || error("dx must work with knots")
    end

    return PiecewiseLegendrePoly(nsegments, polyorder, first(knots), last(knots), knots, dx,
                                 data, symm,
                                 (knots[(begin + 1):end] .+ knots[begin:(end - 1)]) ./ 2,
                                 2 ./ dx, sqrt.(2 ./ dx))
end

function getindex(poly, l...)
    new_symm = poly.symm[l...]
    new_data = poly.data[:, :, l...]
    return PiecewiseLegendrePoly(new_data, poly; symm=new_symm)
end

function (poly::PiecewiseLegendrePoly)(x)
    # i, x̃ = split.(Ref(poly), x isa Array ? x : [x])

    res = split.(Ref(poly), x isa Array ? x : [x])
    i, x̃ = first.(res), last.(res) # TODO: this is kinda hacky

    # Evaluate for all values of l.  x and data array must be
    # broadcast'able against each other, so we append dimensions here
    func_dims = ndims(poly.data) - 2
    data = poly.data[:, i, fill(:, func_dims)...] # TODO: handle trailing dimensions properly
    datashape = (size(i)..., ones(Int, func_dims)...)
    res = legval(reshape(x̃, datashape), data)
    res .*= poly.norm[reshape(i, datashape)] # TODO idk if this is right

    # Finally, exchange the x and vector dimensions
    order = ((ndims(i) + 1):(ndims(i) + func_dims)..., 1:ndims(i)...)
    return permutedims(res, order)
end

function value(poly::PiecewiseLegendrePoly, l, x)
    ndims(poly.data) == 3 || error("data must be 3-dimensional")
    @show size(l)
    @show size(x)
    # l, x = 
end

function overlap(poly::PiecewiseLegendrePoly, f; rtol=2.3e-16, return_error=false)
    # TODO
    return error("not implemented")
end

function deriv(poly::PiecewiseLegendrePoly, n=1)
    # TODO
    ddata = legder(poly.data, n)

    scale_shape = (1, :, fill(1, ndims(poly.data) - 2)...)
    scale = poly.inv_xs .^ n
    ddata .*= reshape(scale, scale_shape)
    return PiecewiseLegendrePoly(ddata, poly; symm=(-1) .^ n .* poly.symm)
end

in_domain(poly, x) = poly.xmin <= x <= poly.xmax

function split(poly, x)
    in_domain(poly, x) || throw(DomainError("x must be in [$poly.xmin, $poly.xmax]"))

    i = clamp(searchsortedlast(poly.knots, x), 1, poly.nsegments)
    x̃ = x - poly.xm[i]
    x̃ *= poly.inv_xs[i]
    return i, x̃
end

const DEFAULT_GRID = [range(0; length=2^6);
                      trunc.(Int, 2 .^ range(6, 25; length=16 * (25 - 6) + 1))]
struct PiecewiseLegendreFT
    poly::Any
    freq::Any
    zeta::Any
    n_asymp::Any

    # internal
    model::Any
end

function PiecewiseLegendreFT(poly, freq=:even, n_asymp=nothing)
    (poly.xmin, poly.xmax) == (-1, 1) || error("Only interval [-1, 1] is supported")
    zeta = Dict(:any => nothing, :even => 0, :odd => 1)[freq] # TODO: type stability
    if isnothing(n_asymp)
        n_asymp = Inf
        model = nothing
    else
        model = power_model(freq, poly)
    end
    return PiecewiseLegendreFT(poly, freq, zeta, n_asymp, model)
end

hat(poly, freq, n_asymp=nothing) = PiecewiseLegendreFT(poly, freq, n_asymp)

function derivs(ppoly, x)
    res = [ppoly(x)]
    for _ in range(ppoly.polyorder - 1)
        ppoly = deriv(ppoly)
        push!(res, ppoly(x))
    end
    return res
end

struct PowerModel
    moments
    nmom
    nl
end

PowerModel(moments) = PowerModel(moments, size(moments)...)

function giw_ravel(wn)
    # TODO
    error("not implemented")
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
    deriv_x1 = derivs(poly; x=1)
    ndims(deriv_x1) == 1 && (deriv_x1 = deriv_x1[:, :])
    moments = power_moments(stat, deriv_x1)
    return PowerModel(moments)
end

#######################
### HERE BE DRAGONS ###
#######################

# legval(x, c) = dot(c, Plm(range(0; length=size(c, 1)), 0, x))

# function legval(x, c)
#     legs = Plm(range(0; length=size(c, 1)), 0, x)
#     legs = permutedims(legs, circshift(1:ndims(c), 1))
#     legs .*= c
#     return dropdims(sum(legs; dims=1); dims=1)
# end

function legval(x, c)
    all = ntuple(_ -> :, ndims(c) - 1)
    nd = size(c, 1)

    if nd == 1
        c0 = c[begin, all...]
        c1 = 0
    elseif nd == 2
        c0 = c[begin, all...]
        c1 = c[begin + 1, all...]
    else
        c0 = c[end - 1, all...]
        c1 = c[end, all...]
        for i in range(3, nd)
            tmp = c0
            nd -= 1
            c0 = c[end - i + 1, all...] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 .* x * (2nd - 1)) / nd
        end
    end
    return c0 + c1 .* x
end

function legder(c, m=1)
    all = ntuple(_ -> :, ndims(c) - 1)

    iszero(m) && return c

    n = size(c, 1)
    if m >= n
        c = c[[begin], all...] * 0
    else
        for _ in 1:m
            n -= 1
            der = Array{eltype(c)}(undef, n, size(c)[(begin + 1):end])
            for j in n:-1:3
                der[j, all...] .= (2j - 1) .* c[j + 1, all...]
                c[j - 1, all...] .+= c[j, all...]
            end
            if n > 1
                der[2, all...] .= 3c[3, all...]
            end
            der[1, all...] .= c[2, all...]
            c = der
        end
    end
    return c
end