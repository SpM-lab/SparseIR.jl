function find_all(f, xgrid::AbstractVector{T}) where {T}
    fx = f.(xgrid)
    hit = iszero.(fx)
    x_hit = @view xgrid[hit]

    sign_change = @views @. signbit(fx[begin:(end - 1)]) ≠ signbit(fx[(begin + 1):end])
    @. @views sign_change &= ~hit[begin:(end - 1)] & ~hit[(begin + 1):end]
    any(sign_change) || return x_hit

    where_a = [sign_change; false]
    where_b = [false; sign_change]
    a = @view xgrid[where_a]
    b = @view xgrid[where_b]
    fa = @view fx[where_a]

    ϵ_x = if T <: AbstractFloat
        eps(T) * maximum(abs, xgrid)
    else
        0
    end
    x_bisect = bisect.(f, a, b, fa, ϵ_x)

    return sort!([x_hit; x_bisect])
end

function bisect(f, a, b, fa, ϵ_x)
    while true
        mid = midpoint(a, b)
        closeenough(a, mid, ϵ_x) && return mid
        fmid = f(mid)
        if signbit(fa) ≠ signbit(fmid)
            b = mid
        else
            a, fa = mid, fmid
        end
    end
end

@inline closeenough(a::T, b::T, ϵ) where {T<:AbstractFloat} = isapprox(a, b; rtol=0, atol=ϵ)
@inline closeenough(a::T, b::T, _) where {T<:Integer} = a == b

function refine_grid(grid, ::Val{α}) where {α}
    n = length(grid)
    newn = α * (n - 1) + 1
    newgrid = Vector{eltype(grid)}(undef, newn)

    @inbounds for i in 1:(n - 1)
        xb = grid[i]
        xe = grid[i + 1]
        Δx = (xe - xb) / α
        newi = α * (i - 1)
        @simd for j in 1:α
            newgrid[newi + j] = xb + Δx * (j - 1)
        end
    end
    newgrid[end] = last(grid)
    return newgrid
end

function discrete_extrema(f::Function, xgrid)
    fx::Vector{Float64} = f.(xgrid)
    absfx               = abs.(fx)

    # Forward differences: derivativesignchange[i] now means that the secant 
    # changes sign fx[i+1]. This means that the extremum is STRICTLY between 
    # x[i] and x[i+2].
    gx                     = diff(fx)
    sgx                    = signbit.(gx)
    derivativesignchange   = @views (sgx[begin:(end - 1)] .≠ sgx[(begin + 1):end])
    derivativesignchange_a = [derivativesignchange; false; false]
    derivativesignchange_b = [false; false; derivativesignchange]

    a      = xgrid[derivativesignchange_a]
    b      = xgrid[derivativesignchange_b]
    absf_a = absfx[derivativesignchange_a]
    absf_b = absfx[derivativesignchange_b]
    res    = bisect_discr_extremum.(f, a, b, absf_a, absf_b)

    # We consider the outer points to be extrema if there is a decrease
    # in magnitude or a sign change inwards
    sfx = signbit.(fx)
    if absfx[begin] > absfx[begin + 1] || sfx[begin] ≠ sfx[begin + 1]
        pushfirst!(res, first(xgrid))
    end
    if absfx[end] > absfx[end - 1] || sfx[end] ≠ sfx[end - 1]
        push!(res, last(xgrid))
    end

    return res
end

function bisect_discr_extremum(f, a, b, absf_a, absf_b)
    d = b - a

    d <= 1 && return absf_a > absf_b ? a : b
    d == 2 && return a + 1

    m      = midpoint(a, b)
    n      = m + 1
    absf_m = abs(f(m))
    absf_n = abs(f(n))

    if absf_m > absf_n
        return bisect_discr_extremum(f, a, n, absf_a, absf_n)
    else
        return bisect_discr_extremum(f, m, b, absf_m, absf_b)
    end
end

# This implementation of `midpoint` is performance-optimized but safe
# only if `lo <= hi`.
@inline midpoint(lo::T, hi::T) where {T<:Integer} = lo + ((hi - lo) >>> 0x01)
@inline midpoint(lo::T, hi::T) where {T<:AbstractFloat} = lo + ((hi - lo) * T(0.5))
@inline midpoint(lo, hi) = midpoint(promote(lo, hi)...)
