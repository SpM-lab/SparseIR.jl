function find_all(f, xgrid::AbstractVector)
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
    fb = @view fx[where_b]

    x_bisect = bisect_cont.(f, a, b, fa, fb)

    return sort!([x_hit; x_bisect])
end

function bisect_cont(f, a, b, fa, fb)
    while true
        mid = (a + b) / 2
        fmid = f(mid)
        if signbit(fa) ≠ signbit(fmid)
            b, fb = mid, fmid
        else
            a, fa = mid, fmid
        end
        isapprox(a, b; rtol=0, atol=1e-10) && return mid
    end
end

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
