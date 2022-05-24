function find_all(f, xgrid::AbstractVector)
    fx = f.(xgrid)
    hit = iszero.(fx)
    x_hit = @view xgrid[hit]

    sign_change = @. signbit(@view fx[1:(end - 1)]) ≠ signbit(@view fx[2:end])
    @. sign_change &= ~(@view hit[1:(end - 1)]) & ~(@view hit[2:end])
    any(sign_change) || return x_hit

    where_a = [sign_change; false]
    where_b = [false; sign_change]
    a = @view xgrid[where_a]
    b = @view xgrid[where_b]
    fa = @view fx[where_a]
    fb = @view fx[where_b]

    x_bisect = _bisect_cont.(f, a, b, fa, fb)

    return sort!([x_hit; x_bisect])
end

function _bisect_cont(f, a, b, fa, fb)
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

function _refine_grid(grid, alpha)
    xbegin = @view grid[begin:(end - 1)]
    xend = @view grid[(begin + 1):end]

    newgrid_iter = range.(xbegin, xend; length=alpha + 1)
    newgrid = mapreduce(collect ∘ (x -> x[begin:(end - 1)]), vcat, newgrid_iter)
    push!(newgrid, last(grid))
    return newgrid
end
