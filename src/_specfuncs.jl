# Minimally adapted from https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/special/_spherical_bessel.pxd#L74
function sphericalbesselj(n::Integer, x::T) where {T<:AbstractFloat}
    isnan(x) && return x
    n < 0 && throw(DomainError(n, "n must be non-negative"))
    isinf(x) && return zero(T)
    iszero(x) && return iszero(n) ? one(T) : zero(T)

    if n > 0 && n ≥ x
        # We don't use SpecialFunctions.sphericalbesselj directly because it errors out on large x
        return T(SpecialFunctions.sphericalbesselj(n, x))
    else
        return _sphericalbesselj(n, x)
    end
end

function _sphericalbesselj(n::Integer, x::T) where {T<:AbstractFloat}
    invx = inv(x)
    s0 = sin(x) * invx
    iszero(n) && return s0

    s1 = (s0 - cos(x)) * invx
    isone(n) && return s1

    sn = zero(T)
    for idx in 2:n
        sn = (2idx - 1) * invx * s1 - s0
        s0 = s1
        s1 = sn
    end
    return sn
end

# Adapted from https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/polynomial/legendre.py#L832-L914
function legval(x, c::AbstractVector)
    nd = length(c)
    nd ≥ 2 || return last(c)

    c0, c1 = @inbounds c[nd - 1], c[nd]
    @inbounds for j in (nd - 2):-1:1
        k = j / (j + 1)
        c0, c1 = c[j] - c1 * k, c0 + c1 * x * (k + 1)
    end
    return c0 + c1 * x
end

# Adapted from https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/polynomial/legendre.py#L1126-L1176
"""
legvander(x, deg)

Pseudo-Vandermonde matrix of degree `deg`.
"""
function legvander(x::AbstractVector{T}, deg::Integer) where {T}
    deg ≥ 0 || throw(DomainError(deg, "degree needs to be non-negative"))

    vsize = (length(x), deg + 1)
    v = Matrix{T}(undef, vsize...)

    # Use forward recursion to generate the entries. This is not as accurate
    # as reverse recursion in this application but it is more efficient.
    @inbounds begin
        v[:, 1] .= one(T)
        if deg > 0
            v[:, 2] .= x
            for i in 2:deg
                invi = inv(i)
                @views @. v[:, i + 1] = v[:, i] * x * (2 - invi) - v[:, i - 1] * (1 - invi)
            end
        end
    end

    return v
end

# Adapted from https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/polynomial/legendre.py#L612-L701
"""
    legder
"""
function legder(c::AbstractMatrix{T}, cnt=1) where {T}
    cnt ≥ 0 || throw(DomainError(cnt, "The order of derivation needs to be non-negative"))
    cnt == 0 && return c

    c = deepcopy(c)
    n, m = size(c)
    if cnt ≥ n
        return zeros(T, (1, m))
    else
        @views @inbounds for _ in 1:cnt
            n -= 1
            der = Matrix{T}(undef, n, m)
            for j in n:-1:3
                @. der[j, :] = (2j - 1) * c[j + 1, :]
                @. c[j - 1, :] += c[j + 1, :]
            end
            if n > 1
                @. der[2, :] = 3c[3, :]
            end
            @. der[1, :] = c[2, :]
            c = der
        end
    end
    return c
end