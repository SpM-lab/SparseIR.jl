import SpecialFunctions: sphericalbesselj as sphericalbesselj_sf
# We don't use SpecialFunctions.sphericalbesselj directly because it errors out on large x

# Minimally adapted from https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/special/_spherical_bessel.pxd#L74
function sphericalbesselj(n::Integer, x::T) where {T<:AbstractFloat}
    isnan(x) && return x
    n < 0 && throw(DomainError("n must be non-negative"))
    isinf(x) && return zero(T)
    iszero(x) && return iszero(n) ? one(T) : zero(T)

    if n > 0 && n ≥ x
        return T(sphericalbesselj_sf(n, x))
    end

    invx = inv(x)

    s0 = sin(x) * invx
    iszero(n) && return s0

    s1 = (s0 - cos(x)) * invx
    isone(n) && return s1

    sn = zero(T)
    for idx in 2:n
        sn = (2idx - 1) * invx * s1 - s0
        s0, s1 = s1, sn
        # Overflow occurred already: terminate recurrence.
        isnan(sn) && return sn
    end
    return sn
end
