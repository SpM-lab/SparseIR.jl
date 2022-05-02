module _SpecFuncs

export sphericalbesselj, legval, legvander, legder

using SpecialFunctions: sphericalbesselj as sphericalbesselj_sf
# We don't use SpecialFunctions.sphericalbesselj directly because it errors out on large x

# Minimally adapted from https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/special/_spherical_bessel.pxd#L74
function sphericalbesselj(n::Integer, x::T) where {T<:AbstractFloat}
    isnan(x) && return x
    n < 0 && throw(DomainError(n, "n must be non-negative"))
    isinf(x) && return zero(T)
    iszero(x) && return iszero(n) ? one(T) : zero(T)

    if n > 0 && n ≥ x
        return T(sphericalbesselj_sf(n, x))
    end

    return _sphericalbesselj(n, x)
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
        # Overflow occurred already: terminate recurrence.
        isnan(sn) && return sn
    end
    return sn
end

# # Adapted from https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/polynomial/legendre.py#L832-L914
# function legval(x, c::Array{T,N}) where {T,N}
#     # trailing dimensions
#     t = ntuple(_ -> :, N - 1)

#     # Pad the coefficient array if it contains only one (i.e. the constant 
#     # polynomial's) coefficient.
#     size(c, 1) ≥ 2 || (c = [c; zero(c)])

#     c0 = c[end - 1, t...]
#     c1 = c[end, t...]
#     tmp = similar(c0)

#     nd = size(c, 1)
#     @inbounds @views for i in 2:(size(c, 1) - 1)
#         nd -= 1
#         invnd = inv(nd)
#         @. tmp = c0
#         @. c0 = c[end - i, t...] - c1 * (1 - invnd)
#         @. c1 = tmp + c1 * x * (2 - invnd)
#     end

#     return c0 .+ c1 .* x
# end

# Adapted from https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/polynomial/legendre.py#L832-L914
function legval(x, c::AbstractVector)
    # Pad the coefficient vector if it contains only one (i.e. the constant 
    # polynomial's) coefficient.
    length(c) ≥ 2 || (c = [c; zero(c)])
    nd = length(c)

    c0, c1 = c[nd - 1], c[nd]
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
function legvander(x::Array{T,N}, deg::Integer) where {T,N}
    deg ≥ 0 || throw(DomainError(deg, "legvander needs a non-negative degree"))

    # leading dimensions
    l = ntuple(_ -> :, N)

    vsize = (size(x)..., deg + 1)
    v = Array{T}(undef, vsize...)

    # Use forward recursion to generate the entries. This is not as accurate
    # as reverse recursion in this application but it is more efficient.
    v[l..., 1] .= one(T)
    if deg > 0
        v[l..., 2] .= x
        @inbounds @views for i in 2:deg
            invi = inv(i)
            @. v[l..., i + 1] = v[l..., i] * x * (2 - invi) - v[l..., i - 1] * (1 - invi)
        end
    end

    return v
end

# Adapted from https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/polynomial/legendre.py#L612-L701
"""
    legder
"""
legder(cc::AbstractMatrix, cnt=1; dims=1) = mapslices(c -> legder(c, cnt), cc; dims)

function legder(c::AbstractVector{T}, cnt=1) where {T}
    cnt ≥ 0 || throw(DomainError(cnt, "The order of derivation must be non-negative"))
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

end #module
