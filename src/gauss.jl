@doc raw"""
    Rule{T<:AbstractFloat}

Quadrature rule.

Approximation of an integral over `[a, b]` by a sum over discrete points `x` with weights `w`:
```math
    ∫ f(x) ω(x) dx ≈ ∑_i f(x_i) w_i
```
where we generally have superexponential convergence for smooth ``f(x)`` in 
the number of quadrature points.
"""
struct Rule{T<:AbstractFloat}
    x          :: Vector{T}
    w          :: Vector{T}
    a          :: T
    b          :: T
    x_forward  :: Vector{T}
    x_backward :: Vector{T}

    function Rule(x::Vector{T}, w::Vector{T}, a=-one(T), b=one(T),
                  x_forward=x .- a, x_backward=b .- x) where {T}
        a ≤ b || error("a must be ≤ b")
        for xx in x
            a ≤ xx ≤ b || error("all x must be in [a, b], found $xx outside of [$a, $b]")
        end
        issorted(x) || error("x must be strictly increasing")
        length(x) == length(w) ||
            throw(DimensionMismatch("x and w must have the same length"))
        return new{T}(x, w, a, b, x_forward, x_backward)
    end
end

"""
    reseat(rule, a, b)

Reseat quadrature rule to new domain.
"""
function reseat(rule::Rule, a, b)
    scaling = (b - a) / (rule.b - rule.a)
    x = @. (rule.x - (rule.a + rule.b) / 2) * scaling + (a + b) / 2
    w = rule.w * scaling
    x_forward = rule.x_forward * scaling
    x_backward = rule.x_backward * scaling
    return Rule(x, w, a, b, x_forward, x_backward)
end

"""
    scale(rule, factor)

Scale weights by `factor`.
"""
function scale(rule, factor)
    Rule(rule.x, rule.w * factor, rule.a, rule.b, rule.x_forward, rule.x_backward)
end

"""
    piecewise(rule, edges)

Piecewise quadrature with the same quadrature rule, but scaled.
"""
function piecewise(rule, edges::Vector)
    issorted(edges) || error("edges must be monotonically increasing")
    start = edges[begin:(end - 1)]
    stop  = edges[(begin + 1):end]
    return joinrules([reseat(rule, a, b) for (a, b) in zip(start, stop)])
end

"""
    joinrules(rules)

Join multiple Gauss quadratures together.
"""
function joinrules(rules::AbstractVector{Rule{T}}) where {T}
    @inbounds for i in Iterators.drop(eachindex(rules), 1)
        rules[i - 1].b == rules[i].a || error("rules must be contiguous")
    end

    x = reduce(vcat, rule.x for rule in rules; init=T[])
    w = reduce(vcat, rule.w for rule in rules; init=T[])
    a = first(rules).a
    b = last(rules).b

    x_forward  = reduce(vcat, rule.x_forward .+ (rule.a - a) for rule in rules; init=T[])
    x_backward = reduce(vcat, rule.x_backward .+ (b - rule.b) for rule in rules; init=T[])

    return Rule(x, w, a, b, x_forward, x_backward)
end

"""
    legendre(n[, T])

Gauss-Legendre quadrature with `n` points on [-1, 1].
"""
legendre(n, ::Type{T}=Float64) where {T} = Rule(gauss(T, n)...)

"""
    legendre_collocation(rule, n=length(rule.x))

Generate collocation matrix from Gauss-Legendre rule.
"""
function legendre_collocation(rule, n=length(rule.x))
    res = permutedims(legvander(rule.x, n - 1) .* rule.w)
    invnorm = range(0.5; length=n)
    res .*= invnorm
    return res
end

function Base.convert(::Type{Rule{T}}, rule::Rule) where {T}
    Rule(T.(rule.x), T.(rule.w), T(rule.a), T(rule.b),
         T.(rule.x_forward), T.(rule.x_backward))
end
