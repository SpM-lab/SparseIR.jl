import AssociatedLegendrePolynomials: Plm
import QuadGK: gauss

export legendre, legvander, legendre_collocation, Rule, piecewise, quadrature, reseat

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
struct Rule{T}
    x::Vector{T}
    w::Vector{T}
    a::T
    b::T

    function Rule(x, w, a=-1, b=1)
        a ≤ b || error("a must be ≤ b")
        all(≤(b), x) || error("x must be ≤ b")
        all(≥(a), x) || error("x must be ≥ a")
        issorted(x) || error("x must be strictly increasing")
        length(x) == length(w) || error("x and w must have the same length")
        return new{eltype(x)}(x, w, a, b)
    end
end

"""
    quadrature(rule, f)

Approximate `f`'s integral.
"""
function quadrature(rule, f)
    return sum(rule.w .* f.(rule.x))
end

"""
    reseat(rule, a, b)

Reseat quadrature rule to new domain.
"""
function reseat(rule::Rule, a, b)
    scaling = (b - a) / (rule.b - rule.a)
    x = (rule.x .- (rule.a + rule.b)/2) * scaling .+ (a + b) / 2
    w = rule.w * scaling
    return Rule(x, w, a, b)
end

"""
    scale(rule, factor)

Scale weights by `factor`.
"""
scale(rule, factor) = Rule(rule.x, rule.w * factor, rule.a, rule.b)

"""
    piecewise(rule, edges)

Piecewise quadrature with the same quadrature rule, but scaled.
"""
function piecewise(rule, edges)
    start = @view edges[begin:(end - 1)]
    stop = @view edges[(begin + 1):end]
    all(stop .> start) || error("edges must be monotonically increasing")
    return joinrules(reseat.(Ref(rule), start, stop))
end

"""
    joinrules(rules)

Join multiple Gauss quadratures together.
"""
function joinrules(rules)
    for i in Iterators.drop(eachindex(rules), 1)
        rules[i - 1].b == rules[i].a || error("rules must be contiguous")
    end

    x = reduce(vcat, rule.x for rule in rules)
    w = reduce(vcat, rule.w for rule in rules)
    a = first(rules).a
    b = last(rules).b

    return Rule(x, w, a, b)
end

"""
    legendre(n[, T])

Gauss-Legendre quadrature with `n` points.
"""
legendre(n) = Rule(gauss(n)...)
legendre(n, T) = Rule(gauss(T, n)...)

"""
    legvander(x, deg)

Pseudo-Vandermonde matrix of degree `deg`.
"""
legvander(x, deg) = Plm(0:deg, 0, x)

"""
    legendre_collocation(rule, n=length(rule.x))

Generate collocation matrix from Gauss-Legendre rule.
"""
function legendre_collocation(rule, n=length(rule.x))
    res = (legvander(rule.x, n - 1) .* rule.w)'
    invnorm = range(0.5; length=n)
    res .*= invnorm
    return res
end

function Base.convert(::Type{Rule{T}}, rule::Rule{S}) where {T,S<:AbstractFloat}
    return Rule(T.(rule.x), T.(rule.w), T(rule.a), T(rule.b))
end
