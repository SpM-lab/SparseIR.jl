import AssociatedLegendrePolynomials: Plm
import FastGaussQuadrature: gausslegendre
using LinearAlgebra: dot

@doc raw"""
    Rule{T<:Real}

Quadrature rule.

Approximation of an integral over `[a, b]` by a sum over discrete points `x` with weights `w`:
```math
    ∫ f(x) ω(x) dx ≈ f.(x) ⋅ w
```
where we generally have superexponential convergence for smooth ``f(x)`` in 
the number of quadrature points.
"""
struct Rule{T<:Real}
    x::Vector{T}
    w::Vector{T}
    a::T
    b::T

    function Rule(x, w, a=-1, b=1)
        a <= b || error("a must be <= b")
        all(x .<= b) || error("x must be <= b")
        all(x .>= a) || error("x must be >= a")
        all(diff(x) .> 0) || error("x must be strictly increasing")
        length(x) == length(w) || error("x and w must have the same length")
        return new{eltype(x)}(x, w, a, b)
    end
end

"Approximate `f`'s integral."
function quadrature(rule, f)
    return dot(f.(rule.x), rule.w)
end

"Reseat quadrature rule to new domain."
function reseat(rule, a, b)
    scaling = (b - a) / (rule.b - rule.a)
    x = (rule.x .- rule.a) * scaling .+ a
    w = rule.w * scaling
    return Rule(x, w, a, b)
end

"Scale weights by factor."
scale(rule, factor) = Rule(rule.x, rule.w * factor, rule.a, rule.b)

"Piecewise quadrature with the same quadrature rule, but scaled."
function piecewise(rule, edges)
    start = @view edges[begin:(end - 1)]
    stop = @view edges[(begin + 1):end]
    all(stop .> start) || error("edges must be monotonically increasing")
    return joinrules(reseat.(Ref(rule), start, stop))
end

"Join multiple Gauss quadratures together."
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

"Gauss-Legendre quadrature."
legendre(n) = Rule(gausslegendre(n)...)

"Pseudo-Vandermonde matrix of given degree."
legvander(x, deg) = Plm(0:deg, 0, x)

"Generate collocation matrix from Gauss-Legendre rule."
function legendre_collocation(rule, n=length(rule.x))
    res = (legvander(rule.x, n - 1) .* rule.w)'
    invnorm = range(0.5; length=n)
    res .*= invnorm
    return res
end

function Base.convert(::Type{Rule{T}}, rule::Rule{S}) where {T,S<:Real}
    return Rule(T.(rule.x), T.(rule.w), T(rule.a), T(rule.b))
end