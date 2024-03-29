@doc raw"""
    LogisticKernel <: AbstractKernel

Fermionic/bosonic analytical continuation kernel.

In dimensionless variables ``x = 2 τ/β - 1``, ``y = β ω/Λ``,
the integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = \frac{e^{-Λ y (x + 1) / 2}}{1 + e^{-Λ y}}
```
LogisticKernel is a fermionic analytic continuation kernel.
Nevertheless, one can model the ``τ`` dependence of
a bosonic correlation function as follows:
```math
    ∫ \frac{e^{-Λ y (x + 1) / 2}}{1 - e^{-Λ y}} ρ(y) dy = ∫ K(x, y) ρ'(y) dy,
```
with
```math
    ρ'(y) = w(y) ρ(y),
```
where the weight function is given by
```math
    w(y) = \frac{1}{\tanh(Λ y/2)}.
```
"""
struct LogisticKernel <: AbstractKernel
    Λ::Float64
    function LogisticKernel(Λ)
        Λ ≥ 0 || throw(DomainError(Λ, "Kernel cutoff Λ must be non-negative"))
        return new(Λ)
    end
end

Λ(kernel::LogisticKernel) = kernel.Λ

@doc raw"""
    RegularizedBoseKernel <: AbstractKernel

Regularized bosonic analytical continuation kernel.

In dimensionless variables ``x = 2 τ/β - 1``, ``y = β ω/Λ``, the fermionic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = y \frac{e^{-Λ y (x + 1) / 2}}{e^{-Λ y} - 1}
```
Care has to be taken in evaluating this expression around ``y = 0``.
"""
struct RegularizedBoseKernel <: AbstractKernel
    Λ::Float64
    function RegularizedBoseKernel(Λ)
        Λ ≥ 0 || throw(DomainError(Λ, "Kernel cutoff Λ must be non-negative"))
        return new(Λ)
    end
end

Λ(kernel::RegularizedBoseKernel) = kernel.Λ

struct SVEHintsLogistic{T} <: AbstractSVEHints
    kernel :: LogisticKernel
    ε      :: T
end

struct SVEHintsRegularizedBose{T} <: AbstractSVEHints
    kernel :: RegularizedBoseKernel
    ε      :: T
end

struct SVEHintsReduced{T<:AbstractSVEHints} <: AbstractSVEHints
    inner_hints::T
end

@doc raw"""
    ReducedKernel

Restriction of centrosymmetric kernel to positive interval.

For a kernel ``K`` on ``[-1, 1] × [-1, 1]`` that is centrosymmetric, i.e.
``K(x, y) = K(-x, -y)``, it is straight-forward to show that the left/right
singular vectors can be chosen as either odd or even functions.

Consequentially, they are singular functions of a reduced kernel ``K_\mathrm{red}``
on ``[0, 1] × [0, 1]`` that is given as either:
```math
    K_\mathrm{red}(x, y) = K(x, y) \pm K(x, -y)
```
This kernel is what this type represents. The full singular functions can be
reconstructed by (anti-)symmetrically continuing them to the negative axis.
"""
struct ReducedKernel{K<:AbstractKernel} <: AbstractReducedKernel
    inner :: K
    sign  :: Int
end

@doc raw"""
    LogisticKernelOdd <: AbstractReducedKernel

Fermionic analytical continuation kernel, odd.

In dimensionless variables ``x = 2τ/β - 1``, ``y = βω/Λ``, the fermionic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = -\frac{\sinh(Λ x y / 2)}{\cosh(Λ y / 2)}
```
"""
struct LogisticKernelOdd <: AbstractReducedKernel
    inner :: LogisticKernel
    sign  :: Int

    function LogisticKernelOdd(inner::LogisticKernel, sign)
        iscentrosymmetric(inner) || error("inner kernel must be centrosymmetric")
        abs(sign) == 1 || throw(DomainError(sign, "sign must be -1 or 1"))
        return new(inner, sign)
    end
end

@doc raw"""
    RegularizedBoseKernelOdd <: AbstractReducedKernel

Bosonic analytical continuation kernel, odd.

In dimensionless variables ``x = 2 τ / β - 1``, ``y = β ω / Λ``, the fermionic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = -y \frac{\sinh(Λ x y / 2)}{\sinh(Λ y / 2)}
```
"""
struct RegularizedBoseKernelOdd <: AbstractReducedKernel
    inner :: RegularizedBoseKernel
    sign  :: Int

    function RegularizedBoseKernelOdd(inner::RegularizedBoseKernel, sign)
        iscentrosymmetric(inner) || error("inner kernel must be centrosymmetric")
        isone(abs(sign)) || throw(DomainError(sign, "sign must be -1 or 1"))
        return new(inner, sign)
    end
end

@doc raw"""
    xrange(kernel)

Return a tuple ``(x_\mathrm{min}, x_\mathrm{max})`` delimiting the range of allowed `x`
values.
"""
function xrange end
xrange(::AbstractKernel)              = (-1, 1)
xrange(kernel::AbstractReducedKernel) = (0, last(xrange(kernel.inner)))

@doc raw"""
    yrange(kernel)

Return a tuple ``(y_\mathrm{min}, y_\mathrm{max})`` delimiting the range of allowed `y`
values.
"""
function yrange end
yrange(::AbstractKernel)              = (-1, 1)
yrange(kernel::AbstractReducedKernel) = (0, last(yrange(kernel.inner)))

function compute(::LogisticKernel, u₊, u₋, v)
    # By introducing u_± = (1 ± x)/2 and v = Λ * y, we can write
    # the kernel in the following two ways:
    #
    #    k = exp(-u₊ * v) / (exp(-v) + 1)
    #      = exp(-u₋ * -v) / (exp(v) + 1)
    #
    # We need to use the upper equation for v ≥ 0 and the lower one for
    # v < 0 to avoid overflowing both numerator and denominator

    mabsv = -abs(v)
    enum = exp(mabsv * ifelse(v ≥ 0, u₊, u₋))
    denom = 1 + exp(mabsv)
    return enum / denom
end

function compute(kernel::RegularizedBoseKernel, u₊, u₋, v::T) where {T}
    # With "reduced variables" u, v we have:
    #
    #   K = -1/Λ * exp(-u_+ * v) * v / (exp(-v) - 1)
    #     = -1/Λ * exp(-u_- * -v) * (-v) / (exp(v) - 1)
    #
    # where we again need to use the upper equation for v ≥ 0 and the
    # lower one for v < 0 to avoid overflow.
    absv = abs(v)
    enum = exp(-absv * (v ≥ 0 ? u₊ : u₋))

    # The expression ``v / (exp(v) - 1)`` is tricky to evaluate: firstly,
    # it has a singularity at v=0, which can be cured by treating that case
    # separately. Secondly, the denominator loses precision around 0 since
    # exp(v) = 1 + v + ..., which can be avoided using expm1(...)
    denom = absv ≥ 1e-200 ? absv / expm1(-absv) : -one(absv)
    return -1 / T(kernel.Λ) * enum * denom
end

"""
    segments_x(sve_hints::AbstractSVEHints[, T])

Segments for piecewise polynomials on the ``x`` axis.

List of segments on the ``x`` axis for the associated piecewise polynomial. Should reflect
the approximate position of roots of a high-order singular function in ``x``.
"""
function segments_x(hints::SVEHintsLogistic, ::Type{T}=Float64) where {T}
    nzeros = max(round(Int, 15 * log10(hints.kernel.Λ)), 1)
    temp = T(0.143) * range(0; length=nzeros)
    diffs = @. inv(cosh(temp))
    zeros = cumsum(diffs)
    zeros ./= last(zeros)
    return T[-reverse(zeros); zero(T); zeros]
end

"""
    segments_y(sve_hints::AbstractSVEHints[, T])

Segments for piecewise polynomials on the ``y`` axis.

List of segments on the ``y`` axis for the associated piecewise polynomial. Should reflect
the approximate position of roots of a high-order singular function in ``y``.
"""
function segments_y(hints::SVEHintsLogistic, ::Type{T}=Float64) where {T}
    nzeros = max(round(Int, 20 * log10(hints.kernel.Λ)), 2)

    # Zeros around -1 and 1 are distributed asymptotically identically
    diffs = T[0.01523, 0.03314, 0.04848, 0.05987, 0.06703, 0.07028, 0.07030,
              0.06791, 0.06391, 0.05896, 0.05358, 0.04814, 0.04288, 0.03795,
              0.03342, 0.02932, 0.02565, 0.02239, 0.01951, 0.01699][1:min(nzeros, 20)]

    temp = T(0.141) * (20:(nzeros - 1))
    trailing_diffs = [0.25 * exp(-x) for x in temp]
    append!(diffs, trailing_diffs)
    zeros = cumsum(diffs)
    zeros ./= pop!(zeros)
    zeros .-= one(T)
    return T[-one(T); zeros; zero(T); -reverse(zeros); one(T)]
end

function segments_x(hints::SVEHintsRegularizedBose, ::Type{T}=Float64) where {T}
    # Somewhat less accurate...
    nzeros = max(round(Int, 15 * log10(hints.kernel.Λ)), 15)
    temp = T(0.18) * range(0; length=nzeros)
    diffs = @. inv(cosh(temp))
    zeros = cumsum(diffs)
    zeros ./= last(zeros)
    return T[-reverse(zeros); zero(T); zeros]
end

function segments_y(hints::SVEHintsRegularizedBose, ::Type{T}=Float64) where {T}
    nzeros = max(round(Int, 20 * log10(hints.kernel.Λ)), 20)
    i = range(0; length=nzeros)
    diffs = @. T(0.12 / exp(0.0337 * i * log(i + 1)))
    zeros = cumsum(diffs)
    zeros ./= pop!(zeros)
    zeros .-= one(T)
    return T[-one(T); zeros; zero(T); -reverse(zeros); one(T)]
end

"""
    matrix_from_gauss(kernel, gauss_x, gauss_y)

Compute matrix for kernel from Gauss rules.
"""
function matrix_from_gauss(kernel, gauss_x::Rule{T}, gauss_y::Rule{T})::Matrix{T} where {T}
    # (1 ± x) is problematic around x = -1 and x = 1, where the quadrature
    # nodes are clustered most tightly. Thus we have the need for the
    # matrix method.
    n = length(gauss_x.x)
    m = length(gauss_y.x)
    res = Matrix{T}(undef, n, m)
    Threads.@threads for i in eachindex(gauss_x.x)
        @inbounds @simd for j in eachindex(gauss_y.x)
            res[i, j] = kernel(gauss_x.x[i], gauss_y.x[j],
                               gauss_x.x_forward[i], gauss_x.x_backward[i])
        end
    end
    res
end

function Base.checkbounds(::Type{Bool}, kernel::AbstractKernel, x::Real, y::Real)
    xmin, xmax = xrange(kernel)
    ymin, ymax = yrange(kernel)
    (xmin ≤ x ≤ xmax) && (ymin ≤ y ≤ ymax)
end

function Base.checkbounds(kernel::AbstractKernel, x::Real, y::Real)
    checkbounds(Bool, kernel, x, y) || throw(BoundsError(kernel, (x, y)))
end

function compute_uv(Λ, x, y, x₊=1 + x, x₋=1 - x)
    u₊ = x₊ / 2
    u₋ = x₋ / 2
    v = Λ * y
    return u₊, u₋, v
end

"""
    get_symmetrized(kernel, sign)

Construct a symmetrized version of `kernel`, i.e. `kernel(x, y) + sign * kernel(x, -y)`.

!!! warning "Beware!"

    By default, this returns a simple wrapper over the current instance which naively
    performs the sum. You may want to override this to avoid cancellation.
"""
get_symmetrized(kernel::AbstractKernel, sign) = ReducedKernel(kernel, sign)

function get_symmetrized(kernel::LogisticKernel, sign)
    sign == -1 && return LogisticKernelOdd(kernel, sign)
    return Base.invoke(get_symmetrized, Tuple{AbstractKernel,typeof(sign)}, kernel, sign)
end

function get_symmetrized(kernel::RegularizedBoseKernel, sign)
    sign == -1 && return RegularizedBoseKernelOdd(kernel, sign)
    return Base.invoke(get_symmetrized, Tuple{AbstractKernel,typeof(sign)}, kernel, sign)
end

get_symmetrized(::AbstractReducedKernel, sign) = error("cannot symmetrize twice")

function callreduced(kernel::AbstractReducedKernel, x, y, x₊, x₋)
    @boundscheck checkbounds(kernel, x, y)

    # The reduced kernel is defined only over the interval [0, 1], which
    # means we must add one to get the x_plus for the inner kernels. We
    # can compute this as 1 + x, since we are away from -1.
    x₊ = 1 + x₊

    K₊ = kernel.inner(x, +y, x₊, x₋)
    K₋ = kernel.inner(x, -y, x₊, x₋)
    return K₊ + kernel.sign * K₋
end

(kernel::ReducedKernel)(x, y, x₊, x₋) = callreduced(kernel, x, y, x₊, x₋)

"""
    iscentrosymmetric(kernel)

Return `true` if `kernel(x, y) == kernel(-x, -y)` for all values of `x` and `y`
in range. This allows the kernel to be block-diagonalized, speeding up the singular
value expansion by a factor of 4. Defaults to `false`.
"""
function iscentrosymmetric end
iscentrosymmetric(::LogisticKernel)        = true
iscentrosymmetric(::RegularizedBoseKernel) = true
iscentrosymmetric(::AbstractReducedKernel) = false

"""
    (kernel::AbstractKernel)(x, y[, x₊, x₋])

Evaluate `kernel` at point `(x, y)`.

The parameters `x₊` and `x₋`, if given, shall contain the values of `x - xₘᵢₙ` and
`xₘₐₓ - x`, respectively. This is useful if either difference is to be formed and
cancellation expected.
"""
function (kernel::AbstractKernel)(x, y,
                                  x₊=x - first(xrange(kernel)),
                                  x₋=last(xrange(kernel)) - x)
    @boundscheck checkbounds(kernel, x, y)
    u₊, u₋, v = compute_uv(kernel.Λ, x, y, x₊, x₋)
    return compute(kernel, u₊, u₋, v)
end

function (kernel::LogisticKernelOdd)(x, y,
                                     x₊=x - first(xrange(kernel)),
                                     x₋=last(xrange(kernel)) - x)
    # For x * y around 0, antisymmetrization introduces cancellation, which
    # reduces the relative precision. To combat this, we replace the
    # values with the explicit form
    v_half = kernel.inner.Λ / 2 * y
    xy_small = x * v_half < 1
    cosh_finite = v_half < 85
    if xy_small && cosh_finite
        return -sinh(v_half * x) / cosh(v_half)
    else
        return callreduced(kernel, x, y, x₊, x₋)
    end
end

function (kernel::RegularizedBoseKernelOdd)(x, y,
                                            x₊=x - first(xrange(kernel)),
                                            x₋=last(xrange(kernel)) - x)
    # For x * y around 0, antisymmetrization introduces cancellation, which
    # reduces the relative precision. To combat this, we replace the
    # values with the explicit form.
    v_half = kernel.inner.Λ / 2 * y
    xv_half = x * v_half
    xy_small = xv_half < 1
    sinh_range = 1e-200 < v_half < 85
    if xy_small && sinh_range
        return -y * sinh(xv_half) / sinh(v_half)
    else
        return callreduced(kernel, x, y, x₊, x₋)
    end
end

function segments_x(hints::SVEHintsReduced, ::Type{T}=Float64) where {T}
    symm_segments(segments_x(hints.inner_hints, T))
end
function segments_y(hints::SVEHintsReduced, ::Type{T}=Float64) where {T}
    symm_segments(segments_y(hints.inner_hints, T))
end

function symm_segments(x::AbstractVector{T}) where {T}
    for (xi, revxi) in zip(x, Iterators.reverse(x))
        xi ≈ -revxi || error("segments must be symmetric")
    end
    xpos = x[(begin + length(x) ÷ 2):end]
    iszero(first(xpos)) || pushfirst!(xpos, zero(T))
    return xpos
end

"""
    sve_hints(kernel, ε)

Provide discretisation hints for the SVE routines.

Advises the SVE routines of discretisation parameters suitable in
tranforming the (infinite) SVE into an (finite) SVD problem.

See also [`AbstractSVEHints`](@ref).
"""
function sve_hints end
sve_hints(kernel::LogisticKernel, ε)        = SVEHintsLogistic(kernel, ε)
sve_hints(kernel::RegularizedBoseKernel, ε) = SVEHintsRegularizedBose(kernel, ε)
sve_hints(kernel::AbstractReducedKernel, ε) = SVEHintsReduced(sve_hints(kernel.inner, ε))

"""
    nsvals(hints)

Upper bound for number of singular values.

Upper bound on the number of singular values above the given threshold, i.e. where
`s[l] ≥ ε * first(s)`.
"""
function nsvals(hints::SVEHintsLogistic)
    log10_Λ = max(1, log10(hints.kernel.Λ))
    return round(Int, (25 + log10_Λ) * log10_Λ)
end
function nsvals(hints::SVEHintsRegularizedBose)
    log10_Λ = max(1, log10(hints.kernel.Λ))
    return round(Int, 28 * log10_Λ)
end
function nsvals(hints::SVEHintsReduced)
    return (nsvals(hints.inner_hints) + 1) ÷ 2
end

"""
    ngauss(hints)

Gauss-Legendre order to use to guarantee accuracy.
"""
function ngauss end
ngauss(hints::SVEHintsLogistic)        = hints.ε ≥ sqrt(eps()) ? 10 : 16
ngauss(hints::SVEHintsRegularizedBose) = hints.ε ≥ sqrt(eps()) ? 10 : 16
ngauss(hints::SVEHintsReduced)         = ngauss(hints.inner_hints)

"""
    ypower(kernel)

Power with which the ``y`` coordinate scales.
"""
function ypower end
ypower(::AbstractKernel)              = 0
ypower(::RegularizedBoseKernel)       = 1
ypower(kernel::AbstractReducedKernel) = ypower(kernel.inner)

"""
    conv_radius(kernel)

Convergence radius of the Matsubara basis asymptotic model.

For improved relative numerical accuracy, the IR basis functions on the
Matsubara axis `uhat(basis, n)` can be evaluated from an asymptotic
expression for `abs(n) > conv_radius`. If `isinf(conv_radius)`, then
the asymptotics are unused (the default).
"""
function conv_radius end
conv_radius(kernel::LogisticKernel)        = 40 * kernel.Λ
conv_radius(kernel::RegularizedBoseKernel) = 40 * kernel.Λ
conv_radius(kernel::AbstractReducedKernel) = conv_radius(kernel.inner)

"""
    weight_func(kernel, statistics::Statistics)

Return the weight function for the given statistics.

  - Fermion: `w(x) == 1`
  - Boson: `w(y) == 1/tanh(Λ*y/2)`
"""
function weight_func end
weight_func(::AbstractKernel, ::Statistics)       = one
weight_func(::LogisticKernel, ::Fermionic)        = one
weight_func(kernel::LogisticKernel, ::Bosonic)    = y -> 1 / tanh(0.5 * kernel.Λ * y)
weight_func(::RegularizedBoseKernel, ::Fermionic) = error("Kernel is designed for bosonic functions")
weight_func(::RegularizedBoseKernel, ::Bosonic)   = inv
