export LogisticKernel, RegularizedBoseKernel, sve_hints, segments_x, segments_y,
       matrix_from_gauss, get_symmetrized, nsvals, ngauss, ypower, conv_radius, weight_func

@doc raw"""
    AbstractKernel

Integral kernel `K(x, y)`.

Abstract base type for an integral kernel, i.e. a AbstractFloat binary function
``K(x, y)`` used in a Fredhold integral equation of the first kind:
```math
    u(x) = ∫ K(x, y) v(y) dy
```
where ``x ∈ [x_\mathrm{min}, x_\mathrm{max}]`` and 
``y ∈ [y_\mathrm{min}, y_\mathrm{max}]``.  For its SVE to exist,
the kernel must be square-integrable, for its singular values to decay
exponentially, it must be smooth.

In general, the kernel is applied to a scaled spectral function ``ρ'(y)`` as:
```math
    ∫ K(x, y) ρ'(y) dy,
```
where ``ρ'(y) = w(y) ρ(y)``.
"""
abstract type AbstractKernel end

@doc raw"""
    LogisticKernel{T} <: AbstractKernel

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
struct LogisticKernel{T<:AbstractFloat} <: AbstractKernel
    Λ::T
end

@doc raw"""
    RegularizedBoseKernel{T} <: AbstractKernel

Regularized bosonic analytical continuation kernel.

In dimensionless variables ``x = 2 τ/β - 1``, ``y = β ω/Λ``, the fermionic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = y \frac{e^{-Λ y (x + 1) / 2}}{e^{-Λ y} - 1}
```
Care has to be taken in evaluating this expression around ``y = 0``.
"""
struct RegularizedBoseKernel{T<:AbstractFloat} <: AbstractKernel
    Λ::T
end

"""
    AbstractSVEHints

Discretization hints for singular value expansion of a given kernel.
"""
abstract type AbstractSVEHints end

struct SVEHintsLogistic{T,S} <: AbstractSVEHints where {T,S<:AbstractFloat}
    kernel::LogisticKernel{T}
    ε::S
end

struct SVEHintsRegularizedBose{T,S} <: AbstractSVEHints where {T,S<:AbstractFloat}
    kernel::RegularizedBoseKernel{T}
    ε::S
end

struct SVEHintsReduced{T} <: AbstractSVEHints where {T<:AbstractSVEHints}
    inner_hints::T
end

abstract type AbstractReducedKernel <: AbstractKernel end

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
This kernel is what this class represents.  The full singular functions can
be reconstructed by (anti-)symmetrically continuing them to the negative
axis.
"""
struct ReducedKernel{K<:AbstractKernel} <: AbstractReducedKernel
    inner::K
    sign::Int
end

@doc raw"""
    LogisticKernelOdd{T} <: AbstractReducedKernel

Fermionic analytical continuation kernel, odd.

In dimensionless variables ``x = 2τ/β - 1``, ``y = βω/Λ``, the fermionic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = -\frac{\sinh(Λ x y / 2)}{\cosh(Λ y / 2)}
```
"""
struct LogisticKernelOdd{T<:AbstractFloat} <: AbstractReducedKernel
    inner::LogisticKernel{T}
    sign::Int

    function LogisticKernelOdd(inner::LogisticKernel{T}, sign) where {T<:AbstractFloat}
        iscentrosymmetric(inner) || error("inner kernel must be centrosymmetric")
        abs(sign) == 1 || error("sign must be -1 or 1")
        return new{T}(inner, sign)
    end
end

@doc raw"""
    RegularizedBoseKernelOdd{T} <: AbstractReducedKernel

Bosonic analytical continuation kernel, odd.

In dimensionless variables ``x = 2 τ / β - 1``, ``y = β ω / Λ``, the fermionic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = -y \frac{\sinh(Λ x y / 2)}{\sinh(Λ y / 2)}
```
"""
struct RegularizedBoseKernelOdd{T} <: AbstractReducedKernel where {T<:AbstractFloat}
    inner::RegularizedBoseKernel{T}
    sign::Int

    function RegularizedBoseKernelOdd(inner::RegularizedBoseKernel{T},
                                      sign) where {T<:AbstractFloat}
        iscentrosymmetric(inner) || error("inner kernel must be centrosymmetric")
        abs(sign) == 1 || error("sign must be -1 or 1")
        return new{T}(inner, sign)
    end
end

@doc raw"""
    xrange(kernel)

Return a tuple ``(x_\mathrm{min}, x_\mathrm{max})`` delimiting the range 
of allowed `x` values.
"""
xrange(::AbstractKernel) = (-1, 1)
xrange(kernel::AbstractReducedKernel) = (0, last(xrange(kernel.inner)))

@doc raw"""
    yrange(kernel)

Return a tuple ``(y_\mathrm{min}, y_\mathrm{max})`` delimiting the range
 of allowed `y` values.
"""
yrange(::AbstractKernel) = (-1, 1)
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

    enum = exp(-abs(v) * (v ≥ 0 ? u₊ : u₋))
    denom = 1 + exp(-abs(v))
    return enum / denom
end

function compute(kernel::RegularizedBoseKernel, u₊, u₋, v)
    # With "reduced variables" u, v we have:
    #
    #   K = -1/lambda * exp(-u_+ * v) * v / (exp(-v) - 1)
    #     = -1/lambda * exp(-u_- * -v) * (-v) / (exp(v) - 1)
    #
    # where we again need to use the upper equation for v ≥ 0 and the
    # lower one for v < 0 to avoid overflow.
    absv = abs(v)
    enum = exp(-absv * (v ≥ 0 ? u₊ : u₋))
    T = eltype(v)

    # The expression ``v / (exp(v) - 1)`` is tricky to evaluate: firstly,
    # it has a singularity at v=0, which can be cured by treating that case
    # separately.  Secondly, the denominator loses precision around 0 since
    # exp(v) = 1 + v + ..., which can be avoided using expm1(...)
    denom = absv ≥ 1e-200 ? absv / expm1(-absv) : one(absv)
    return -1 / T(kernel.Λ) * enum * denom
end

"""
    segments_x(kernel)

Segments for piecewise polynomials on the ``x`` axis.

List of segments on the ``x`` axis for the associated piecewise
polynomial. Should reflect the approximate position of roots of a
high-order singular function in ``x``.
"""
function segments_x end

"""
    segments_y(kernel)

Segments for piecewise polynomials on the ``y`` axis.

List of segments on the ``y`` axis for the associated piecewise
polynomial. Should reflect the approximate position of roots of a
high-order singular function in ``y``.
"""
function segments_y end

function segments_x(hints::SVEHintsLogistic)
    nzeros = max(round(Int, 15 * log10(hints.kernel.Λ)), 1)
    diffs = 1 ./ cosh.(0.143 * range(0; length=nzeros))
    cumsum!(diffs, diffs; dims=1) # From here on, `diffs` contains the zeros
    diffs ./= last(diffs)
    return [-reverse(diffs); 0; diffs]
end

function segments_y(hints::SVEHintsLogistic)
    nzeros = max(round(Int, 20 * log10(hints.kernel.Λ)), 2)

    # Zeros around -1 and 1 are distributed asymptotically identically
    leading_diffs = [0.01523, 0.03314, 0.04848, 0.05987, 0.06703, 0.07028, 0.07030, 0.06791,
                     0.06391, 0.05896, 0.05358, 0.04814, 0.04288, 0.03795, 0.03342, 0.02932,
                     0.02565, 0.02239, 0.01951, 0.01699][begin:min(nzeros, 20)]

    diffs = [leading_diffs; 0.25 ./ exp.(0.141 * (20:(nzeros - 1)))]

    cumsum!(diffs, diffs; dims=1) # From here on, `diffs` contains the zeros
    diffs ./= pop!(diffs)
    diffs .-= 1
    return [-1; diffs; 0; -reverse(diffs); 1]
end

function segments_x(hints::SVEHintsRegularizedBose)
    # Somewhat less accurate...
    nzeros = max(round(Int, 15 * log10(hints.kernel.Λ)), 15)
    diffs = 1 ./ cosh.(0.18 * range(0; length=nzeros))
    cumsum!(diffs, diffs; dims=1) # From here on, `diffs` contains the zeros
    diffs ./= last(diffs)
    return [-reverse(diffs); 0; diffs]
end

function segments_y(hints::SVEHintsRegularizedBose)
    nzeros = max(round(Int, 20 * log10(hints.kernel.Λ)), 20)
    i = range(0; length=nzeros)
    diffs = @. 0.12 / exp(0.0337 * i * log(i + 1))
    cumsum!(diffs, diffs; dims=1) # From here on, `diffs` contains the zeros
    diffs ./= pop!(diffs)
    diffs .-= 1
    return [-1; diffs; 0; -reverse(diffs); 1]
end

"""
    matrix_from_gauss(kernel, gauss_x, gauss_y)

Compute matrix for kernel from Gauss rules.
"""
function matrix_from_gauss(kernel, gauss_x, gauss_y)
    # (1 ± x) is problematic around x = -1 and x = 1, where the quadrature
    # nodes are clustered most tightly.  Thus we have the need for the
    # matrix method.
    return kernel.(gauss_x.x, permutedims(gauss_y.x), gauss_x.x .- gauss_x.a,
                   gauss_x.b .- gauss_x.x)
end

"""
    check_domain(kernel, x, y)

Check that `(x, y)` lies within `kernel`'s domain and return it.
"""
function check_domain(kernel, x, y)
    xmin, xmax = xrange(kernel)
    xmin ≤ x ≤ xmax || throw(DomainError("x value not in range [$xmin, $xmax]"))

    ymin, ymax = yrange(kernel)
    ymin ≤ y ≤ ymax || throw(DomainError("y value not in range [$ymin, $ymax]"))

    return x, y
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

    By default, this returns a simple wrapper over the current instance
    which naively performs the sum.  You may want to override this
    to avoid cancellation.
"""
get_symmetrized(kernel::AbstractKernel, sign) = ReducedKernel(kernel, sign)

function get_symmetrized(kernel::LogisticKernel, sign)
    sign == -1 && return LogisticKernelOdd(kernel, sign)
    return Base.@invoke get_symmetrized(kernel::AbstractKernel, sign)
end

function get_symmetrized(kernel::RegularizedBoseKernel, sign)
    sign == -1 && return RegularizedBoseKernelOdd(kernel, sign)
    return Base.@invoke get_symmetrized(kernel::AbstractKernel, sign)
end

get_symmetrized(::AbstractReducedKernel, sign) = error("cannot symmetrize twice")

function callreduced(kernel::AbstractReducedKernel, x, y, x₊, x₋)
    x, y = check_domain(kernel, x, y)

    # The reduced kernel is defined only over the interval [0, 1], which
    # means we must add one to get the x_plus for the inner kernels.  We
    # can compute this as 1 + x, since we are away from -1.
    x₊ = 1 + x₊

    K₊ = kernel.inner(x, +y, x₊, x₋)
    K₋ = kernel.inner(x, -y, x₊, x₋)
    return K₊ + kernel.sign * K₋
end

(kernel::ReducedKernel)(x, y, x₊, x₋) = callreduced(kernel, x, y, x₊, x₋)

"""
    is_centrosymmetric(kernel)

Return `true` if `kernel(x, y) == kernel(-x, -y)` for all values of `x` and `y` 
in range. This allows the kernel to be block-diagonalized,
speeding up the singular value expansion by a factor of 4.  Defaults
to `false`.
"""
iscentrosymmetric(::AbstractKernel) = false
iscentrosymmetric(::LogisticKernel) = true
iscentrosymmetric(::RegularizedBoseKernel) = true
iscentrosymmetric(::AbstractReducedKernel) = false

"""
    kernel(x, y[, x₊, x₋])

Evaluate `kernel::AbstractKernel` at point (`x`, `y`).

The parameters `x₊` and `x₋`, if given, shall contain the
values of `x - xₘᵢₙ` and `xₘₐₓ - x`, respectively.  This is useful
if either difference is to be formed and cancellation expected.
"""
function (kernel::AbstractKernel)(x, y,
                                  x₊=x - first(xrange(kernel)),
                                  x₋=last(xrange(kernel)) - x)
    x, y = check_domain(kernel, x, y)
    u₊, u₋, v = compute_uv(kernel.Λ, x, y, x₊, x₋)
    return compute(kernel, u₊, u₋, v)
end

function (kernel::LogisticKernelOdd)(x, y,
                                     x₊=x - first(xrange(kernel)),
                                     x₋=last(xrange(kernel)) - x)
    result = callreduced(kernel, x, y, x₊, x₋)

    # For x * y around 0, antisymmetrization introduces cancellation, which
    # reduces the relative precision. To combat this, we replace the
    # values with the explicit form
    v_half = kernel.inner.Λ / 2 * y
    xy_small = x * v_half < 1
    cosh_finite = v_half < 85
    return xy_small && cosh_finite ? -sinh(v_half * x) / cosh(v_half) : result
end

function (kernel::RegularizedBoseKernelOdd)(x, y,
                                            x₊=x - first(xrange(kernel)),
                                            x₋=last(xrange(kernel)) - x)
    result = callreduced(kernel, x, y, x₊, x₋)

    # For x * y around 0, antisymmetrization introduces cancellation, which
    # reduces the relative precision.  To combat this, we replace the
    # values with the explicit form
    v_half = kernel.inner.Λ / 2 * y
    xv_half = x * v_half
    xy_small = xv_half < 1
    sinh_range = 1e-200 < v_half < 85
    return xy_small && sinh_range ? -y * sinh(xv_half) / sinh(v_half) : result
end

segments_x(hints::SVEHintsReduced) = symm_segments(segments_x(hints.inner_hints))
segments_y(hints::SVEHintsReduced) = symm_segments(segments_y(hints.inner_hints))

function symm_segments(x)
    x ≈ -reverse(x) || error("segments must be symmetric")
    xpos = x[(begin + length(x) ÷ 2):end]
    iszero(first(xpos)) || (xpos = [0; xpos])
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

sve_hints(kernel::LogisticKernel, ε) = SVEHintsLogistic(kernel, ε)
sve_hints(kernel::RegularizedBoseKernel, ε) = SVEHintsRegularizedBose(kernel, ε)
sve_hints(kernel::AbstractReducedKernel, ε) = SVEHintsReduced(sve_hints(kernel.inner, ε))

"""
    nsvals(hints)

Upper bound for number of singular values.

Upper bound on the number of singular values above the given threshold, i.e. where
`s[l] ≥ ε * first(s)`.
"""
function nsvals end
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
ngauss(hints::SVEHintsLogistic) = hints.ε ≥ 1e-8 ? 10 : 16
ngauss(hints::SVEHintsRegularizedBose) = hints.ε ≥ 1e-8 ? 10 : 16
ngauss(hints::SVEHintsReduced) = ngauss(hints.inner_hints)

"""
    ypower(kernel)

Power with which the ``y`` coordinate scales.
"""
ypower(::AbstractKernel) = 0
ypower(::RegularizedBoseKernel) = 1
ypower(kernel::AbstractReducedKernel) = ypower(kernel.inner)

"""
    conv_radius(kernel)

Convergence radius of the Matsubara basis asymptotic model.

For improved relative numerical accuracy, the IR basis functions on the
Matsubara axis `uhat(basis, n)` can be evaluated from an asymptotic
expression for `abs(n) > conv_radius`.  If `isnothing(conv_radius)`, then 
the asymptotics are unused (the default).
"""
conv_radius(::AbstractKernel) = nothing
conv_radius(kernel::LogisticKernel) = 40 * kernel.Λ
conv_radius(kernel::RegularizedBoseKernel) = 40 * kernel.Λ
conv_radius(kernel::AbstractReducedKernel) = conv_radius(kernel.inner)

"""
    weight_func(kernel, statistics::Statistics)

Return the weight function for the given statistics.
"""
function weight_func(::AbstractKernel, statistics)
    statistics ∈ (fermion, boson) ||
        error("statistics must be fermion for fermions or boson for bosons")
    return x -> ones(eltype(x), size(x))
end
function weight_func(kernel::LogisticKernel, statistics)
    statistics ∈ (fermion, boson) ||
        error("statistics must be fermion for fermions or boson for bosons")
    if statistics == fermion
        return y -> ones(eltype(y), size(y))
    else
        return y -> 1 / tanh(0.5 * kernel.Λ * y)
    end
end
function weight_func(::RegularizedBoseKernel, statistics)
    statistics == boson || error("Kernel is designed for bosonic functions")
    return y -> 1 / y
end