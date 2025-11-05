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
mutable struct LogisticKernel <: AbstractKernel
    ptr::Ptr{spir_kernel}
    Λ::Float64

    function LogisticKernel(Λ::Real)
        Λ ≥ 0 || throw(DomainError(Λ, "Kernel cutoff Λ must be non-negative"))
        status = Ref{Cint}(-100)
        ptr = spir_logistic_kernel_new(Float64(Λ), status)
        status[] == 0 || error("Failed to create logistic kernel")
        kernel = new(ptr, Float64(Λ))
        finalizer(k -> spir_kernel_release(k.ptr), kernel)
        return kernel
    end
end

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
mutable struct RegularizedBoseKernel <: AbstractKernel
    ptr::Ptr{spir_kernel}
    Λ::Float64

    function RegularizedBoseKernel(Λ::Real)
        Λ ≥ 0 || throw(DomainError(Λ, "Kernel cutoff Λ must be non-negative"))
        status = Ref{Cint}(-100)
        ptr = spir_reg_bose_kernel_new(Float64(Λ), status)
        status[] != 0 && error("Failed to create regularized Bose kernel")
        kernel = new(ptr, Float64(Λ))
        finalizer(k -> spir_kernel_release(k.ptr), kernel)
        return kernel
    end
end

Λ(kernel::LogisticKernel) = kernel.Λ
Λ(kernel::RegularizedBoseKernel) = kernel.Λ

iscentrosymmetric(::LogisticKernel) = true
iscentrosymmetric(::RegularizedBoseKernel) = true

function xrange(kernel::LogisticKernel)
    xmin = Ref{Float64}(0.0)
    xmax = Ref{Float64}(0.0)
    ymin = Ref{Float64}(0.0)
    ymax = Ref{Float64}(0.0)
    status = spir_kernel_domain(_get_ptr(kernel), xmin, xmax, ymin, ymax)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get kernel domain")
    return (xmin[], xmax[])
end

function yrange(kernel::LogisticKernel)
    xmin = Ref{Float64}(0.0)
    xmax = Ref{Float64}(0.0)
    ymin = Ref{Float64}(0.0)
    ymax = Ref{Float64}(0.0)
    status = spir_kernel_domain(_get_ptr(kernel), xmin, xmax, ymin, ymax)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get kernel domain")
    return (ymin[], ymax[])
end

function xrange(kernel::RegularizedBoseKernel)
    xmin = Ref{Float64}(0.0)
    xmax = Ref{Float64}(0.0)
    ymin = Ref{Float64}(0.0)
    ymax = Ref{Float64}(0.0)
    status = spir_kernel_domain(_get_ptr(kernel), xmin, xmax, ymin, ymax)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get kernel domain")
    return (xmin[], xmax[])
end

function yrange(kernel::RegularizedBoseKernel)
    xmin = Ref{Float64}(0.0)
    xmax = Ref{Float64}(0.0)
    ymin = Ref{Float64}(0.0)
    ymax = Ref{Float64}(0.0)
    status = spir_kernel_domain(_get_ptr(kernel), xmin, xmax, ymin, ymax)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get kernel domain")
    return (ymin[], ymax[])
end

"""
    weight_func(kernel::LogisticKernel, statistics::Statistics)

Return the weight function for LogisticKernel.

For fermionic statistics, returns the identity function (weight = 1).
For bosonic statistics, returns w(y) = 1 / tanh(Λ y / 2) where y = βω/Λ.
"""
function weight_func(kernel::LogisticKernel, statistics::Statistics)
    if statistics == Fermionic()
        return y -> ones(eltype(y), size(y))
    else  # Bosonic
        return y -> 1 ./ tanh.(0.5 * kernel.Λ * y)
    end
end

"""
    weight_func(kernel::RegularizedBoseKernel, statistics::Statistics)

Return the weight function for RegularizedBoseKernel.

Only supports bosonic statistics. Returns w(y) = 1 / y where y = βω/Λ.
"""
function weight_func(kernel::RegularizedBoseKernel, statistics::Statistics)
    if statistics == Fermionic()
        error("RegularizedBoseKernel does not support fermionic functions")
    else  # Bosonic
        return y -> 1 ./ y
    end
end

"""
    (kernel::LogisticKernel)(x, y)

Evaluate the LogisticKernel at point `(x, y)`.

In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the kernel is:
K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
"""
function (kernel::LogisticKernel)(x::Real, y::Real)
    # Check domain
    xmin, xmax = xrange(kernel)
    ymin, ymax = yrange(kernel)
    (xmin ≤ x ≤ xmax) || throw(DomainError(x, "x value not in range [$xmin, $xmax]"))
    (ymin ≤ y ≤ ymax) || throw(DomainError(y, "y value not in range [$ymin, $ymax]"))
    
    # Compute u_± = (1 ± x)/2 and v = Λ * y
    x₊ = 1 + x
    x₋ = 1 - x
    u₊ = x₊ / 2
    u₋ = x₋ / 2
    v = kernel.Λ * y
    
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

"""
    (kernel::RegularizedBoseKernel)(x, y)

Evaluate the RegularizedBoseKernel at point `(x, y)`.

In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the kernel is:
K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)
"""
function (kernel::RegularizedBoseKernel)(x::Real, y::Real)
    # Check domain
    xmin, xmax = xrange(kernel)
    ymin, ymax = yrange(kernel)
    (xmin ≤ x ≤ xmax) || throw(DomainError(x, "x value not in range [$xmin, $xmax]"))
    (ymin ≤ y ≤ ymax) || throw(DomainError(y, "y value not in range [$ymin, $ymax]"))
    
    # Compute u_± = (1 ± x)/2 and v = Λ * y
    x₊ = 1 + x
    x₋ = 1 - x
    u₊ = x₊ / 2
    u₋ = x₋ / 2
    v = kernel.Λ * y
    
    # With "reduced variables" u, v we have:
    #
    #   K = -1/lambda * exp(-u_+ * v) * v / (exp(-v) - 1)
    #     = -1/lambda * exp(-u_- * -v) * (-v) / (exp(v) - 1)
    #
    # where we again need to use the upper equation for v ≥ 0 and the
    # lower one for v < 0 to avoid overflow.
    absv = abs(v)
    enum = exp(-absv * (v ≥ 0 ? u₊ : u₋))
    
    # The expression ``v / (exp(v) - 1)`` is tricky to evaluate: firstly,
    # it has a singularity at v=0, which can be cured by treating that case
    # separately.  Secondly, the denominator loses precision around 0 since
    # exp(v) = 1 + v + ..., which can be avoided using expm1(...)
    denom = absv ≥ 1e-200 ? absv / expm1(-absv) : -one(absv)
    return -1 / kernel.Λ * enum * denom
end

# SVE hints for LogisticKernel and RegularizedBoseKernel
# These use the new C-API functions to get hints from the C++ implementation

# Import C_API constants and functions
using .C_API: SPIR_COMPUTATION_SUCCESS, spir_kernel_get_sve_hints_segments_x,
              spir_kernel_get_sve_hints_segments_y, spir_kernel_get_sve_hints_nsvals,
              spir_kernel_get_sve_hints_ngauss

# Internal helper to create SVEHints from C-API
# No caching - each method call queries C-API to ensure consistency with epsilon
struct SVEHintsFromCAPI <: AbstractSVEHints
    kernel::Union{LogisticKernel, RegularizedBoseKernel}
    epsilon::Float64
end

function segments_x(hints::SVEHintsFromCAPI, ::Type{T}=Float64) where {T}
    # Get segments_x from C-API
    n_segments_x = Ref{Cint}(0)
    status = spir_kernel_get_sve_hints_segments_x(_get_ptr(hints.kernel), hints.epsilon, C_NULL, n_segments_x)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get segments_x size: status=$status")
    segments_x = Vector{Float64}(undef, Int(n_segments_x[]))
    status = spir_kernel_get_sve_hints_segments_x(_get_ptr(hints.kernel), hints.epsilon, segments_x, n_segments_x)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get segments_x: status=$status")
    return T.(segments_x)
end

function segments_y(hints::SVEHintsFromCAPI, ::Type{T}=Float64) where {T}
    # Get segments_y from C-API
    n_segments_y = Ref{Cint}(0)
    status = spir_kernel_get_sve_hints_segments_y(_get_ptr(hints.kernel), hints.epsilon, C_NULL, n_segments_y)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get segments_y size: status=$status")
    segments_y = Vector{Float64}(undef, Int(n_segments_y[]))
    status = spir_kernel_get_sve_hints_segments_y(_get_ptr(hints.kernel), hints.epsilon, segments_y, n_segments_y)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get segments_y: status=$status")
    return T.(segments_y)
end

function nsvals(hints::SVEHintsFromCAPI)
    # Get nsvals from C-API
    nsvals = Ref{Cint}(0)
    status = spir_kernel_get_sve_hints_nsvals(_get_ptr(hints.kernel), hints.epsilon, nsvals)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get nsvals: status=$status")
    return Int(nsvals[])
end

function ngauss(hints::SVEHintsFromCAPI)
    # Get ngauss from C-API
    ngauss = Ref{Cint}(0)
    status = spir_kernel_get_sve_hints_ngauss(_get_ptr(hints.kernel), hints.epsilon, ngauss)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get ngauss: status=$status")
    return Int(ngauss[])
end

"""
    sve_hints(kernel::LogisticKernel, epsilon::Real)

Return discretization hints for singular value expansion of a LogisticKernel.
Uses the C++ implementation via C-API.
"""
function sve_hints(kernel::LogisticKernel, epsilon::Real)
    return SVEHintsFromCAPI(kernel, epsilon)
end

"""
    sve_hints(kernel::RegularizedBoseKernel, epsilon::Real)

Return discretization hints for singular value expansion of a RegularizedBoseKernel.
Uses the C++ implementation via C-API.
"""
function sve_hints(kernel::RegularizedBoseKernel, epsilon::Real)
    return SVEHintsFromCAPI(kernel, epsilon)
end
