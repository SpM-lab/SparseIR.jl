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
    inv_weight_func(kernel::LogisticKernel, statistics::Statistics, beta::Float64, lambda::Float64)

Return the inverse weight function for LogisticKernel.

For fermionic statistics, returns the identity function (inv_weight = 1).
For bosonic statistics, returns inv_weight(omega) = tanh(Λ * beta * omega / (2 * lambda)).

The function is evaluated as `omega` only, with `beta` and `lambda` captured from the context.
"""
function inv_weight_func(kernel::LogisticKernel, statistics::Statistics, beta::Float64, lambda::Float64)
    if statistics == Fermionic()
        return (omega::Float64) -> 1.0
    else  # Bosonic
        # For bosonic: inv_weight(omega) = tanh(Λ * beta * omega / (2 * lambda))
        return (omega::Float64) -> tanh(0.5 * kernel.Λ * beta * omega / lambda)
    end
end

"""
    inv_weight_func(kernel::RegularizedBoseKernel, statistics::Statistics, beta::Float64, lambda::Float64)

Return the inverse weight function for RegularizedBoseKernel.

Only supports bosonic statistics. Returns inv_weight(omega) = beta * omega / lambda.

The function is evaluated as `omega` only, with `beta` and `lambda` captured from the context.
"""
function inv_weight_func(kernel::RegularizedBoseKernel, statistics::Statistics, beta::Float64, lambda::Float64)
    if statistics == Fermionic()
        error("RegularizedBoseKernel does not support fermionic functions")
    else  # Bosonic
        # For bosonic: inv_weight(omega) = beta * omega / lambda
        return (omega::Float64) -> beta * omega / lambda
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
              spir_kernel_get_sve_hints_ngauss, spir_gauss_legendre_rule_piecewise_double

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

"""
    _get_gauss_points_from_capi(n::Int, segments::Vector{Float64})

Get Gauss-Legendre quadrature points and weights from C-API.

# Arguments
- `n`: Number of Gauss points per segment
- `segments`: Array of segment boundaries (length = n_segments + 1)

# Returns
- `x`: Gauss points (length = n * n_segments)
- `w`: Gauss weights (length = n * n_segments)
"""
function _get_gauss_points_from_capi(n::Int, segments::Vector{Float64})
    n_segments = length(segments) - 1
    n_points = n * n_segments
    x = Vector{Float64}(undef, n_points)
    w = Vector{Float64}(undef, n_points)
    
    status = Ref{Cint}(-100)
    result = spir_gauss_legendre_rule_piecewise_double(
        Cint(n), segments, Cint(n_segments), x, w, status
    )
    
    status[] == SPIR_COMPUTATION_SUCCESS || 
        error("Failed to get Gauss points from C-API: status=$(status[])")
    
    return x, w
end

"""
    matrix_from_gauss(kernel::AbstractKernel, gauss_x::Vector{Float64}, gauss_y::Vector{Float64})

Compute matrix for kernel from Gauss points.

Evaluates the kernel at all pairs of Gauss points (gauss_x[i], gauss_y[j]) to create
a discretized kernel matrix. This is used for singular value expansion via C-API.

# Arguments
- `kernel`: The kernel to evaluate
- `gauss_x`: Gauss points for x direction (length nx)
- `gauss_y`: Gauss points for y direction (length ny)

# Returns
- Matrix of size (nx, ny) containing kernel values K(gauss_x[i], gauss_y[j])
"""
function matrix_from_gauss(kernel::AbstractKernel, gauss_x::Vector{Float64}, gauss_y::Vector{Float64})
    nx = length(gauss_x)
    ny = length(gauss_y)
    K = Matrix{Float64}(undef, nx, ny)
    
    # Evaluate kernel at all pairs (x[i], y[j])
    for i in 1:nx
        for j in 1:ny
            K[i, j] = Float64(kernel(gauss_x[i], gauss_y[j]))
        end
    end
    
    return K
end
