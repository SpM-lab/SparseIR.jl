"""
    AbstractBasis

Abstract base class for bases on the imaginary-time axis.

Let `basis` be an abstract basis. Then we can expand a two-point
propagator  `G(τ)`, where `τ` is imaginary time, into a set of basis
functions:

    G(τ) == sum(basis.u[l](τ) * g[l] for l in 1:length(basis)) + ϵ(τ),

where `basis.u[l]` is the `l`-th basis function, `g[l]` is the associated
expansion coefficient and `ϵ(τ)` is an error term. Similarly, the Fourier
transform `Ĝ(n)`, where `n` is now a Matsubara frequency, can be expanded
as follows:

    Ĝ(n) == sum(basis.uhat[l](n) * g[l] for l in 1:length(basis)) + ϵ(n),

where `basis.uhat[l]` is now the Fourier transform of the basis function.
"""
abstract type AbstractBasis{S<:Statistics} end

@doc raw"""
    AbstractKernel

Integral kernel `K(x, y)`.

Abstract base type for an integral kernel, i.e. a AbstractFloat binary function
``K(x, y)`` used in a Fredhold integral equation of the first kind:
```math
    u(x) = ∫ K(x, y) v(y) dy
```
where ``x ∈ [x_\mathrm{min}, x_\mathrm{max}]`` and
``y ∈ [y_\mathrm{min}, y_\mathrm{max}]``. For its SVE to exist,
the kernel must be square-integrable, for its singular values to decay
exponentially, it must be smooth.

In general, the kernel is applied to a scaled spectral function ``ρ'(y)`` as:
```math
    ∫ K(x, y) ρ'(y) dy,
```
where ``ρ'(y) = w(y) ρ(y)``.
"""
abstract type AbstractKernel end

abstract type AbstractReducedKernel <: AbstractKernel end

"""
    Base.checkbounds(kernel::AbstractKernel, x, y)

Check if `(x, y)` is within the domain of `kernel`.

This method uses `xrange` and `yrange` to determine if the point is valid.
"""
function Base.checkbounds(kernel::AbstractKernel, x, y)
    try
        xmin, xmax = xrange(kernel)
        ymin, ymax = yrange(kernel)
        xmin ≤ x ≤ xmax || return false
        ymin ≤ y ≤ ymax || return false
        return true
    catch e
        # If xrange or yrange fails, allow the call to proceed
        # The kernel's compute method will handle domain errors
        return true
    end
end

###############################################################################

"""
    AbstractSVEHints

Discretization hints for singular value expansion of a given kernel.
"""
abstract type AbstractSVEHints end

###############################################################################

"""
    AbstractSampling

Abstract type for sparse sampling.

Encodes the "basis transformation" of a propagator from the truncated IR
basis coefficients `G_ir[l]` to time/frequency sampled on sparse points
`G(x[i])` together with its inverse, a least squares fit:

         ________________                   ___________________
        |                |    evaluate     |                   |
        |     Basis      |---------------->|     Value on      |
        |  coefficients  |<----------------|  sampling points  |
        |________________|      fit        |___________________|
"""
abstract type AbstractSampling{T,Tmat,F} end

###############################################################################

abstract type AbstractSVE end

_get_ptr(basis::AbstractBasis) = basis.ptr

Base.broadcastable(b::AbstractBasis) = Ref(b)
Base.firstindex(::AbstractBasis) = 1
Base.length(basis::AbstractBasis) = length(basis.s)

"""
    accuracy(basis::AbstractBasis)

Accuracy of the basis.

Upper bound to the relative error of representing a propagator with
the given number of basis functions (number between 0 and 1).
"""
function accuracy end

"""
    significance(basis::AbstractBasis)

Return vector `σ`, where `0 ≤ σ[i] ≤ 1` is the significance level of the `i`-th
basis function. If `ϵ` is the desired accuracy to which to represent a
propagator, then any basis function where `σ[i] < ϵ` can be neglected.

For the IR basis, we simply have that `σ[i] = s[i] / first(s)`.
"""
function significance end

"""
    s(basis::AbstractBasis)

Get the singular values of the basis.
"""
function s end

"""
    u(basis::AbstractBasis)

Get the u basis functions (imaginary time).
"""
function u end

"""
    v(basis::AbstractBasis)

Get the v basis functions (real frequency).
"""
function v end

"""
    uhat(basis::AbstractBasis)

Get the uhat basis functions (Matsubara frequency).
"""
function uhat end

"""
    default_tau_sampling_points(basis::AbstractBasis)

Default sampling points on the imaginary time/x axis.
"""
function default_tau_sampling_points end

"""
    default_matsubara_sampling_points(basis::AbstractBasis; positive_only=false)

Default sampling points on the imaginary frequency axis.

# Arguments

  - `positive_only::Bool`: Only return non-negative frequencies. This is useful if the
    object to be fitted is symmetric in Matsubura frequency, `ĝ(ω) == conj(ĝ(-ω))`,
    or, equivalently, real in imaginary time.
"""
function default_matsubara_sampling_points end

"""
    statistics(basis::AbstractBasis)

Quantum statistic (Statistics instance, Fermionic() or Bosonic()).
"""
statistics(::AbstractBasis{S}) where {S<:Statistics} = S()

"""
    Λ(basis::AbstractBasis)
    lambda(basis::AbstractBasis)

Basis cutoff parameter, `Λ = β * ωmax`, or None if not present
"""
function Λ end
const lambda = Λ

"""
    ωmax(basis::AbstractBasis)
    wmax(basis::AbstractBasis)

Real frequency cutoff or `nothing` if unscaled basis.
"""
function ωmax end
const wmax = ωmax

"""
    β(basis::AbstractBasis)
    beta(basis::AbstractBasis)

Inverse temperature of the basis.

Returns the inverse temperature parameter β used in the basis construction.
"""
β(basis::AbstractBasis) = basis.β
const beta = β

"""
    iswellconditioned(basis::AbstractBasis)

Returns true if the sampling is expected to be well-conditioned.
"""
iswellconditioned(::AbstractBasis) = true

###############################################################################

"""
    iscentrosymmetric(kernel::AbstractKernel)

Return whether the kernel satisfies `K(x, y) == K(-x, -y)` for all values of x and y.
Defaults to `false`.

A centrosymmetric kernel can be block-diagonalized, speeding up the singular value
expansion by a factor of 4.
"""
iscentrosymmetric(::AbstractKernel) = false

"""
    xrange(kernel::AbstractKernel)

Return the x-domain range `(xmin, xmax)` of the kernel.
"""
function xrange end

"""
    yrange(kernel::AbstractKernel)

Return the y-domain range `(ymin, ymax)` of the kernel.
"""
function yrange end

"""
    conv_radius(kernel::AbstractKernel)

Convergence radius of the Matsubara basis asymptotic model.

For improved relative numerical accuracy, the IR basis functions on the Matsubara axis
can be evaluated from an asymptotic expression for abs(n) > conv_radius. If conv_radius
is infinity, then the asymptotics are unused (the default).
"""
conv_radius(::AbstractKernel) = Inf

"""
    weight_func(kernel::AbstractKernel, stat::Statistics)

Return the weight function for given statistics.

Returns a function `(beta, omega) -> weight` that scales the spectral function.
"""
function weight_func end

"""
    sve_hints(kernel::AbstractKernel, epsilon::Real)

Return discretization hints for singular value expansion of a given kernel.
"""
function sve_hints end

"""
    segments_x(hints::AbstractSVEHints, ::Type{T}=Float64)

Return the x-segments for the SVE discretization.
"""
function segments_x end

"""
    segments_y(hints::AbstractSVEHints, ::Type{T}=Float64)

Return the y-segments for the SVE discretization.
"""
function segments_y end

"""
    nsvals(hints::AbstractSVEHints)

Return the number of singular values for the SVE discretization.
"""
function nsvals end

"""
    ngauss(hints::AbstractSVEHints)

Return the number of Gauss points for the SVE discretization.
"""
function ngauss end

Base.broadcastable(kernel::AbstractKernel) = Ref(kernel)

# Global cache to store custom kernel pointers for Julia kernel objects
# This allows us to cache the C-API custom kernel objects created from Julia kernels
# The cache is used to manage the lifetime of the FunctionKernel objects
const _custom_kernel_cache = Dict{Any, Ptr{spir_kernel}}()

"""
    _get_ptr(kernel::AbstractKernel)

Get the underlying C pointer from a kernel. 

For custom kernels (not LogisticKernel or RegularizedBoseKernel), automatically
creates a C-API custom kernel object using spir_function_kernel_new if not already created.
The result is cached to manage the lifetime of the FunctionKernel object.
"""
function _get_ptr(kernel::AbstractKernel)
    # Check if this is a standard kernel (LogisticKernel or RegularizedBoseKernel)
    if kernel isa LogisticKernel || kernel isa RegularizedBoseKernel
        return kernel.ptr
    end
    
    # For custom kernels, check cache first
    if haskey(_custom_kernel_cache, kernel)
        return _custom_kernel_cache[kernel]
    end
    
    # Not in cache, create custom kernel
    custom_ptr = _create_custom_kernel(kernel)
    
    # Store in cache to manage lifetime
    _custom_kernel_cache[kernel] = custom_ptr
    
    return custom_ptr
end

Base.broadcastable(sampling::AbstractSampling) = Ref(sampling)

function LinearAlgebra.cond(sampling::AbstractSampling)
    cond_num = Ref{Float64}(-1.0)
    status = spir_sampling_get_cond_num(sampling.ptr, cond_num)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get condition number: $status")
    return cond_num[]
end

"""
    sampling_points(sampling::AbstractSampling)

Return sampling points.
"""
sampling_points(sampling::AbstractSampling) = sampling.sampling_points

"""
    basis(sampling::AbstractSampling)

Return the IR basis associated with `sampling`.
"""
basis(sampling::AbstractSampling) = sampling.basis

function Base.show(io::IO, ::MIME"text/plain", smpl::S) where {S<:AbstractSampling}
    println(io, "$S with sampling points:")
    for p in sampling_points(smpl)[begin:(end - 1)]
        println(io, " $p")
    end
    print(io, " $(last(sampling_points(smpl)))")
end
