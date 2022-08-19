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
abstract type AbstractBasis{S <: Statistics} end

Base.size(::AbstractBasis) = error("unimplemented")
Base.broadcastable(b::AbstractBasis) = Ref(b)

"""
    Base.getindex(basis::AbstractBasis, I)

Return basis functions/singular values for given index/indices.

This can be used to truncate the basis to the `n` most significant
singular values: `basis[1:3]`.
"""
Base.getindex(::AbstractBasis, _) = error("unimplemented")
Base.firstindex(::AbstractBasis) = 1
Base.length(basis::AbstractBasis) = length(basis.s)

"""
    accuracy(basis::AbstractBasis)

Accuracy of the basis.

Upper bound to the relative error of reprensenting a propagator with
the given number of basis functions (number between 0 and 1).
"""
accuracy(basis::AbstractBasis) = last(significance(basis))

"""
    significance(basis::AbstractBasis)

Return vector `σ`, where `0 ≤ σ[i] ≤ 1` is the significance level of the `i`-th
basis function.  If `ϵ` is the desired accuracy to which to represent a
propagator, then any basis function where `σ[i] < ϵ` can be neglected.

For the IR basis, we simply have that `σ[i] = s[i] / first(s)`.
"""
significance(::AbstractBasis) = error("unimplemented")

"""
    default_tau_sampling_points(basis::AbstractBasis)

Default sampling points on the imaginary time/x axis.
"""
default_tau_sampling_points(::AbstractBasis) = error("unimplemented")

"""
    default_matsubara_sampling_points(basis::AbstractBasis)

Default sampling points on the imaginary frequency axis.
"""
default_matsubara_sampling_points(::AbstractBasis) = error("unimplemented")

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
Λ(::AbstractBasis) = error("unimplemented")
const lambda = Λ

"""
    ωmax(basis::AbstractBasis)
    wmax(basis::AbstractBasis)

Real frequency cutoff or `nothing` if unscaled basis.
"""
ωmax(::AbstractBasis) = error("unimplemented")
const wmax = ωmax

"""
    β(basis::AbstractBasis)
    beta(basis::AbstractBasis)

Inverse temperature or `nothing` if unscaled basis.
"""
β(basis::AbstractBasis) = basis.β
const beta = β

"""
    iswellconditioned(basis::AbstractBasis)

Returns True if the sampling is expected to be well-conditioned.
"""
iswellconditioned(::AbstractBasis) = true

###############################################################################

"""
    AbstractCompositeBasisFunction

Union of several basis functions.
"""
abstract type AbstractCompositeBasisFunction end

function Base.getindex(compfunc::AbstractCompositeBasisFunction, l::Integer)
    offsets = cumsum(length(p) for p in compfunc.polys)
	idx = searchsortedfirst(offsets, l)
	l = idx > 1 ? l - offsets[idx-1] : l
	compfunc.polys[idx][l]
end

Base.length(compfunc::AbstractCompositeBasisFunction) = sum(length(p) for p in compfunc.polys)
Base.size(compfunc::AbstractCompositeBasisFunction) = (length(compfunc), )

###############################################################################

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

abstract type AbstractReducedKernel <: AbstractKernel end

Base.broadcastable(kernel::AbstractKernel) = Ref(kernel)

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
abstract type AbstractSampling{T,Tmat,F<:SVD} end

Base.broadcastable(sampling::AbstractSampling) = Ref(sampling)

function LinearAlgebra.cond(sampling::AbstractSampling)
    first(sampling.matrix_svd.S) / last(sampling.matrix_svd.S)
end

sampling_points(sampling::AbstractSampling) = sampling.sampling_points

function Base.show(io::IO, smpl::S) where {S<:AbstractSampling}
    println(io, S)
    print(io, "Sampling points: ")
    return println(io, smpl.sampling_points)
end

###############################################################################

abstract type AbstractSVE end
