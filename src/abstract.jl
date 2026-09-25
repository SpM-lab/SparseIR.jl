"""
    AbstractBasis

Abstract base class for bases on the imaginary-time axis.

Let `basis` be an abstract basis with `L = length(basis)` functions. Then we can
expand a two-point propagator `G(τ)`, where `τ` is imaginary time, into the
basis functions `U_l(τ)`, `l = 0, …, L-1`:

    G(τ) ≈ sum(basis.u[l+1](τ) * g[l+1] for l in 0:L-1),

where Julia's `basis.u[l+1]` is `U_l` and `g[l+1]` is the associated expansion
coefficient `G_l`; the difference is the truncation error of the basis.
Similarly, the Fourier transform `G(iν)`, where `ν = nπ/β` is a Matsubara
frequency with reduced frequency `n`, can be expanded as follows:

    G(iν) ≈ sum(basis.uhat[l+1](n) * g[l+1] for l in 0:L-1),

where `basis.uhat[l+1]` is `Û_l`, the Fourier transform of `U_l`.
"""
abstract type AbstractBasis{S<:Statistics} end

@doc raw"""
    AbstractKernel

Integral kernel `K(x, y)`.

Abstract base type for an integral kernel, i.e. a real-valued function
``K(x, y)`` of the dimensionless variables ``x`` and ``y``, used in a Fredholm
integral equation of the first kind:
```math
    u(x) = ∫ K(x, y) v(y) dy
```
where ``x ∈ [x_\mathrm{min}, x_\mathrm{max}]`` and
``y ∈ [y_\mathrm{min}, y_\mathrm{max}]``. For its SVE to exist,
the kernel must be square-integrable, for its singular values to decay
exponentially, it must be smooth.

In general, the kernel is applied to a weighted spectral function ``ρ(y)`` as:
```math
    ∫ K(x, y) ρ(y) dy,
```
where ``ρ(y) = w(y) A(y)`` is the spectral function ``A`` times a weight ``w``
that depends on the kernel and the statistics (see [`LogisticKernel`](@ref)).
"""
abstract type AbstractKernel end

abstract type AbstractReducedKernel <: AbstractKernel end

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
basis coefficients `G_l` to its values `G(τ_i)` or `G(iν_i)` on sparse sampling
points in imaginary time or Matsubara frequency, together with its inverse, a
least squares fit:

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
the given number of basis functions (number between 0 and 1). For an IR basis
of size `L` it is `S_L/S_0`, the first discarded singular value relative to the
largest one.
"""
function accuracy end

"""
    significance(basis::AbstractBasis)

Return vector `σ`, where `0 ≤ σ[l+1] ≤ 1` is the significance level of the
basis function `U_l`. If `ε` is the desired accuracy to which to represent a
propagator, then any basis function where `σ[l+1] < ε` can be neglected.

For the IR basis, we simply have that `σ[l+1] = S_l / S_0`.
"""
function significance end

"""
    s(basis::AbstractBasis)

Get the singular values `S_l` of the basis, `basis.s`; `s(basis)[l+1]` is `S_l`.
"""
function s end

"""
    u(basis::AbstractBasis)

Get the basis functions in imaginary time, `basis.u`: for an IR basis the
`U_l(τ)`, with `u(basis)[l+1]` being `U_l`. They accept `τ ∈ [-β, β]`; see
[`FiniteTempBasis`](@ref) for the extension to negative `τ` and the endpoints.
"""
function u end

"""
    v(basis::AbstractBasis)

Get the basis functions `V_l(ω)` in real frequency, `basis.v`, for
`ω ∈ [-ωmax, ωmax]`; `v(basis)[l+1]` is `V_l`.
"""
function v end

"""
    uhat(basis::AbstractBasis)

Get the basis functions in Matsubara frequency, `basis.uhat`, the Fourier
transforms of those of [`u`](@ref): for an IR basis the `Û_l(iν)`, with
`uhat(basis)[l+1]` being `Û_l`. They take the reduced frequency `n`
(`ν = nπ/β`) or a [`MatsubaraFreq`](@ref).
"""
function uhat end

"""
    default_tau_sampling_points(basis::AbstractBasis; use_positive_taus=true)

Default sampling points in imaginary time: the roots of `U_L`, the first basis
function beyond a basis of size `L = length(basis)`.

With `use_positive_taus=true` (the default) the points are folded into `(0, β)`
and sorted. With `use_positive_taus=false` they are returned unfolded, in
`(-β/2, β/2]` and symmetric about 0. A DLR uses the points of its IR basis.
"""
function default_tau_sampling_points end

"""
    default_matsubara_sampling_points(basis::AbstractBasis; positive_only=false)

Default sampling points on the imaginary frequency axis, as a `Vector{Int}` of
reduced frequencies `n` (`ν = nπ/β`, odd for fermions, even for bosons).

The points are the sign changes of the first discarded transform `Û_l`, with
`l ≥ L = length(basis)` chosen to fit the parity. Bosonic sets always include
`n = 0`. A DLR uses the points of its IR basis.

# Arguments

  - `positive_only::Bool`: Only return non-negative frequencies, `n ≥ 0`. This is
    useful if the object to be fitted is symmetric in Matsubara frequency,
    `G(-iν) == conj(G(iν))`, or, equivalently, real in imaginary time.
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

Basis cutoff parameter, `Λ = β * ωmax`.
"""
function Λ end
const lambda = Λ

"""
    ωmax(basis::AbstractBasis)
    wmax(basis::AbstractBasis)

Real frequency cutoff `ωmax` of the basis: the spectral function is represented
on `[-ωmax, ωmax]`.
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

Base.broadcastable(kernel::AbstractKernel) = Ref(kernel)

Base.broadcastable(sampling::AbstractSampling) = Ref(sampling)

LinearAlgebra.cond(sampling::AbstractSampling) = _cond_from_c(sampling)

function _cond_from_c(sampling::AbstractSampling)
    cond_num = Ref{Float64}(-1.0)
    status = spir_sampling_get_cond_num(sampling.ptr, cond_num)
    _check_status(status, "spir_sampling_get_cond_num")
    return cond_num[]
end

"""
    sampling_points(sampling::AbstractSampling)

Return sampling points: a `Vector{Float64}` of imaginary times `τ` for a
[`TauSampling`](@ref), a `Vector{FermionicFreq}` or `Vector{BosonicFreq}` for a
[`MatsubaraSampling`](@ref). For a DLR, `sampling_points(dlr)` returns its poles.
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
