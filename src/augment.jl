"""
    AbstractAugmentation

Scalar function in imaginary time/frequency.

This represents a single function in imaginary time and frequency, together
with some auxiliary methods that make it suitable for augmenting a basis.

See also: [`AugmentedBasis`](@ref)
"""
abstract type AbstractAugmentation{S<:Statistics} <: Function end

const AugmentationTuple{S} = Tuple{Vararg{AbstractAugmentation{S}}} where {S<:Statistics}

# An augmentation passed as an instance must have been built for the basis it
# augments: the same β, and the same statistics (MatsubaraConst, which does not
# depend on the statistics, has its own method below).
function create(aug::AbstractAugmentation{S1}, basis::AbstractBasis{S2}) where {S1,S2}
    _check_augmentation_beta(aug, basis)
    S1 === S2 || throw(ArgumentError("$(nameof(typeof(aug))) is $(nameof(S1)), \
                                      but the basis is $(nameof(S2))"))
    return aug
end

function _check_augmentation_beta(aug::AbstractAugmentation, basis::AbstractBasis)
    isapprox(β(aug), β(basis); rtol=1e-12) ||
        throw(ArgumentError("$(nameof(typeof(aug))) has β = $(β(aug)), \
                             but the basis has β = $(β(basis))"))
    return nothing
end
β(aug::AbstractAugmentation) = aug.β

"""
    AugmentedBasis <: AbstractBasis

Augmented basis on the imaginary-time/frequency axis.

Groups a set of additional functions, `augmentations`, with a given
`basis`. The augmented functions then form the first basis
functions, while the rest is provided by the regular basis, i.e. with Julia's
1-based index `i`:

    u[i](τ) == i ≤ naug ? augmentations[i](τ) : basis.u[i-naug](τ),
    uhat[i](n) == i ≤ naug ? augmentations[i](n) : basis.uhat[i-naug](n),

where `naug = length(augmentations)` is the number of added basis functions
through augmentation, `τ ∈ [-β, β]` and `n` is a reduced frequency (or a
[`MatsubaraFreq`](@ref)).

`AugmentedBasis(basis, augmentations...)` takes each augmentation as a type
(`TauConst`, `TauLinear`, `MatsubaraConst`), which is then built for the β and
the statistics of `basis`, or as an instance, which must have the β of `basis`
and, except for a `MatsubaraConst`, its statistics (`ArgumentError` otherwise).
[`TauConst`](@ref) and [`TauLinear`](@ref) exist for bosons only;
[`MatsubaraConst`](@ref) works for both statistics, and an instance of it adopts
the statistics of the basis.

The default sampling points are those for `L = naug + length(basis)`
functions, the size of the augmented basis: the roots of `U_L` in imaginary
time, always folded into `(0, β)`, and the sign changes of the first discarded
`Û_l` (`l ≥ L`) in Matsubara frequency.

Augmentation is useful in constructing bases for vertex-like quantities
such as self-energies [^wallerberger2021] and when constructing a two-point kernel
that serves as a base for multi-point functions [^shinaoka2018].

!!! warning

    Bases augmented with `TauConst` and `TauLinear` tend to be poorly
    conditioned. Care must be taken while fitting and compactness should
    be enforced if possible to regularize the problem.

    While vertex bases, i.e. bases augmented with `MatsubaraConst`, stay
    reasonably well-conditioned, it is still good practice to treat the
    Hartree--Fock term separately rather than including it in the basis,
    if possible.

See also: [`MatsubaraConst`](@ref) for vertex basis [^wallerberger2021],
[`TauConst`](@ref),
[`TauLinear`](@ref) for multi-point [^shinaoka2018]

[^wallerberger2021]: https://doi.org/10.1103/PhysRevResearch.3.033168
[^shinaoka2018]: https://doi.org/10.1103/PhysRevB.97.205111
"""
struct AugmentedBasis{S<:Statistics,B<:FiniteTempBasis{S},A<:AugmentationTuple{S},F,FHAT} <:
       AbstractBasis{S}
    basis         :: B
    augmentations :: A
    u             :: F
    uhat          :: FHAT
end

function TauSampling(basis::AugmentedBasis{S};
        sampling_points=default_tau_sampling_points(basis; use_positive_taus=true)) where {S}
    # Normalize the element type before the ccall: the C entry point reads
    # Ptr{Cdouble} for both arguments, so both must come from Float64 arrays we
    # own, and both must be finite (a NaN reaches a Rust-side factorization,
    # which panics and returns an uninitialized handle).
    points = convert(Vector{Float64}, collect(sampling_points))
    isempty(points) && throw(ArgumentError("sampling_points must not be empty"))
    _check_all_finite(points, "sampling_points")
    _check_unique(points, "sampling_points")
    matrix = convert(Matrix{Float64}, eval_matrix(TauSampling, basis, points))
    _check_all_finite(matrix, "evaluation matrix")
    status = Ref{Int32}(-100)
    ptr = GC.@preserve points matrix C_API.spir_tau_sampling_new_with_matrix(
        C_API.SPIR_ORDER_COLUMN_MAJOR, _statistics_to_c(S), length(basis),
        length(points), pointer(points), pointer(matrix), status)
    _check_status(status[], "spir_tau_sampling_new_with_matrix")
    _check_handle(ptr, "spir_tau_sampling_new_with_matrix")

    return TauSampling{Float64,typeof(basis)}(ptr, points, basis)
end

function MatsubaraSampling(
        basis::AugmentedBasis{S};
        positive_only=false,
        sampling_points=default_matsubara_sampling_points(basis; positive_only)
) where {S}
    # Integers, parity and statistics are checked as for a plain basis.
    pts = MatsubaraFreq{S}[_to_freq(S, p) for p in sampling_points]
    isempty(pts) && throw(ArgumentError("sampling_points must not be empty"))
    # The C entry point reads Ptr{Int64}; build the Int64 index vector
    # explicitly instead of letting a Vector{<:MatsubaraFreq} be reinterpreted.
    indices = Int64[Int64(Int(p)) for p in pts]
    _check_unique(indices, "sampling_points")
    if positive_only && any(<(0), indices)
        throw(ArgumentError("positive_only=true requires non-negative sampling points, \
                             got $(first(filter(<(0), indices)))π/β"))
    end
    matrix_raw = eval_matrix(MatsubaraSampling, basis, pts)
    # Ensure column-major contiguous memory layout
    # permutedims may create a non-contiguous view, so we create a new Matrix
    matrix = Matrix{ComplexF64}(undef, size(matrix_raw)...)
    matrix .= matrix_raw
    _check_all_finite(matrix, "evaluation matrix")
    status = Ref{Int32}(-100)
    # Keep both arrays pinned for the duration of the call and take the
    # pointers from the very objects that are preserved.
    ptr = GC.@preserve indices matrix C_API.spir_matsu_sampling_new_with_matrix(
        C_API.SPIR_ORDER_COLUMN_MAJOR,
        _statistics_to_c(S),
        length(basis),
        positive_only,
        length(indices),
        pointer(indices),
        pointer(matrix),
        status
    )
    _check_status(status[], "spir_matsu_sampling_new_with_matrix")
    _check_handle(ptr, "spir_matsu_sampling_new_with_matrix")
    return MatsubaraSampling{eltype(pts),typeof(basis)}(ptr, pts, positive_only, basis)
end

function _get_ptr(basis::AugmentedBasis)
    _get_ptr(basis.basis)
end

function AugmentedBasis(basis::AbstractBasis, augmentations...)
    augs = create.(augmentations, basis)
    u = AugmentedTauFunction(basis.u, augs)
    û = AugmentedMatsubaraFunction(basis.uhat, augs)
    return AugmentedBasis(basis, augs, u, û)
end

naug(basis::AugmentedBasis) = length(basis.augmentations)
u(basis::AugmentedBasis) = basis.u
uhat(basis::AugmentedBasis) = basis.uhat

function Base.getindex(basis::AugmentedBasis, index::AbstractRange)
    stop = range_to_length(index)
    stop > naug(basis) ||
        throw(ArgumentError("cannot truncate to only the augmentation functions"))
    return AugmentedBasis(basis.basis[begin:(stop - naug(basis))], basis.augmentations...)
end

Base.size(basis::AugmentedBasis) = (length(basis),)
Base.length(basis::AugmentedBasis) = naug(basis) + length(basis.basis)
accuracy(basis::AugmentedBasis) = accuracy(basis.basis)
Λ(basis::AugmentedBasis) = Λ(basis.basis)
β(basis::AugmentedBasis) = β(basis.basis)
ωmax(basis::AugmentedBasis) = ωmax(basis.basis)

significance(basis::AugmentedBasis) = vcat(ones(naug(basis)), significance(basis.basis))

function default_tau_sampling_points(basis::AugmentedBasis; use_positive_taus::Bool=true)
    points = Vector{Float64}(undef, length(basis))
    n_points_returned = Ref{Cint}(0)
    status = spir_basis_get_default_taus_ext(
        _get_ptr(basis.basis), length(basis), points, n_points_returned)
    _check_status(status, "spir_basis_get_default_taus_ext")
    points = points[1:n_points_returned[]]

    if use_positive_taus
        points = mod.(points, β(basis))
        sort!(points)
    end

    return points
end

function default_matsubara_sampling_points(basis::AugmentedBasis; positive_only=false)
    # The positive-only points are the non-negative half of the full set, as for
    # a plain basis. Requesting that variant from C with the count as the point
    # limit returned a truncated set and left the rest of the buffer unwritten.
    if positive_only
        return filter(≥(0), default_matsubara_sampling_points(basis; positive_only=false))
    end
    n_points = Ref{Cint}(0)
    basis_ptr = _get_ptr(basis.basis)
    mitigate = false # corresponds to false in older version
    status = spir_basis_get_n_default_matsus_ext(
        basis_ptr, positive_only, mitigate, length(basis), n_points)
    _check_status(status, "spir_basis_get_n_default_matsus_ext")
    points = zeros(Int64, n_points[])
    n_points_returned = Ref{Cint}(0)
    status = spir_basis_get_default_matsus_ext(
        basis_ptr, positive_only, mitigate, n_points[], points, n_points_returned)
    _check_status(status, "spir_basis_get_default_matsus_ext")
    # Never return entries the C library did not write.
    0 ≤ n_points_returned[] ≤ length(points) ||
        throw(SparseIRError("spir_basis_get_default_matsus_ext reported \
                             $(n_points_returned[]) points for a buffer of $(length(points))"))
    return points[1:n_points_returned[]]
end

function iswellconditioned(basis::AugmentedBasis)
    wbasis = iswellconditioned(basis.basis)
    waug = isone(naug(basis)) && (only(basis.augmentations) isa MatsubaraConst)
    return wbasis && waug
end

############################################################################################
#                                   Augmented Functions                                    #
############################################################################################

abstract type AbstractAugmentedFunction <: Function end

struct AugmentedFunction{FB,FA} <: AbstractAugmentedFunction
    fbasis :: FB
    faug   :: FA
end

augmentedfunction(a::AugmentedFunction) = a

fbasis(a::AbstractAugmentedFunction) = augmentedfunction(a).fbasis
faug(a::AbstractAugmentedFunction) = augmentedfunction(a).faug
naug(a::AbstractAugmentedFunction) = length(faug(a))

Base.length(a::AbstractAugmentedFunction) = naug(a) + length(fbasis(a))
Base.size(a::AbstractAugmentedFunction) = (length(a),)

function (a::AbstractAugmentedFunction)(x)
    fbasis_x = fbasis(a)(x)
    # Promote to the element type of the basis part: the augmentations of a
    # Matsubara function mix Float64 and ComplexF64 values.
    faug_x = eltype(fbasis_x)[faug_l(x) for faug_l in faug(a)]
    return vcat(faug_x, fbasis_x)
end

function (a::AbstractAugmentedFunction)(x::AbstractArray)
    fbasis_x = fbasis(a)(x)
    n_aug = naug(a)
    if n_aug == 0
        return fbasis_x
    end
    n_x = length(x)
    T = eltype(fbasis_x)
    faug_x = Matrix{T}(undef, n_aug, n_x)
    for (i, faug_l) in enumerate(faug(a))
        for j in 1:n_x
            faug_x[i, j] = convert(T, faug_l(x[j]))
        end
    end
    return vcat(faug_x, fbasis_x)
end

Base.firstindex(::AbstractAugmentedFunction) = 1
Base.lastindex(a::AbstractAugmentedFunction) = length(a)

function _truncate(a::AbstractAugmentedFunction, r::AbstractRange)
    stop = range_to_length(r)
    stop > naug(a) ||
        throw(ArgumentError("cannot truncate to only the augmentation functions"))
    return fbasis(a)[begin:(stop - naug(a))], faug(a)
end
function Base.getindex(a::AugmentedFunction, r::AbstractRange)
    AugmentedFunction(_truncate(a, r)...)
end
function Base.getindex(a::AbstractAugmentedFunction, l::Integer)
    1 ≤ l ≤ length(a) || throw(BoundsError(a, l))
    return l ≤ naug(a) ? faug(a)[l] : fbasis(a)[l - naug(a)]
end

### AugmentedTauFunction

struct AugmentedTauFunction{FB,FA} <: AbstractAugmentedFunction
    a::AugmentedFunction{FB,FA}
end

augmentedfunction(aτ::AugmentedTauFunction) = aτ.a

AugmentedTauFunction(fbasis, faug) = AugmentedTauFunction(AugmentedFunction(fbasis, faug))

xmin(aτ::AugmentedTauFunction) = xmin(fbasis(aτ))
xmax(aτ::AugmentedTauFunction) = xmax(fbasis(aτ))

# Keep the wrapper type: a plain AugmentedFunction would evaluate integer
# Matsubara indices as imaginary times.
function Base.getindex(aτ::AugmentedTauFunction, r::AbstractRange)
    AugmentedTauFunction(_truncate(aτ, r)...)
end

function deriv(aτ::AugmentedTauFunction, n=Val(1))
    # `fbasis(aτ)` is a single `PiecewiseLegendrePolyVector` handle, not an
    # iterable of polynomials, so differentiate it as a whole.
    dbasis = deriv(fbasis(aτ), n)
    daug = deriv.(faug(aτ), n)
    return AugmentedTauFunction(dbasis, daug)
end

### AugmentedMatsubaraFunction

struct AugmentedMatsubaraFunction{FB,FA} <: AbstractAugmentedFunction
    a::AugmentedFunction{FB,FA}
end

augmentedfunction(amat::AugmentedMatsubaraFunction) = amat.a

function AugmentedMatsubaraFunction(fbasis, faug)
    AugmentedMatsubaraFunction(AugmentedFunction(fbasis, faug))
end

zeta(amat::AugmentedMatsubaraFunction) = zeta(fbasis(amat))

function Base.getindex(amat::AugmentedMatsubaraFunction, r::AbstractRange)
    return AugmentedMatsubaraFunction(_truncate(amat, r)...)
end

# An integer is a reduced Matsubara index. The augmentations are defined on
# `MatsubaraFreq`s (on plain numbers they are functions of imaginary time), so
# the index is converted first; the wrong parity throws `DomainError`.
function _as_freq(amat::AugmentedMatsubaraFunction, n::Integer)
    return MatsubaraFreq{typeof(Statistics(zeta(amat)))}(n)
end
(amat::AugmentedMatsubaraFunction)(n::Integer) = amat(_as_freq(amat, n))
function (amat::AugmentedMatsubaraFunction)(ns::AbstractVector{<:Integer})
    return amat([_as_freq(amat, n) for n in ns])
end

############################################################################################
#                                      Augmentations                                       #
############################################################################################

"""
    normalize_tau(S::Type{<:Statistics}, tau, beta) -> (tau_normalized, sign)

Normalize τ to the range [0, β] with statistics-dependent boundary conditions.

Handles boundary conditions based on statistics:

  - Fermions: Anti-periodic G(τ + β) = -G(τ)
  - Bosons: Periodic G(τ + β) = G(τ)

The endpoints are read as one-sided limits, as by `basis.u`: `0.0` is `0⁺` and
`β` is `β⁻` (both returned unchanged), `-0.0` is `0⁻` and `-β` is `(-β)⁺`.

# Arguments

  - `S`: Statistics type (Fermionic or Bosonic)
  - `tau`: Imaginary time in range [-β, β]; `DomainError` outside
  - `beta`: Inverse temperature

# Returns

  - `(tau_normalized, sign)`: Normalized τ ∈ [0, β] and sign factor

# Special Cases

For Fermionic statistics:

  - `tau = -0.0` (negative zero) → `(tau_normalized = β, sign = -1.0)`
  - `tau ∈ [-β, 0)` → wraps to `tau + β ∈ [0, β)` with `sign = -1.0`; in
    particular `tau = -β` → `(0.0, -1.0)`

For Bosonic statistics:

  - `tau = -0.0` (negative zero) → `(tau_normalized = β, sign = 1.0)`
  - `tau ∈ [-β, 0)` → wraps to `tau + β ∈ [0, β)` with `sign = 1.0`; in
    particular `tau = -β` → `(0.0, 1.0)`
"""
function normalize_tau(::Type{S}, tau::Real, beta::Real) where {S<:Statistics}
    tau_f = Float64(tau)
    beta_f = Float64(beta)

    # Check range
    if tau_f < -beta_f || tau_f > beta_f
        throw(DomainError(tau_f, "τ must be in [-β, β] = [$(-beta_f), $beta_f]"))
    end

    # Special handling for negative zero
    if signbit(tau_f) && tau_f == 0.0
        # tau = -0.0
        if S === Fermionic
            return (beta_f, -1.0)  # Anti-periodic: wraps to beta with sign flip
        else  # Bosonic
            return (beta_f, 1.0)   # Periodic: wraps to beta with sign unchanged
        end
    end

    # If already in [0, β], return as-is with sign = 1
    if tau_f >= 0.0 && tau_f <= beta_f
        return (tau_f, 1.0)
    end

    # tau ∈ [-β, 0): wrap to [0, β]
    tau_normalized = tau_f + beta_f

    # Sign depends on statistics
    sign = S === Fermionic ? -1.0 : 1.0

    return (tau_normalized, sign)
end

"""
    TauConst{Bosonic} <: AbstractAugmentation{Bosonic}

Constant function in imaginary time, `1/√β` on `[0, β]` and periodic, whose
Matsubara transform is `√β` at `ν = 0` and zero at every other frequency.

Defined for bosons only: `TauConst{Fermionic}` throws `ArgumentError`.
"""
struct TauConst{S<:Statistics} <: AbstractAugmentation{S}
    β::Float64
    function TauConst{S}(β) where {S<:Statistics}
        S === Bosonic || throw(ArgumentError("TauConst is defined for bosons only, got $S"))
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new{S}(β)
    end
end

# Backward compatibility: TauConst(β) defaults to Bosonic
TauConst(β) = TauConst{Bosonic}(β)

# The statistics of the basis; the constructor rejects fermions.
create(::Type{TauConst}, basis::AbstractBasis{S}) where {S} = TauConst{S}(β(basis))
function create(::Type{TauConst{S}}, basis::AbstractBasis{S}) where {S<:Statistics}
    TauConst{S}(β(basis))
end

function (aug::TauConst{S})(τ) where {S<:Statistics}
    tau_normalized, sign = normalize_tau(S, τ, β(aug))
    return sign / sqrt(β(aug))
end
function (aug::TauConst{S})(n::MatsubaraFreq{S}) where {S<:Statistics}
    iszero(n.n) || return zero(β(aug))
    return sqrt(β(aug))
end

function deriv(aug::TauConst, (::Val{n})=Val(1)) where {n}
    iszero(n) && return aug
    return τ -> zero(β(aug))
end

"""
    TauLinear{Bosonic} <: AbstractAugmentation{Bosonic}

Linear function in imaginary time, `√(3/β) (2τ/β - 1)` on `[0, β]`, antisymmetric
around `β/2` and periodic, whose Matsubara transform is `2√(3/β)/(iν)` and zero
at `ν = 0`.

Defined for bosons only: `TauLinear{Fermionic}` throws `ArgumentError`.
"""
struct TauLinear{S<:Statistics} <: AbstractAugmentation{S}
    β::Float64
    norm::Float64
    function TauLinear{S}(β) where {S<:Statistics}
        S === Bosonic ||
            throw(ArgumentError("TauLinear is defined for bosons only, got $S"))
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        norm = sqrt(3 / β)
        return new{S}(β, norm)
    end
end

# Backward compatibility: TauLinear(β) defaults to Bosonic
TauLinear(β) = TauLinear{Bosonic}(β)

# The statistics of the basis; the constructor rejects fermions.
create(::Type{TauLinear}, basis::AbstractBasis{S}) where {S} = TauLinear{S}(β(basis))
function create(::Type{TauLinear{S}}, basis::AbstractBasis{S}) where {S<:Statistics}
    TauLinear{S}(β(basis))
end

function (aug::TauLinear{S})(τ) where {S<:Statistics}
    tau_normalized, sign = normalize_tau(S, τ, β(aug))
    x = 2 / β(aug) * tau_normalized - 1
    return sign * aug.norm * x
end
function (aug::TauLinear{S})(n::MatsubaraFreq{S}) where {S<:Statistics}
    inv_w = value(n, β(aug))
    inv_w = iszero(n.n) ? inv_w : 1 / inv_w
    return aug.norm * 2 / im * inv_w
end

function deriv(aug::TauLinear, (::Val{n})=Val(1)) where {n}
    iszero(n) && return aug
    isone(n) && return τ -> aug.norm * 2 / β(aug)
    return τ -> zero(β(aug))
end

"""
    MatsubaraConst{S} <: AbstractAugmentation{S}

Constant in Matsubara, undefined in imaginary time: its value is `1` at every
Matsubara frequency, and it returns `NaN` for `τ ∈ [-β, β]` (`DomainError`
outside).

# Type Parameters

  - `S`: Statistics type (Fermionic or Bosonic). This is required for type consistency,
    though MatsubaraConst works identically for both statistics. `MatsubaraConst(β)`
    is bosonic; as an augmentation, both the bare type and an instance take the
    statistics of the basis they augment.
"""
struct MatsubaraConst{S<:Statistics} <: AbstractAugmentation{S}
    β::Float64
    function MatsubaraConst{S}(β) where {S<:Statistics}
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new{S}(β)
    end
end

# Backward compatibility: MatsubaraConst(β) is bosonic; `create` adopts the
# statistics of the basis it is added to.
MatsubaraConst(β) = MatsubaraConst{Bosonic}(β)

function create(::Type{MatsubaraConst}, basis::AbstractBasis{S}) where {S}
    MatsubaraConst{S}(β(basis))
end
function create(::Type{MatsubaraConst{S}}, basis::AbstractBasis{S}) where {S<:Statistics}
    MatsubaraConst{S}(β(basis))
end
# An instance takes the statistics of the basis; its β must match.
function create(aug::MatsubaraConst, basis::AbstractBasis{S}) where {S}
    _check_augmentation_beta(aug, basis)
    return MatsubaraConst{S}(β(aug))
end

function (aug::MatsubaraConst)(τ)
    -β(aug) ≤ τ ≤ β(aug) || throw(DomainError(τ, "τ must be in [-β, β]."))
    return NaN
end

function (aug::MatsubaraConst)(::MatsubaraFreq)
    return one(β(aug))
end
(aug::TauConst)(n::MatsubaraFreq) = _statistics_mismatch(aug, n)
(aug::TauLinear)(n::MatsubaraFreq) = _statistics_mismatch(aug, n)

function _statistics_mismatch(aug::AbstractAugmentation{S}, n::MatsubaraFreq) where {S}
    throw(ArgumentError("the frequency $(Int(n))π/β is $(nameof(typeof(statistics(n)))), \
                         but $(nameof(typeof(aug))) is $(nameof(S))"))
end

deriv(aug::MatsubaraConst, _=Val(1)) = aug
