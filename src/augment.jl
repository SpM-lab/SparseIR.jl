"""
    AbstractAugmentation

Scalar function in imaginary time/frequency.

This represents a single function in imaginary time and frequency, together
with some auxiliary methods that make it suitable for augmenting a basis.

See also: [`AugmentedBasis`](@ref)
"""
abstract type AbstractAugmentation <: Function end

const AugmentationTuple = Tuple{Vararg{AbstractAugmentation}}

create(aug::AbstractAugmentation, ::AbstractBasis) = aug
β(aug::AbstractAugmentation) = aug.β

"""
    AugmentedBasis <: AbstractBasis

Augmented basis on the imaginary-time/frequency axis.

Groups a set of additional functions, `augmentations`, with a given
`basis`. The augmented functions then form the first basis
functions, while the rest is provided by the regular basis, i.e.:

    u[l](x) == l < naug ? augmentations[l](x) : basis.u[l-naug](x),

where `naug = length(augmentations)` is the number of added basis functions
through augmentation. Similar expressions hold for Matsubara frequencies.

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
struct AugmentedBasis{S<:Statistics,B<:FiniteTempBasis{S},A<:AugmentationTuple,F,FHAT} <:
       AbstractBasis{S}
    basis         :: B
    augmentations :: A
    u             :: F
    uhat          :: FHAT
end

function TauSampling(basis::AugmentedBasis{S};
        sampling_points=default_tau_sampling_points(basis; use_positive_taus=true)) where {S}
    matrix = eval_matrix(TauSampling, basis, sampling_points)
    status = Ref{Int32}(-100)
    ptr = C_API.spir_tau_sampling_new_with_matrix(
        C_API.SPIR_ORDER_COLUMN_MAJOR, _statistics_to_c(S), length(basis),
        length(sampling_points), sampling_points, matrix, status)
    status[] == C_API.SPIR_COMPUTATION_SUCCESS ||
        error("Failed to create tau sampling: status=$(status[])")
    ptr != C_NULL || error("Failed to create tau sampling: null pointer returned")

    return TauSampling{Float64,typeof(basis)}(ptr, sampling_points, basis)
end

function MatsubaraSampling(
        basis::AugmentedBasis{S};
        positive_only=false,
        sampling_points=default_matsubara_sampling_points(basis; positive_only)
) where {S}
    pts = MatsubaraFreq.(sampling_points)
    matrix_raw = eval_matrix(MatsubaraSampling, basis, pts)
    # Ensure column-major contiguous memory layout
    # permutedims may create a non-contiguous view, so we create a new Matrix
    matrix = Matrix{ComplexF64}(undef, size(matrix_raw)...)
    matrix .= matrix_raw
    status = Ref{Int32}(-100)
    # Ensure matrix is pinned in memory to prevent GC from moving it during ccall
    GC.@preserve matrix begin
    ptr = C_API.spir_matsu_sampling_new_with_matrix(
        C_API.SPIR_ORDER_COLUMN_MAJOR,
        _statistics_to_c(S),
        length(basis),
        positive_only,
        length(sampling_points),
        sampling_points,
        matrix,
        status
    )
    end
    status[] == C_API.SPIR_COMPUTATION_SUCCESS ||
        error("Failed to create Matsubara sampling: status=$(status[])")
    ptr != C_NULL || error("Failed to create Matsubara sampling: null pointer returned")
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

function Base.getindex(basis::AugmentedBasis, index::AbstractRange)
    stop = range_to_length(index)
    stop > naug(basis) || error("Cannot truncate to only augmentation.")
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
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get default tau sampling points")
    points = points[1:n_points_returned[]]
    
    if use_positive_taus
        points = mod.(points, β(basis))
        sort!(points)
    end
    
    return points
end

function default_matsubara_sampling_points(basis::AugmentedBasis; positive_only=false)
    n_points = Ref{Cint}(0)
    basis_ptr = _get_ptr(basis.basis)
    mitigate = false # corresponds to false in older version
    status = spir_basis_get_n_default_matsus_ext(
        basis_ptr, positive_only, mitigate, length(basis), n_points)
    status == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get number of default Matsubara sampling points")
    points = Vector{Int64}(undef, n_points[])
    n_points_returned = Ref{Cint}(0)
    status = spir_basis_get_default_matsus_ext(
        basis_ptr, positive_only, mitigate, n_points[], points, n_points_returned)
    status == SPIR_COMPUTATION_SUCCESS ||
        error("Failed to get default Matsubara sampling points")
    return points
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
    faug_x = [faug_l(x) for faug_l in faug(a)]
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

function Base.getindex(a::AbstractAugmentedFunction, r::AbstractRange)
    stop = range_to_length(r)
    stop > naug(a) || error("Don't truncate to only augmentation")
    return AugmentedFunction(fbasis(a)[begin:(stop - naug(a))], faug(a))
end
function Base.getindex(a::AbstractAugmentedFunction, l::Integer)
    return l ≤ naug(a) ? faug(a)[l] : fbasis(a)[l - naug(a)]
end

### AugmentedTauFunction

struct AugmentedTauFunction{FB,FA} <: AbstractAugmentedFunction
    a::AugmentedFunction{FB,FA}
end

augmentedfunction(aτ::AugmentedTauFunction) = aτ.a

AugmentedTauFunction(fbasis, faug) = AugmentedTauFunction(AugmentedFunction(fbasis, faug))

# Not supported yet
#xmin(aτ::AugmentedTauFunction) = xmin(fbasis(aτ))
#xmax(aτ::AugmentedTauFunction) = xmax(fbasis(aτ))

function deriv(aτ::AugmentedTauFunction, n=Val(1))
    dbasis = PiecewiseLegendrePolyVector(deriv.(fbasis(aτ), n))
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

############################################################################################
#                                      Augmentations                                       #
############################################################################################

"""
    normalize_tau(S::Type{<:Statistics}, tau, beta) -> (tau_normalized, sign)

Normalize τ to the range [0, β] with statistics-dependent boundary conditions.

Handles boundary conditions based on statistics:
- Fermions: Anti-periodic G(τ + β) = -G(τ)
- Bosons: Periodic G(τ + β) = G(τ)

# Arguments
- `S`: Statistics type (Fermionic or Bosonic)
- `tau`: Imaginary time in range [-β, β]
- `beta`: Inverse temperature

# Returns
- `(tau_normalized, sign)`: Normalized τ ∈ [0, β] and sign factor

# Special Cases
For Fermionic statistics:
- `tau = -0.0` (negative zero) → `(tau_normalized = β, sign = -1.0)`
- `tau ∈ [-β, 0)` → wraps to [0, β] with `sign = -1.0`

For Bosonic statistics:
- `tau = -0.0` (negative zero) → `(tau_normalized = β, sign = 1.0)`
- `tau ∈ [-β, 0)` → wraps to [0, β] with `sign = 1.0`
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
    TauConst{S} <: AbstractAugmentation

Constant function in imaginary time with statistics-dependent periodicity.

# Type Parameters
- `S`: Statistics type (Fermionic or Bosonic)
"""
struct TauConst{S<:Statistics} <: AbstractAugmentation
    β::Float64
    function TauConst{S}(β) where {S<:Statistics}
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new{S}(β)
    end
end

# Backward compatibility: TauConst(β) defaults to Bosonic
TauConst(β) = TauConst{Bosonic}(β)

create(::Type{TauConst}, basis::AbstractBasis{Bosonic}) = TauConst{Bosonic}(β(basis))
create(::Type{TauConst{S}}, basis::AbstractBasis{S}) where {S<:Statistics} = TauConst{S}(β(basis))

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
    TauLinear{S} <: AbstractAugmentation

Linear function in imaginary time, antisymmetric around β/2, with statistics-dependent periodicity.

# Type Parameters
- `S`: Statistics type (Fermionic or Bosonic)
"""
struct TauLinear{S<:Statistics} <: AbstractAugmentation
    β::Float64
    norm::Float64
    function TauLinear{S}(β) where {S<:Statistics}
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        norm = sqrt(3 / β)
        return new{S}(β, norm)
    end
end

# Backward compatibility: TauLinear(β) defaults to Bosonic
TauLinear(β) = TauLinear{Bosonic}(β)

create(::Type{TauLinear}, basis::AbstractBasis{Bosonic}) = TauLinear{Bosonic}(β(basis))
create(::Type{TauLinear{S}}, basis::AbstractBasis{S}) where {S<:Statistics} = TauLinear{S}(β(basis))

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
    MatsubaraConst <: AbstractAugmentation

Constant in Matsubara, undefined in imaginary time.
"""
struct MatsubaraConst <: AbstractAugmentation
    β::Float64
    function MatsubaraConst(β)
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new(β)
    end
end

create(::Type{MatsubaraConst}, basis::AbstractBasis) = MatsubaraConst(β(basis))

function (aug::MatsubaraConst)(τ)
    -β(aug) ≤ τ ≤ β(aug) || throw(DomainError(τ, "τ must be in [-β, β]."))
    return NaN
end

function (aug::MatsubaraConst)(::MatsubaraFreq)
    return one(β(aug))
end

deriv(aug::MatsubaraConst, _=Val(1)) = aug
