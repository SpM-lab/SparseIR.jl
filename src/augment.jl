"""
    AbstractAugmentation

Scalar function in imaginary time/frequency.

This represents a single function in imaginary time and frequency, together
with some auxiliary methods that make it suitable for augmenting a basis.

See also: [`AugmentedBasis`](@ref)
"""
abstract type AbstractAugmentation <: Function end

const AugmentationTuple = Tuple{Vararg{<:AbstractAugmentation}}

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
struct AugmentedBasis{S<:Statistics,B<:FiniteTempBasis{S},A<:AugmentationTuple,F,FHAT} <: AbstractBasis{S}
    basis         :: B
    augmentations :: A
    u             :: F
    uhat          :: FHAT
end

function AugmentedBasis(basis::AbstractBasis, augmentations...)
    augmentations = Tuple(create(aug, basis) for aug in augmentations)
    u = AugmentedTauFunction(basis.u, augmentations)
    û = AugmentedMatsubaraFunction(basis.uhat, augmentations)
    return AugmentedBasis(basis, augmentations, u, û)
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

significance(basis::AugmentedBasis) =
    vcat(ones(naug(basis)), significance(basis.basis))

function default_tau_sampling_points(basis::AugmentedBasis)
    x = default_sampling_points(basis.basis.sve_result.u, length(basis))
    return β(basis) / 2 * (x .+ 1)
end
function default_matsubara_sampling_points(basis::AugmentedBasis)
    return default_matsubara_sampling_points(basis.basis.uhat_full, length(basis))
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
    faug_x = [faug_l.(transpose(x)) for faug_l in faug(a)]
    return vcat(faug_x..., fbasis_x)
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

xmin(aτ::AugmentedTauFunction) = fbasis(aτ).xmin
xmax(aτ::AugmentedTauFunction) = fbasis(aτ).xmax

function deriv(aτ::AugmentedTauFunction, n=1)
    dbasis = deriv.(fbasis(aτ), n)
    daug = [deriv(faug_l, n) for faug_l in faug(aτ)]
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

### TauConst

struct TauConst <: AbstractAugmentation
    β::Float64
    function TauConst(β)
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new(β)
    end
end

create(::Type{TauConst}, basis::AbstractBasis{Bosonic}) = TauConst(β(basis))

function (aug::TauConst)(τ)
    0 ≤ τ ≤ β(aug) || throw(DomainError(τ, "τ must be in [0, β]."))
    return 1 / √(β(aug))
end
function (aug::TauConst)(n::BosonicFreq)
    iszero(n) || return zero(β(aug))
    return √(β(aug))
end
(::TauConst)(::FermionicFreq) = error("TauConst is not a Fermionic basis.")

function deriv(aug::TauConst, n=1)
    iszero(n) && return aug
    return τ -> zero(β(aug))
end

### TauLinear

struct TauLinear <: AbstractAugmentation
    β    :: Float64
    norm :: Float64
    function TauLinear(β)
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        norm = √(3 / β)
        return new(β, norm)
    end
end

create(::Type{TauLinear}, basis::AbstractBasis{Bosonic}) = TauLinear(β(basis))

function (aug::TauLinear)(τ)
    0 ≤ τ ≤ β(aug) || throw(DomainError(τ, "τ must be in [0, β]."))
    x = 2 / β(aug) * τ - 1
    return aug.norm * x
end
function (aug::TauLinear)(n::BosonicFreq)
    inv_w = value(n, β(aug))
    inv_w = iszero(n) ? inv_w : 1 / inv_w
    return aug.norm * 2 / im * inv_w
end
(::TauLinear)(::FermionicFreq) = error("TauLinear is not a Fermionic basis.")

function deriv(aug::TauLinear, n=1)
    iszero(n) && return aug
    isone(n) && return τ -> aug.norm * 2 / β(aug)
    return τ -> zero(β(aug))
end

### MatsubaraConst

struct MatsubaraConst <: AbstractAugmentation
    β::Float64
    function MatsubaraConst(β)
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new(β)
    end
end

create(::Type{MatsubaraConst}, basis::AbstractBasis) = MatsubaraConst(β(basis))

function (aug::MatsubaraConst)(τ)
    0 ≤ τ ≤ β(aug) || throw(DomainError(τ, "τ must be in [0, β]."))
    return NaN
end
function (aug::MatsubaraConst)(::MatsubaraFreq)
    return one(β(aug))
end

deriv(aug::MatsubaraConst, _=1) = aug
