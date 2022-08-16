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

See also: [`MatsubaraConst`](@ref) for vertex basis [^wallerberger2021], [`TauConst`](@ref), [`TauLinear`](@ref) for multi-point [^shinaoka2018]

[^wallerberger2021]: https://doi.org/10.1103/PhysRevResearch.3.033168
[^shinaoka2018]: https://doi.org/10.1103/PhysRevB.97.205111
"""
struct AugmentedBasis{B<:AbstractBasis} <: AbstractBasis
    basis::B
    augmentations::Any
    u::Any
    uhat::Any
end

function AugmentedBasis(basis::AbstractBasis, augmentations...)
    augmentations = Tuple(augmentation_factory(basis, augmentations...))
    u = AugmentedTauFunction(basis.u, augmentations)
    û = AugmentedMatsubaraFunction(basis.uhat, [n -> hat(aug, n) for aug in augmentations])
    return AugmentedBasis(basis, augmentations, u, û)
end

statistics(basis::AugmentedBasis) = statistics(basis.basis)
naug(basis::AugmentedBasis) = length(basis.augmentations)

function getindex(basis::AugmentedBasis, index)
    stop = range_to_size(index)
    stop > naug(basis) || error("Cannot truncate to only augmentation.")
    return AugmentedBasis(basis.basis[begin:(stop - naug(basis))], basis.augmentations)
end

Base.size(basis::AugmentedBasis) = (length(basis), )
Base.length(basis::AugmentedBasis) = naug(basis) + length(basis.basis)
significance(basis::AugmentedBasis) = significance(basis.basis)
accuracy(basis::AugmentedBasis) = accuracy(basis.basis)
Λ(basis::AugmentedBasis) = Λ(basis.basis)
β(basis::AugmentedBasis) = β(basis.basis)
ωmax(basis::AugmentedBasis) = ωmax(basis.basis)

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



abstract type AbstractAugmentedFunction <: Function end

struct AugmentedFunction <: AbstractAugmentedFunction
    fbasis::Any
    faug::Any
end

augmentedfunction(a::AugmentedFunction) = a 

fbasis(a::AbstractAugmentedFunction) = augmentedfunction(a).fbasis
faug(a::AbstractAugmentedFunction) = augmentedfunction(a).faug
naug(a::AbstractAugmentedFunction) = length(faug(a))

Base.length(a::AbstractAugmentedFunction) = naug(a) + length(fbasis(a))
Base.size(a::AbstractAugmentedFunction) = (length(a), )

function (a::AbstractAugmentedFunction)(x)
    fbasis_x = fbasis(a)(x)
    faug_x = [faug_l(x) for faug_l in faug(a)]
    return fbasis_x .+ faug_x
end
function (a::AbstractAugmentedFunction)(x::AbstractArray)
    fbasis_x = fbasis(a)(x)
    faug_x = (faug_l.(x) for faug_l in faug(a))
    return sum(fbasis_x .+ transpose(faug_xi) for faug_xi in faug_x)
end

function Base.getindex(a::AbstractAugmentedFunction, r::AbstractRange)
    stop = range_to_size(r)
    stop > naug(a) || error("Don't truncate to only augmentation")
    return AugmentedFunction(fbasis(a)[begin:(stop-naug(a))], faug(a))
end
function Base.getindex(a::AbstractAugmentedFunction, l::Integer)
    if l < naug(a)
        return faug(a)[l]
    else
        return fbasis(a)[l-naug(a)]
    end
end

    
struct AugmentedTauFunction <: AbstractAugmentedFunction
    a::AugmentedFunction
end

augmentedfunction(aτ::AugmentedTauFunction) = aτ.a

AugmentedTauFunction(fbasis, faug) = AugmentedTauFunction(AugmentedFunction(fbasis, faug))

xmin(aτ::AugmentedTauFunction) = fbasis(aτ).xmin
xmax(aτ::AugmentedTauFunction) = fbasis(aτ).xmax

function deriv(aτ::AugmentedTauFunction, n=1)
    dbasis = deriv(fbasis(aτ), n)
    daug = [deriv(faug_l, n) for faug_l in faug(aτ)]
    return AugmentedTauFunction(dbasis, daug)
end


struct AugmentedMatsubaraFunction <: AbstractAugmentedFunction
    a::AugmentedFunction
end

augmentedfunction(amat::AugmentedMatsubaraFunction) = amat.a

AugmentedMatsubaraFunction(fbasis, faug) = AugmentedMatsubaraFunction(AugmentedFunction(fbasis, faug))

zeta(amat::AugmentedMatsubaraFunction) = zeta(fbasis(amat))


abstract type AbstractAugmentation end

struct TauConst <: AbstractAugmentation
    β::Float64
    function TauConst(β)
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new(β)
    end
end

function create(::Type{TauConst}, basis::AbstractBasis)
    statistics(basis)::Bosonic
    return TauConst(β(basis))
end

function (aug::TauConst)(τ)
    0 ≤ τ ≤ aug.β || throw(DomainError(τ, "τ must be in [0, β]."))
    return 1 / √(aug.β)
end

function deriv(aug::TauConst, n=1)
    iszero(n) && return aug
    return τ -> 0.0
end

function hat(aug::TauConst, n::BosonicFreq)
    iszero(n) || return 0.0
    return √(aug.β)
end


struct TauLinear <: AbstractAugmentation
    β::Float64
    norm::Float64
    function TauLinear(β)
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        norm = √(3 / β)
        return new(β, norm)
    end
end

function create(::Type{TauLinear}, basis::AbstractBasis)
    statistics(basis)::Bosonic
    return TauLinear(β(basis))
end

function (aug::TauLinear)(τ)
    0 ≤ τ ≤ aug.β || throw(DomainError(τ, "τ must be in [0, β]."))
    x = 2 / aug.β * τ - 1
    return aug.norm * x
end

function deriv(aug::TauLinear, n=1)
    iszero(n) && return aug
    isone(n) && return τ -> aug.norm * 2 / aug.β
    return τ -> 0.0
end

function hat(aug::TauLinear, n::BosonicFreq)
    inv_w = value(n, aug.β)
    inv_w = iszero(n) ? inv_w : 1 / inv_w
    return aug.norm * 2/im * inv_w
end


struct MatsubaraConst <: AbstractAugmentation
    β::Float64
    function MatsubaraConst(β)
        β > 0 || throw(DomainError(β, "Temperature must be positive."))
        return new(β)
    end
end

create(::Type{MatsubaraConst}, basis::AbstractBasis) = MatsubaraConst(β(basis))

function (aug::MatsubaraConst)(τ)
    0 ≤ τ ≤ aug.β || throw(DomainError(τ, "τ must be in [0, β]."))
    return NaN
end

deriv(aug::MatsubaraConst, n=1) = aug

function hat(::MatsubaraConst, ::MatsubaraFreq)
    return 1.0
end


augmentation_factory(basis::AbstractBasis, augs...) = 
    Iterators.map(augs) do aug
        if aug isa AbstractAugmentation
            return aug
        else
            return create(aug, basis)
        end
    end

create(aug, )