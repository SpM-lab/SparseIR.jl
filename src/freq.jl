"""
    Statistics(zeta)

Abstract type for quantum statistics. The argument is the parity `ζ` of the
statistics (see [`zeta`](@ref)): `Statistics(1)` is `Fermionic()` and
`Statistics(0)` is `Bosonic()`; any other value throws `DomainError`.
"""
abstract type Statistics end

function Statistics(zeta::Integer)
    if isone(zeta)
        return Fermionic()
    elseif iszero(zeta)
        return Bosonic()
    else
        throw(DomainError(zeta, "does not correspond to known statistics"))
    end
end

"""
Fermionic statistics, parity `ζ = 1`: a function of imaginary time is
anti-periodic, `G(τ + β) = -G(τ)`, and its Matsubara frequencies are `ν = nπ/β`
with odd `n`.
"""
struct Fermionic <: Statistics end

"""
Bosonic statistics, parity `ζ = 0`: a function of imaginary time is periodic,
`G(τ + β) = G(τ)`, and its Matsubara frequencies are `ν = nπ/β` with even `n`.
"""
struct Bosonic <: Statistics end

# Convert Julia statistics to C API constants
_statistics_to_c(::Type{Fermionic}) = SPIR_STATISTICS_FERMIONIC
_statistics_to_c(::Type{Bosonic}) = SPIR_STATISTICS_BOSONIC
_statistics_from_c(s::Cint) = s == SPIR_STATISTICS_FERMIONIC ? Fermionic() : Bosonic()

"""
    MatsubaraFreq(n)

Matsubara frequency `ν = nπ/β`, stored as its reduced frequency `n`.

Struct representing the Matsubara frequency ν entering the Fourier transform of
a propagator G(τ) on imaginary time τ to its Matsubara equivalent G(iν) on the
imaginary-frequency axis:

            β
    G(iν) = ∫  dτ exp(iντ) G(τ)      with    ν = n π/β,
            0

    G(τ) = (1/β) Σ_ν exp(-iντ) G(iν),

where β is inverse temperature and by convention we include the imaginary unit
in the frequency argument, i.e., G(iν); the argument tells the function and its
transform apart. The frequencies depend on the statistics of the propagator
through its parity ζ (1 for fermions, 0 for bosons, see [`zeta`](@ref)):

    G(τ + β) = (-1)^ζ G(τ).

The reduced frequency `n` is an integer with `n ≡ ζ (mod 2)`:

  - Bosonic frequency (`S == Bosonic`): `n` even (periodic in β)
  - Fermionic frequency (`S == Fermionic`): `n` odd (anti-periodic in β)

`MatsubaraFreq(n)` takes the statistics from the parity of `n`;
`MatsubaraFreq{S}(n)`, [`FermionicFreq`](@ref) and [`BosonicFreq`](@ref) throw
`DomainError` for the wrong parity.
"""
struct MatsubaraFreq{S<:Statistics} <: Number
    n::Int

    MatsubaraFreq(stat::Statistics, n::Integer) = new{typeof(stat)}(n)

    function MatsubaraFreq{S}(n::Integer) where {S<:Statistics}
        allowed(S, n) || throw(DomainError(n, "Frequency $(n)π/β is not $S"))
        return new{S}(n)
    end
end

"""
    BosonicFreq(n)

Bosonic Matsubara frequency `n π/β` with even `n`; an alias of
`MatsubaraFreq{Bosonic}`. An odd `n` throws `DomainError`.
"""
const BosonicFreq = MatsubaraFreq{Bosonic}

"""
    FermionicFreq(n)

Fermionic Matsubara frequency `n π/β` with odd `n`; an alias of
`MatsubaraFreq{Fermionic}`. An even `n` throws `DomainError`.
"""
const FermionicFreq = MatsubaraFreq{Fermionic}

MatsubaraFreq(n::Integer) = MatsubaraFreq(Statistics(mod(n, 2)), n)

Base.broadcastable(s::Statistics) = Ref(s)
zeta(::Fermionic) = 1
zeta(::Bosonic) = 0

allowed(::Type{Fermionic}, a::Integer) = isodd(a)
allowed(::Type{Bosonic}, a::Integer)   = iseven(a)

Base.:+(::Fermionic, ::Bosonic)   = Fermionic()
Base.:+(::Bosonic, ::Fermionic)   = Fermionic()
Base.:+(::Fermionic, ::Fermionic) = Bosonic()
Base.:+(::Bosonic, ::Bosonic)     = Bosonic()

statistics(::MatsubaraFreq{S}) where {S} = S()

"""
    Integer(freq::MatsubaraFreq)

The reduced frequency `n` of the Matsubara frequency `ν = nπ/β`.
"""
Base.Integer(a::MatsubaraFreq) = a.n

"""
    Int(freq::MatsubaraFreq)

The reduced frequency `n` of the Matsubara frequency `ν = nπ/β`.
"""
Base.Int(a::MatsubaraFreq) = a.n

"""
    value(freq::MatsubaraFreq, β)

The Matsubara frequency `ν = nπ/β` as a real number.
"""
value(a::MatsubaraFreq, β::Real) = Int(a) * (π / β)

"""
    valueim(freq::MatsubaraFreq, β)

The imaginary frequency `iν = i nπ/β` as a complex number.
"""
valueim(a::MatsubaraFreq, β::Real) = 1im * value(a, β)

"""
    zeta(stat::Statistics)
    zeta(freq::MatsubaraFreq)

Parity `ζ` of the statistics: `1` for `Fermionic()` and `0` for `Bosonic()`.
A shift by β multiplies a function of imaginary time by `(-1)^ζ`, and the
reduced frequency of a Matsubara frequency is `n = 2m + ζ`, where `m` is the
ordinary Matsubara index, i.e. `ν = (2m + ζ)π/β`.
"""
zeta(a::MatsubaraFreq) = zeta(statistics(a))

Base.:+(a::MatsubaraFreq, b::MatsubaraFreq) = MatsubaraFreq(statistics(a) + statistics(b), a.n + b.n)
Base.:-(a::MatsubaraFreq, b::MatsubaraFreq) = MatsubaraFreq(statistics(a) + statistics(b), a.n - b.n)
Base.:+(a::MatsubaraFreq)                   = a
Base.:-(a::MatsubaraFreq)                   = MatsubaraFreq(statistics(a), -a.n)
Base.:*(a::BosonicFreq, c::Integer)         = BosonicFreq(a.n * c)
Base.:*(a::FermionicFreq, c::Integer)       = MatsubaraFreq(a.n * c)
Base.:*(c::Integer, a::MatsubaraFreq)       = a * c

Base.:(==)(::FermionicFreq, ::BosonicFreq)      = false
Base.:(==)(::BosonicFreq, ::FermionicFreq)      = false
Base.sign(a::MatsubaraFreq)                     = sign(a.n)
Base.zero(::MatsubaraFreq)                      = BosonicFreq(0)
Base.iszero(::FermionicFreq)                    = false
Base.iszero(a::BosonicFreq)                     = iszero(a.n)
Base.isless(a::MatsubaraFreq, b::MatsubaraFreq) = isless(a.n, b.n)

# This is to get rid of the weird "promotion failed to change any of the types"
# errors you get when mixing frequencies and numbers. These originate from the
# `promote_rule(<:Number, <:Number) = Number` default, together with the fact
# that `@(x::Number, y::Number) = @(promote(x,y)...)` for most operations.
# Let's make this error more explicit instead.
Base.promote_rule(::Type{<:MatsubaraFreq}, ::Type{<:MatsubaraFreq}) = MatsubaraFreq
function Base.promote_rule(::Type{T1}, ::Type{T2}) where {T1<:MatsubaraFreq,T2<:Number}
    throw(ArgumentError("""
        Will not promote (automatically convert) $T2 and $T1.

        You were probably mixing a number ($T2) and a Matsubara frequency ($T1)
        in an additive or comparative expression, e.g. `MatsubaraFreq(0) + 1`.
        We disallow this. Please use `MatsubaraFreq(x)` explicitly."""))
end

function Base.show(io::IO, ::MIME"text/plain", a::MatsubaraFreq)
    if a.n == 0
        print(io, "0")
    elseif a.n == 1
        print(io, "π/β")
    elseif a.n == -1
        print(io, "-π/β")
    else
        print(io, a.n, "π/β")
    end
end

"""
    pioverbeta

The fermionic Matsubara frequency `π/β`, `FermionicFreq(1)`. Multiples give
other frequencies: `3 * pioverbeta == FermionicFreq(3)`, `2 * pioverbeta == BosonicFreq(2)`.
"""
const pioverbeta = MatsubaraFreq(1)
Base.oneunit(::MatsubaraFreq) = pioverbeta

Base.rem(a::MatsubaraFreq, b::FermionicFreq) = MatsubaraFreq(rem(a.n, b.n))
Base.rem(a::MatsubaraFreq{S}, b::BosonicFreq) where {S} = MatsubaraFreq{S}(rem(a.n, b.n))
Base.div(a::MatsubaraFreq, b::MatsubaraFreq) = div(a.n, b.n)

# Ranges may only consist of elements of a single type
function Base.:(:)(start::MatsubaraFreq{S}, stop::MatsubaraFreq{S}) where {S}
    start:BosonicFreq(2):stop
end

function frequency_range(len::Integer)
    len > 0 || throw(ArgumentError("Length must be positive"))
    MatsubaraFreq(-(len - 1)):MatsubaraFreq(+(len - 1))
end
