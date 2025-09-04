"""
    Statistics(zeta)

Abstract type for quantum statistics (fermionic/bosonic/etc.)
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
Fermionic statistics.
"""
struct Fermionic <: Statistics end

"""
Bosonic statistics.
"""
struct Bosonic <: Statistics end


# Convert Julia statistics to C API constants
_statistics_to_c(::Type{Fermionic}) = SPIR_STATISTICS_FERMIONIC
_statistics_to_c(::Type{Bosonic}) = SPIR_STATISTICS_BOSONIC
_statistics_from_c(s::Cint) = s == SPIR_STATISTICS_FERMIONIC ? Fermionic() : Bosonic()


"""
    MatsubaraFreq(n)

Prefactor `n` of the Matsubara frequency `ω = n*π/β`

Struct representing the Matsubara frequency ω entering the Fourier transform of
a propagator G(τ) on imaginary time τ to its Matsubara equivalent Ĝ(iω) on the
imaginary-frequency axis:

            β
    Ĝ(iω) = ∫  dτ exp(iωτ) G(τ)      with    ω = n π/β,
            0

where β is inverse temperature and by convention we include the imaginary unit
in the frequency argument, i.e, Ĝ(iω). The frequencies depend on the
statistics of the propagator, i.e., we have that:

    G(τ - β) = ± G(τ)

where + is for bosons and - is for fermions. The frequencies are restricted
accordingly.

  - Bosonic frequency (`S == Fermionic`): `n` even (periodic in β)
  - Fermionic frequency (`S == Bosonic`): `n` odd (anti-periodic in β)
"""
struct MatsubaraFreq{S<:Statistics} <: Number
    n::Int

    MatsubaraFreq(stat::Statistics, n::Integer) = new{typeof(stat)}(n)

    function MatsubaraFreq{S}(n::Integer) where {S<:Statistics}
        allowed(S, n) || throw(DomainError(n, "Frequency $(n)π/β is not $S"))
        return new{S}(n)
    end
end

const BosonicFreq   = MatsubaraFreq{Bosonic}
const FermionicFreq = MatsubaraFreq{Fermionic}

MatsubaraFreq(n::Integer) = MatsubaraFreq(Statistics(mod(n, 2)), n)

Base.broadcastable(s::Statistics) = Ref(s)
zeta(::Fermionic) = 1
zeta(::Bosonic)   = 0

allowed(::Type{Fermionic}, a::Integer) = isodd(a)
allowed(::Type{Bosonic}, a::Integer)   = iseven(a)

Base.:+(::Fermionic, ::Bosonic)   = Fermionic()
Base.:+(::Bosonic, ::Fermionic)   = Fermionic()
Base.:+(::Fermionic, ::Fermionic) = Bosonic()
Base.:+(::Bosonic, ::Bosonic)     = Bosonic()


statistics(::MatsubaraFreq{S}) where {S} = S()

"""
Get prefactor `n` for the Matsubara frequency `ω = n*π/β`
"""
Base.Integer(a::MatsubaraFreq) = a.n

"""
Get prefactor `n` for the Matsubara frequency `ω = n*π/β`
"""
Base.Int(a::MatsubaraFreq) = a.n

"""
Get value of the Matsubara frequency `ω = n*π/β`
"""
value(a::MatsubaraFreq, β::Real) = Int(a) * (π / β)

"""
Get complex value of the Matsubara frequency `iω = iπ/β * n`
"""
valueim(a::MatsubaraFreq, β::Real) = 1im * value(a, β)

"""
Get statistics `ζ` for Matsubara frequency `ω = (2*m+ζ)*π/β`
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
