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

Base.broadcastable(s::Statistics) = Ref(s)

"""
Fermionic statistics.
"""
struct Fermionic <: Statistics end

"""
Bosonic statistics.
"""
struct Bosonic <: Statistics end

zeta(::Fermionic) = 1
zeta(::Bosonic)   = 0

allowed(::Fermionic, a::Integer) = isodd(a)
allowed(::Bosonic, a::Integer)   = iseven(a)

Base.:+(::Fermionic, ::Bosonic)   = Fermionic()
Base.:+(::Bosonic, ::Fermionic)   = Fermionic()
Base.:+(::Fermionic, ::Fermionic) = Bosonic()
Base.:+(::Bosonic, ::Bosonic)     = Bosonic()

"""
    MatsubaraFreq(n)

Prefactor `n` of the Matsubara frequency `ω = n*π/β`

Struct representing the Matsubara frequency ω entering the Fourier transform of
a propagator G(τ) on imaginary time τ to its Matsubara equivalent Ĝ(iω) on the
imaginary-frequency axis:

            β
    Ĝ(iω) = ∫  dτ exp(iωτ) G(τ)      with    ω = n π/β,
            0

where β is inverse temperature and  by convention we include the imaginary unit
in the frequency argument, i.e, Ĝ(iω).  The frequencies depend on the
statistics of the propagator, i.e., we have that:

    G(τ - β) = ± G(τ)

where + is for bosons and - is for fermions.  The frequencies are restricted
accordingly.

  - Bosonic frequency (`S == Fermionic`): `n` even (periodic in β)
  - Fermionic frequency (`S == Bosonic`): `n` odd (anti-periodic in β)
"""
struct MatsubaraFreq{S<:Statistics} <: Number
    stat :: S
    n    :: Int

    MatsubaraFreq(stat::Statistics, n::Integer) = new{typeof(stat)}(stat, n)

    function MatsubaraFreq{S}(n::Integer) where {S<:Statistics}
        stat = S()
        if !allowed(stat, n)
            throw(ArgumentError("Frequency $(n)π/β is not $stat"))
        end
        return new{S}(stat, n)
    end
end

const BosonicFreq   = MatsubaraFreq{Bosonic}
const FermionicFreq = MatsubaraFreq{Fermionic}

MatsubaraFreq(n::Integer) = MatsubaraFreq(Statistics(mod(n, 2)), n)

"""
Get prefactor `n` for the Matsubara frequency `ω = n*π/β`
"""
Integer(a::MatsubaraFreq) = a.n

"""
Get prefactor `n` for the Matsubara frequency `ω = n*π/β`
"""
Int(a::MatsubaraFreq) = a.n

"""
Get value of the Matsubara frequency `ω = n*π/β`
"""
value(a::MatsubaraFreq, beta::Real) = Int(a) * (π / beta)

"""
Get complex value of the Matsubara frequency `iω = iπ/β * n`
"""
valueim(a::MatsubaraFreq, beta::Real) = 1im * value(a, beta)

"""
Get statistics `ζ` for Matsubara frequency `ω = (2*m+ζ)*π/β`
"""
zeta(a::MatsubaraFreq) = zeta(a.stat)

Base.:+(a::MatsubaraFreq, b::MatsubaraFreq) = MatsubaraFreq(a.stat + b.stat, a.n + b.n)
Base.:-(a::MatsubaraFreq, b::MatsubaraFreq) = MatsubaraFreq(a.stat + b.stat, a.n - b.n)
Base.:+(a::MatsubaraFreq)                   = a
Base.:-(a::MatsubaraFreq)                   = MatsubaraFreq(a.stat, -a.n)
Base.:*(a::BosonicFreq, c::Integer)         = MatsubaraFreq(a.stat, a.n * c)
Base.:*(a::FermionicFreq, c::Integer)       = MatsubaraFreq(a.n * c)
Base.:*(c::Integer, a::MatsubaraFreq)       = a * c

Base.sign(a::MatsubaraFreq)                     = sign(a.n)
Base.zero(::MatsubaraFreq)                      = MatsubaraFreq(0)
Base.iszero(self::MatsubaraFreq)                = iszero(self.n)
Base.isless(a::MatsubaraFreq, b::MatsubaraFreq) = isless(a.n, b.n)

# This is to get rid of the weird "promotion failed to change any of the types"
# errors you get when mixing frequencies and numbers.  These originate from the
# `promote_rule(<:Number, <:Number) = Number` default, together with the fact
# that `@(x::Number, y::Number) = @(promote(x,y)...)` for most operations.
# Let's make this error more explicit instead.
Base.promote_rule(::Type{<: MatsubaraFreq}, ::Type{<:MatsubaraFreq}) = MatsubaraFreq
Base.promote_rule(::Type{T1}, ::Type{T2}) where {T1<:MatsubaraFreq, T2<:Number} =
    throw(ArgumentError("""
        Will not promote (automatically convert) $T2 and $T1.

        You were probably mixing a number ($T2) and a Matsubara frequency ($T1)
        in an additive or comparative expression, e.g., `MatsubaraFreq(0) + 1`.
        We disallow this.  Please use `$T1(x)` explicitly."""))

function Base.show(io::IO, self::MatsubaraFreq)
    if self.n == 0
        print(io, "0")
    elseif self.n == 1
        print(io, "π/β")
    else
        print(io, self.n, "π/β")
    end
end

const pioverbeta = MatsubaraFreq(1)

Base.oneunit(::MatsubaraFreq) = pioverbeta

"""
Dense grid of frequencies in an implicit representation
"""
struct FreqRange{A<:Statistics} <: OrdinalRange{MatsubaraFreq{A},BosonicFreq}
    start :: MatsubaraFreq{A}
    stop  :: MatsubaraFreq{A}
    step  :: BosonicFreq

    function FreqRange(start::MatsubaraFreq{A}, stop::MatsubaraFreq{A}, step_::BosonicFreq) where {A}
        range = Int(start):Int(step_):Int(stop)
        start = MatsubaraFreq{A}(first(range))
        step_ = BosonicFreq(step(range))
        stop = iszero(length(range)) ? start - step_ : MatsubaraFreq{A}(last(range))
        return new{A}(start, stop, step_)
    end
end

Base.first(self::FreqRange)                          = self.start
Base.last(self::FreqRange)                           = self.stop
Base.step(self::FreqRange)                           = self.step
Base.length(self::FreqRange)                         = Int(last(self) - first(self)) ÷ Int(step(self)) + 1
Base.:(:)(start::MatsubaraFreq, stop::MatsubaraFreq) = start:BosonicFreq(2):stop
Base.:(:)(start::MatsubaraFreq, step::BosonicFreq, stop::MatsubaraFreq) = FreqRange(start, stop, step)
