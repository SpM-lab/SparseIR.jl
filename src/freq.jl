export
    MatsubaraFreq,
    BosonicFreq,
    FermionicFreq,
    pioverbeta

"""
    Statistics(zeta)

Abstract type for quantum statistics (fermionic/bosonic/etc.)
"""
abstract type Statistics end

Statistics(zeta::Bool) = zeta ? Fermionic() : Bosonic()

"""Fermionic statistics."""
struct Fermionic <: Statistics end

"""Bosonic statistics."""
struct Bosonic <: Statistics end

zeta(::Fermionic) = true
zeta(::Bosonic) = false

Base.:+(::Fermionic, ::Bosonic)   = Fermionic()
Base.:+(::Bosonic,   ::Fermionic) = Fermionic()
Base.:+(::Fermionic, ::Fermionic) = Bosonic()
Base.:+(::Bosonic,   ::Bosonic)   = Bosonic()


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
struct MatsubaraFreq{S <: Statistics} <: Number
    stat::S
    n::Int

    MatsubaraFreq(stat::Statistics, n::Integer) = new{typeof(stat)}(stat, n)

    function MatsubaraFreq{S}(n::Integer) where {S <: Statistics}
        stat = S()
        if isodd(n) != zeta(stat)
            throw(ArgumentError("Frequency $(n)π/β is not $stat"))
        end
        new{S}(stat, n)
    end
end

const BosonicFreq = MatsubaraFreq{Bosonic}

const FermionicFreq = MatsubaraFreq{Fermionic}

MatsubaraFreq(n::Integer) = MatsubaraFreq(Statistics(isodd(n)), n)

"""Get prefactor `n` for the Matsubara frequency `ω = n*π/β`"""
Integer(a::MatsubaraFreq) = a.n

"""Get prefactor `n` for the Matsubara frequency `ω = n*π/β`"""
Int(a::MatsubaraFreq) = a.n

"""Get statistics `ζ` for Matsubara frequency `ω = (2*m+ζ)*π/β`"""
zeta(a::MatsubaraFreq) = zeta(a.stat)

Base.:+(a::MatsubaraFreq, b::MatsubaraFreq) =
    MatsubaraFreq(a.stat + b.stat, a.n + b.n)

Base.:-(a::MatsubaraFreq, b::MatsubaraFreq) =
    MatsubaraFreq(a.stat + b.stat, a.n - b.n)

Base.:+(a::MatsubaraFreq) = a

Base.:-(a::MatsubaraFreq) = MatsubaraFreq(a.stat, -a.n)

Base.:*(a::BosonicFreq, c::Integer) = MatsubaraFreq(a.stat, a.n * c)

Base.:*(a::FermionicFreq, c::Integer) = MatsubaraFreq(a.n * c)

Base.:*(c::Integer, a::MatsubaraFreq) = a * c

Base.zero(::MatsubaraFreq) = MatsubaraFreq(0)

Base.iszero(self::MatsubaraFreq) = self.n == 0

Base.isless(a::MatsubaraFreq, b::MatsubaraFreq) = isless(a.n, b.n)

Base.isequal(a::MatsubaraFreq, b::MatsubaraFreq) = isequal(a.n, b.n)

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
struct FreqRange{A<:Statistics} <: OrdinalRange{MatsubaraFreq{A}, BosonicFreq}
    start::MatsubaraFreq{A}
    stop::MatsubaraFreq{A}

    function FreqRange(start::MatsubaraFreq{A}, stop::MatsubaraFreq{A}) where {A}
        if stop < start
            stop = start - 2 * pioverbeta
        end
        new{A}(start, stop)
    end
end

Base.first(self::FreqRange) = self.start

Base.last(self::FreqRange) = self.stop

Base.step(::FreqRange) = BosonicFreq(2)

Base.length(self::FreqRange) = Integer(self.stop - self.start) ÷ 2 + 1

Base.:(:)(start::MatsubaraFreq, stop::MatsubaraFreq) = FreqRange(start, stop)
