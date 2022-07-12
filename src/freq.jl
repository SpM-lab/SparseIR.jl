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
