```@meta
CurrentModule = SparseIR
```

# SparseIR.jl

Documentation for [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl).

There is a [guide](@ref guide) available which explains the intermediate representation and how SparseIR.jl computes it by means of a worked example.

The notation and conventions are those of the [notation page](https://spm-lab.github.io/sparse-ir-doc/src/notation.html), shared by the Julia, Python and Rust libraries. In brief:

  - The statistics have the parity ``ζ`` (`SparseIR.zeta`): 1 for fermions, 0 for bosons; a function of imaginary time obeys ``G(τ + β) = (-1)^ζ G(τ)``.
  - Matsubara frequencies are ``ν = nπ/β`` with the reduced frequency ``n``, odd for fermions and even for bosons; the functions of Matsubara frequency take ``n`` or a [`MatsubaraFreq`](@ref).
  - ``G(\mathrm{i}ν) = ∫_0^β dτ\, e^{\mathrm{i}ντ} G(τ)``, and ``G_l = -S_l ∫ dω\, ρ(ω) V_l(ω)`` with ``ρ = A`` for fermions and ``ρ = A/\tanh(βω/2)`` for bosons.
  - The basis functions accept ``τ ∈ [-β, β]``; `0.0` is read as ``0^+``, `β` as ``β^-``, `-0.0` as ``0^-`` and `-β` as ``(-β)^+``.
  - The index ``l`` of ``U_l``, ``S_l`` and ``V_l`` counts from 0; Julia's `basis.u[l+1]` is ``U_l``.

For listings of all documented names, see [Public names index](@ref) and the [Private names index](@ref).

