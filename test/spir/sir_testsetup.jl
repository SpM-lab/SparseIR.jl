@testsetup module SIRTestSetup

# Shared by the test items that declare `setup=[SIRTestSetup]`:
#
#   * a per-process cache of SVEs and bases (the SVE dominates the run time),
#   * closed forms that do not use libsparseir,
#   * quadrature and array-view helpers,
#   * the oracle checks of the design spec (2026-09-23, §5 and §12.4).
#
# Each oracle check returns the largest ratio error/atol over all points it
# visits, so a test reads `@test o1_tau_ratio(...) <= 1` and a failure prints
# how far off the result is. The reference formulas are keyword arguments so
# that a development script can pass deliberately wrong ones and confirm the
# checks reject them. The same parameters and tolerances are used by the
# Python suite (tests/test_oracle.py in SpM-lab/sparse-ir).

using SparseIR
using LinearAlgebra: SymTridiagonal, eigen, norm

export get_sve, get_basis, gtau_pole, giv_pole, freq_of, gauss_legendre_panels,
       strided_views, poles_for, B1, B2, HARD,
       o1_tau_ratio, o1_matsubara_ratio, o2_ratio, o3_ratios, o4_dlr_ratio

# (β, ωmax, ε)
const B1 = (10.0, 1.0, 1e-10)
const B2 = (1000.0, 1.0, 1e-10)
const HARD = Dict("R1" => (1.0, 0.1, 1e-6), "R2" => (1000.0, 100.0, 1e-8),
    "R3" => (10.0, 1.0, 1e-15), "R4" => (10.0, 1.0, 1e-6))

const LOCK = ReentrantLock()
const SVES = Dict{Tuple{Float64,Float64},SparseIR.SVEResult{LogisticKernel}}()
const BASES = Dict{Tuple{SparseIR.Statistics,Float64,Float64,Float64},FiniteTempBasis}()

"""
    get_sve(Λ, ε)

SVE of `LogisticKernel(Λ)`, computed once per process.
"""
function get_sve(Λ::Real, ε::Real)
    lock(LOCK) do
        get!(SVES, (Float64(Λ), Float64(ε))) do
            SparseIR.SVEResult(LogisticKernel(Float64(Λ)), Float64(ε))
        end
    end
end

"""
    get_basis(stat, β, ωmax, ε)

`FiniteTempBasis` built once per process from the shared SVE. Test items must
not mutate the returned object.
"""
function get_basis(stat::SparseIR.Statistics, β::Real, ωmax::Real, ε::Real)
    lock(LOCK) do
        get!(BASES, (stat, Float64(β), Float64(ωmax), Float64(ε))) do
            sve = get_sve(β * ωmax, ε)
            FiniteTempBasis(stat, Float64(β), Float64(ωmax), Float64(ε);
                kernel=sve.kernel, sve_result=sve)
        end
    end
end

"""
    gtau_pole(τ, ω0, β)

`G(τ)` of a single pole at `ω0` for the logistic kernel used by both statistics,
`-exp(-τ ω0) / (1 + exp(-β ω0))`, in a form that cannot overflow.
"""
gtau_pole(τ, ω0, β) = ω0 ≥ 0 ? -exp(-τ * ω0) / (1 + exp(-β * ω0)) :
                      -exp((β - τ) * ω0) / (1 + exp(β * ω0))

"""
    giv_pole(stat, n, ω0, β)

`Ĝ(iν_n) = ∫₀^β e^{iν_n τ} G(τ) dτ` of the same pole: `1/(iν - ω0)` for fermions and
`tanh(β ω0 / 2)/(iν - ω0)` for bosons.
"""
giv_pole(::Fermionic, n::Integer, ω0, β) = 1 / (im * π * n / β - ω0)
giv_pole(::Bosonic, n::Integer, ω0, β) = tanh(β * ω0 / 2) / (im * π * n / β - ω0)

freq_of(::Fermionic, n::Integer) = FermionicFreq(n)
freq_of(::Bosonic, n::Integer) = BosonicFreq(n)

# 2/β keeps tanh(β ω0 / 2) away from ±1, so that the fermionic and bosonic
# closed forms differ even at β = 1000.
poles_for(β, ωmax) = (-0.8ωmax, 0.3ωmax, 2 / β)

"""
    gauss_legendre_panels(a, b; npanels=400, order=16)

Nodes and weights of a composite Gauss–Legendre rule on `[a, b]`. The nodes of
each panel come from the Golub–Welsch eigenvalue problem, not from SparseIR.
"""
function gauss_legendre_panels(a::Real, b::Real; npanels::Integer=400, order::Integer=16)
    off = [k / sqrt(4k^2 - 1) for k in 1:(order - 1)]
    x0, V = eigen(SymTridiagonal(zeros(order), off))
    w0 = 2 .* V[1, :] .^ 2
    h = (b - a) / npanels
    xs = Float64[]
    ws = Float64[]
    for i in 0:(npanels - 1)
        lo = a + i * h
        append!(xs, lo .+ h / 2 .* (x0 .+ 1))
        append!(ws, h / 2 .* w0)
    end
    return xs, ws
end

"""
    strided_views(a)

`label => view` pairs that hold the values of `a` but are not `Array`s: a view
with stride 2, a view with negative stride along the first dimension and a lazily
permuted wrapper.
"""
function strided_views(a::AbstractArray{T,N}) where {T,N}
    idx = ntuple(d -> 1:2:(2 * size(a, d)), N)
    padded = zeros(T, (2 .* size(a))...)
    padded[idx...] .= a
    rev = reverse(a; dims=1)
    reversed = view(rev, reverse(axes(rev, 1)), ntuple(_ -> Colon(), N - 1)...)
    permuted = N == 1 ? vec(transpose(reshape(collect(a), 1, :))) :
               PermutedDimsArray(permutedims(a, N:-1:1), N:-1:1)
    views = ["strided view" => view(padded, idx...), "reversed view" => reversed,
        "permuted wrapper" => permuted]
    for (label, v) in views
        v isa Array && error("$label is an Array")
        v == a || error("$label does not hold the values of a")
    end
    return views
end

ratio(got, ref, atol) = isempty(got) ? 0.0 : maximum(abs, got .- ref) / atol

"""
    o1_tau_ratio(stat, β, ωmax, ε; gtau=gtau_pole)

O1 in τ (T-ε): `G_l = -s_l v_l(ω0)` evaluated through `u` and through
`TauSampling` must reproduce `gtau(τ, ω0, β)`, and `G(0) + G(β) = -1`.
"""
function o1_tau_ratio(stat, β, ωmax, ε; gtau=gtau_pole)
    basis = get_basis(stat, β, ωmax, ε)
    smpl = TauSampling(basis)
    τs = [0.0, β / 7, β / 2, β]
    worst = 0.0
    for ω0 in poles_for(β, ωmax)
        gl = -basis.s .* basis.v(ω0)
        ref = gtau.(τs, ω0, β)
        got = transpose(basis.u(τs)) * gl
        tol = 300ε * maximum(abs, ref)
        worst = max(worst, ratio(got, ref, tol),
            ratio(evaluate(smpl, gl), gtau.(sampling_points(smpl), ω0, β), tol),
            ratio(got[1] + got[end], -1.0, 300ε))
    end
    return worst
end

"""
    o1_matsubara_ratio(stat, β, ωmax, ε; giv=giv_pole)

O1 in Matsubara frequency (T-ε), at the default points for both values of
`positive_only` and at `±ζ, ±(20 + ζ)`, through `uhat` and `MatsubaraSampling`.
"""
function o1_matsubara_ratio(stat, β, ωmax, ε; giv=giv_pole)
    basis = get_basis(stat, β, ωmax, ε)
    ζ = SparseIR.zeta(stat)
    worst = 0.0
    for positive_only in (false, true)
        smpl = MatsubaraSampling(basis; positive_only)
        points = Int.(sampling_points(smpl))
        ns = unique(vcat(points, [ζ, -ζ, 20 + ζ, -(20 + ζ)]))
        U = basis.uhat(freq_of.(Ref(stat), ns))
        for ω0 in poles_for(β, ωmax)
            gl = -basis.s .* basis.v(ω0)
            ref = giv.(Ref(stat), ns, ω0, β)
            tol = 300ε * maximum(abs, ref)
            worst = max(worst, ratio(transpose(U) * gl, ref, tol),
                ratio(evaluate(smpl, gl), giv.(Ref(stat), points, ω0, β), tol))
        end
    end
    return worst
end

"""
    o2_ratio(stat; sign=1, shift=0)

O2 (T-m): `uhat_l(n) == ∫₀^β exp(iπnτ/β) u_l(τ) dτ` for `|n| ≤ 20Λ` on basis B1.
From `n_asymp = 40Λ` on, the backend uses an asymptotic series that is wrong for
`l ≡ 1, 2 (mod 4)` (SpM-lab/sparse-ir-rs#265); that regime is covered by the
`@test_broken` items in oracle_tests.jl. `sign` and `shift` exist only to let a
development script break the reference.
"""
function o2_ratio(stat; sign=1, shift=0)
    β, ωmax, ε = B1
    basis = get_basis(stat, β, ωmax, ε)
    xs, ws = gauss_legendre_panels(0.0, β)
    U = basis.u(xs)
    L = length(basis)
    ls = unique([1, 2, L ÷ 2 + 1, L])
    ns = stat isa Fermionic ? [1, -1, 11, -11, 101, -101] : [0, 10, -10, 100, -100]
    worst = 0.0
    for n in ns
        ref = U[ls, :] * (cis.(sign * π * (n + shift) .* xs ./ β) .* ws)
        worst = max(worst, ratio(basis.uhat(freq_of(stat, n))[ls], ref, 1e-10 * sqrt(β)))
    end
    return worst
end

"""
    o3_ratios(stat; reflect=true, swap_reim=false, swap_periodic=false)

O3 on basis B1, as a named tuple of ratios: `u_l(β - τ) = (-1)^l u_l(τ)` and
`v_l(-ω) = (-1)^l v_l(ω)` (T-ε), `uhat_l(-n) = conj(uhat_l(n))` (T-m), the vanishing
real or imaginary part of `uhat_l` (T-ε) and the (anti)periodic extension to
negative τ (T-m). `l` is 0-based in the formulas and `n` stays below 20Λ (see
`o2_ratio`). The keywords exist only to let a development script break them.
"""
function o3_ratios(stat; reflect=true, swap_reim=false, swap_periodic=false)
    β, ωmax, ε = B1
    basis = get_basis(stat, β, ωmax, ε)
    L = length(basis)
    sgn = [(-1.0)^(l - 1) for l in 1:L]
    τ = collect(range(0, β; length=101))
    U = basis.u(τ)
    Ur = basis.u(reflect ? β .- τ : τ)
    u_ratio = maximum(l -> ratio(Ur[l, :], sgn[l] .* U[l, :], 300ε * maximum(abs, U[l, :])), 1:L)
    ω = collect(range(-ωmax, ωmax; length=101))
    V = basis.v(ω)
    Vr = basis.v(-ω)
    v_ratio = maximum(l -> ratio(Vr[l, :], sgn[l] .* V[l, :], 300ε * maximum(abs, V[l, :])), 1:L)
    ζ = SparseIR.zeta(stat)
    ns = [ζ, 2 + ζ, 10 + ζ, 100 + ζ]
    Up = basis.uhat(freq_of.(Ref(stat), ns))
    Um = basis.uhat(freq_of.(Ref(stat), -ns))
    conj_ratio = ratio(Um, conj.(Up), 1e-10 * sqrt(β))
    # u_l(β - τ) = (-1)^l u_l(τ) and exp(iν_n β) = -1 (F), +1 (B): for fermions
    # Re uhat_l vanishes for even l and Im uhat_l for odd l; for bosons the
    # other way round.
    real_vanishes(l) = xor(iseven(l - 1) == (stat isa Fermionic), swap_reim)
    vanishing = [real_vanishes(l) ? real(Up[l, k]) : imag(Up[l, k]) for l in 1:L, k in eachindex(ns)]
    reim_ratio = ratio(vanishing, 0.0, 300ε * sqrt(β))
    inner = τ[2:(end - 1)]
    s = xor(stat isa Fermionic, swap_periodic) ? -1.0 : 1.0
    ref = s .* basis.u(β .- inner)
    periodic_ratio = ratio(basis.u(-inner), ref, 1e-10 * maximum(abs, ref))
    return (; u=u_ratio, v=v_ratio, conj=conj_ratio, reim=reim_ratio, periodic=periodic_ratio)
end

"""
    o4_dlr_ratio(stat; gtau=gtau_pole, giv=giv_pole)

O4 for the DLR on basis B1: `dlr.u` and `dlr.uhat` have the closed forms of a
single pole (T-m), `to_IR(c) = -s .* (v(ω_p) c)` (T-m), and DLR and IR agree in τ
for a single pole (T-ε).
"""
function o4_dlr_ratio(stat; gtau=gtau_pole, giv=giv_pole)
    β, ωmax, ε = B1
    basis = get_basis(stat, β, ωmax, ε)
    dlr = DiscreteLehmannRepresentation(basis)
    poles = SparseIR.get_poles(dlr)
    τ = [0.0, 1.3, 7.0, β]
    ref_u = [gtau(t, ω, β) for ω in poles, t in τ]
    ζ = SparseIR.zeta(stat)
    ns = [ζ, 6 + ζ, -(6 + ζ)]
    ref_uhat = [giv(stat, n, ω, β) for ω in poles, n in ns]
    c = [sin(3.0k) for k in eachindex(poles)]
    ref_gl = -basis.s .* (basis.v(poles) * c)
    gl = -basis.s .* basis.v(0.3ωmax)
    ref_τ = transpose(basis.u(τ)) * gl
    return max(ratio(dlr.u(τ), ref_u, 1e-10 * maximum(abs, ref_u)),
        ratio(dlr.uhat(freq_of.(Ref(stat), ns)), ref_uhat, 1e-10 * maximum(abs, ref_uhat)),
        ratio(to_IR(dlr, c), ref_gl, 1e-10 * maximum(abs, ref_gl)),
        ratio(transpose(dlr.u(τ)) * from_IR(dlr, gl), ref_τ, 300ε * maximum(abs, ref_τ)))
end

end
