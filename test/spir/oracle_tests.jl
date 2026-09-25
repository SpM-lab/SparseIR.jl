# Oracle tests: closed forms, definitions and symmetries that do not depend on
# libsparseir internals. The checks themselves live in sir_testsetup.jl and
# return the largest error/atol ratio. Tolerance classes (design spec §5):
#
#   T-ε: atol = 300 ε scale        (IR truncation)
#   T-m: atol = 1e-10 scale        (identities exact for the stored polynomials)
#   T-c: atol = 100 cond eps scale (round trips)

@testitem "oracle O1: single pole in imaginary time" tags=[:julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    @testset "$(nameof(typeof(stat))) $label" for stat in (Fermionic(), Bosonic()),
        (label, params) in ("B1" => B1, "B2" => B2)

        @test o1_tau_ratio(stat, params...) <= 1
    end
end

@testitem "oracle O1: single pole in Matsubara frequency" tags=[:julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    @testset "$(nameof(typeof(stat))) $label" for stat in (Fermionic(), Bosonic()),
        (label, params) in ("B1" => B1, "B2" => B2)

        @test o1_matsubara_ratio(stat, params...) <= 1
    end
end

@testitem "oracle O2: uhat is the Fourier transform of u" tags=[:julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    @testset "$(nameof(typeof(stat)))" for stat in (Fermionic(), Bosonic())
        @test o2_ratio(stat) <= 1
    end
end

@testitem "oracle O2/O4: asymptotic regime of uhat (SpM-lab/sparse-ir-rs#265)" tags=[
    :julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    # From n_asymp = 40Λ on, the backend evaluates uhat through an asymptotic
    # series whose parity is wrong for l ≡ 1, 2 (mod 4). These checks are
    # expected to fail until SpM-lab/sparse-ir-rs#265 is fixed; they will then
    # report an unexpected pass, and `@test_broken` must become `@test`.
    β, ωmax, ε = B1
    @testset "$(nameof(typeof(stat)))" for stat in (Fermionic(), Bosonic())
        basis = get_basis(stat, B1...)
        ζ = SparseIR.zeta(stat)

        n = 80 * round(Int, β * ωmax) + ζ               # twice n_asymp
        xs, ws = gauss_legendre_panels(0.0, β; npanels=1600, order=24)
        ref = basis.u(xs) * (cis.(π * n .* xs ./ β) .* ws)
        @test_broken maximum(abs, basis.uhat(freq_of(stat, n)) .- ref) <= 1e-10 * sqrt(β)

        # i ν_n uhat_l(n) → -(u_l(β) + u_l(0)) (F), u_l(β) - u_l(0) (B), by
        # integrating the definition by parts; the O(1/ν) remainder is below
        # 1e-9 at n ~ 2^40, whereas a truncated index gives an O(1) error.
        nbig = 2^40 + ζ
        ν = π * nbig / β
        u0, uβ = basis.u(0.0), basis.u(β)
        limit = stat isa Fermionic ? -(uβ .+ u0) : uβ .- u0
        @test_broken maximum(abs, im * ν .* basis.uhat(freq_of(stat, nbig)) .- limit) <=
                     1e-6 * maximum(abs, u0)
    end
end

@testitem "oracle O3: symmetries" tags=[:julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    @testset "$(nameof(typeof(stat)))" for stat in (Fermionic(), Bosonic())
        r = o3_ratios(stat)
        @test r.u <= 1
        @test r.v <= 1
        @test r.conj <= 1
        @test r.reim <= 1
        @test r.periodic <= 1
    end
end

@testitem "oracle O4: roots, orthonormality and default sampling points" tags=[
    :julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR
    using LinearAlgebra: Diagonal, I

    β, ωmax, ε = B1
    @testset "$(nameof(typeof(stat)))" for stat in (Fermionic(), Bosonic())
        basis = get_basis(stat, B1...)
        L = length(basis)

        grid = collect(range(0, β; length=20001))[2:(end - 1)]
        U = basis.u(grid)
        changes = [count(i -> sign(U[l, i]) != sign(U[l, i + 1]), 1:(length(grid) - 1))
                   for l in 1:L]
        @test changes == collect(0:(L - 1))

        xs, ws = gauss_legendre_panels(0.0, β)
        Ux = basis.u(xs)
        @test maximum(abs, Ux * Diagonal(ws) * transpose(Ux) - I) <= 1e-12
        xv, wv = gauss_legendre_panels(-ωmax, ωmax)
        Vx = basis.v(xv)
        @test maximum(abs, Vx * Diagonal(wv) * transpose(Vx) - I) <= 1e-12

        # Only structure: the values depend on the backend version.
        τs = SparseIR.default_tau_sampling_points(basis)
        @test length(τs) == L
        @test issorted(τs)
        @test all(0 .< τs .< β)
        full = SparseIR.default_matsubara_sampling_points(basis)
        nonneg = SparseIR.default_matsubara_sampling_points(basis; positive_only=true)
        @test length(full) >= L
        @test all(n -> mod(n, 2) == SparseIR.zeta(stat), full)
        @test sort(full) == sort(-full)
        @test sort(nonneg) == sort(filter(>=(0), full))
    end
end

@testitem "oracle O4: DLR functions have closed forms" tags=[:julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    @testset "$(nameof(typeof(stat)))" for stat in (Fermionic(), Bosonic())
        @test o4_dlr_ratio(stat) <= 1
    end
end

@testitem "oracle O5: hard regimes" tags=[:julia, :oracle] setup=[SIRTestSetup] begin
    using Test
    using SparseIR
    using LinearAlgebra: cond

    @testset "$(nameof(typeof(stat))) $regime" for stat in (Fermionic(), Bosonic()),
        regime in sort(collect(keys(HARD)))

        β, ωmax, ε = HARD[regime]
        basis = get_basis(stat, β, ωmax, ε)
        L = length(basis)
        gl = collect(range(1.0, -0.5; length=L)) .* basis.s ./ first(basis.s)
        for smpl in (TauSampling(basis), MatsubaraSampling(basis),
            MatsubaraSampling(basis; positive_only=true))
            tol = 100 * cond(smpl) * eps() * maximum(abs, gl)
            @test maximum(abs, fit(smpl, evaluate(smpl, gl)) .- gl) <= tol
        end

        ω0 = 0.3ωmax
        τs = [0.0, β / 3, β]
        ref = gtau_pole.(τs, ω0, β)
        got = transpose(basis.u(τs)) * (-basis.s .* basis.v(ω0))
        @test maximum(abs, got - ref) <= 300ε * maximum(abs, ref)

        if length(basis.sve_result.s) > L        # truncated by ε
            @test SparseIR.accuracy(basis) < ε <= last(SparseIR.significance(basis))
        end
    end
end
