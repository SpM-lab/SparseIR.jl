@testitem "DLR Constructor" tags=[:julia, :spir] setup=[SIRTestSetup] begin
    using SparseIR
    using Test
    using StableRNGs

    β, ωmax, ε = 10000.0, 1.0, 1e-12
    @testset "Constructor with default poles - $stat" for stat in (Fermionic(), Bosonic())
        basis = get_basis(stat, β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        @test dlr isa DiscreteLehmannRepresentation
        @test dlr isa SparseIR.AbstractBasis
        @test SparseIR.statistics(dlr) === stat
        @test SparseIR.β(dlr) == β
        @test SparseIR.ωmax(dlr) == ωmax
        @test SparseIR.Λ(dlr) == β * ωmax
        @test SparseIR.accuracy(dlr) ≤ ε
        @test length(dlr) == length(dlr.poles)
        @test size(dlr) == (length(dlr),)
        @test npoles(dlr) == length(dlr.poles)
        @test get_poles(dlr) == dlr.poles
        @test all(isone, SparseIR.significance(dlr))
        @test !SparseIR.iswellconditioned(dlr)

        # The default poles are the default omega sampling points
        @test dlr.poles == default_omega_sampling_points(basis)
    end

    @testset "Constructor with custom poles - $stat" for stat in (Fermionic(), Bosonic())
        basis = get_basis(stat, β, ωmax, ε)
        rng = StableRNG(123)
        random_poles = ωmax .* (2 .* rand(rng, 10) .- 1)
        dlr_random = DiscreteLehmannRepresentation(basis, random_poles)
        @test SparseIR.statistics(dlr_random) === stat
        @test length(dlr_random) == 10
        @test get_poles(dlr_random) == random_poles
        @test maximum(abs, get_poles(dlr_random)) ≤ ωmax
    end
end

@testitem "IR to DLR transformation" tags=[:julia, :spir] setup=[SIRTestSetup] begin
    using SparseIR
    using Test
    using StableRNGs
    using LinearAlgebra

    # IR coefficients of a sum of poles lie in the space the DLR represents, so
    # from_IR followed by to_IR must reproduce them to the accuracy of the basis.
    function pole_sum(basis, rng, T, extra...)
        poles = ωmax .* (2 .* rand(rng, 8) .- 1)
        coeffs = T <: Complex ? randn(rng, ComplexF64, 8, extra...) :
                 randn(rng, 8, extra...)
        return to_IR(DiscreteLehmannRepresentation(basis, poles), coeffs)
    end

    β, ωmax, ε = 1000.0, 1.0, 1e-10
    @testset "$T coefficients - $stat" for stat in (Fermionic(), Bosonic()),
        T in (Float64, ComplexF64)

        basis = get_basis(stat, β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)
        gl = pole_sum(basis, StableRNG(42), T)
        @test eltype(gl) === T

        g_dlr = from_IR(dlr, gl)
        @test length(g_dlr) == length(dlr)
        @test eltype(g_dlr) === T
        gl_reconst = to_IR(dlr, g_dlr)
        @test eltype(gl_reconst) === T
        @test norm(gl_reconst - gl) <= 300ε * norm(gl)
    end

    @testset "Multi-dimensional arrays" begin
        basis = get_basis(Fermionic(), β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)
        rng = StableRNG(7)
        for T in (Float64, ComplexF64), extra in ((3,), (2, 4))
            gl = pole_sum(basis, rng, T, extra...)
            g_dlr = from_IR(dlr, gl)
            @test size(g_dlr) == (length(dlr), extra...)
            @test norm(to_IR(dlr, g_dlr) - gl) <= 300ε * norm(gl)
            # The same data with the basis axis last
            N = ndims(gl)
            perm = (2:N..., 1)
            gl_last = permutedims(gl, perm)
            g_dlr_last = from_IR(dlr, gl_last, N)
            @test permutedims(g_dlr_last, invperm(perm)) ≈ g_dlr
            @test norm(to_IR(dlr, g_dlr_last, N) - gl_last) <= 300ε * norm(gl)
        end
    end

    @testset "Error handling" begin
        basis = get_basis(Fermionic(), β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)
        @test_throws DimensionMismatch from_IR(dlr, randn(length(basis) + 1))
        @test_throws DimensionMismatch to_IR(dlr, randn(length(dlr) + 1))
        gl_2d = randn(length(basis), 5)
        g_dlr_2d = from_IR(dlr, gl_2d, 1)
        @test_throws DimensionMismatch from_IR(dlr, gl_2d, 2)
        @test_throws DimensionMismatch to_IR(dlr, g_dlr_2d, 2)
    end
end

@testitem "DLR with sampling" tags=[:julia, :spir] setup=[SIRTestSetup] begin
    using SparseIR
    using Test
    using StableRNGs

    @testset "Compression test - $stat" for stat in (Fermionic(), Bosonic())
        β, ωmax, ε = 10000.0, 1.0, 1e-12
        basis = get_basis(stat, β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        rng = StableRNG(982743)
        poles = ωmax .* (2 .* rand(rng, 10) .- 1)
        coeffs = 2 .* rand(rng, 10) .- 1

        # IR coefficients of the pole sum, represented again with the default DLR
        Gl = to_IR(DiscreteLehmannRepresentation(basis, poles), coeffs)
        g_dlr = from_IR(dlr, Gl)

        smpl = MatsubaraSampling(basis)
        smpl_for_dlr = MatsubaraSampling(dlr; sampling_points=sampling_points(smpl))
        @test isapprox(evaluate(smpl_for_dlr, g_dlr), evaluate(smpl, Gl); atol=300ε, rtol=0)

        smpl_τ = TauSampling(basis)
        smpl_τ_for_dlr = TauSampling(dlr; sampling_points=sampling_points(smpl_τ))
        @test isapprox(
            evaluate(smpl_τ_for_dlr, g_dlr), evaluate(smpl_τ, Gl); atol=300ε, rtol=0)
    end

    @testset "Bosonic pole representation" begin
        β, ωmax, ε = 2.0, 21.0, 1e-7
        basis_b = FiniteTempBasis(Bosonic(), β, ωmax, ε)

        # G_l = -s_l Σ_p c_p v_l(ω_p) for G(iν) = Σ_p c_p tanh(βω_p/2)/(iν - ω_p)
        coeff = [1.1, 2.0]
        ω_p = [2.2, -1.0]
        gl_pole = to_IR(DiscreteLehmannRepresentation(basis_b, ω_p), coeff)
        @test isapprox(gl_pole, -basis_b.s .* (basis_b.v(ω_p) * coeff); atol=300ε, rtol=0)
    end
end

@testitem "DLR properties" tags=[:julia, :spir] setup=[SIRTestSetup] begin
    using SparseIR
    using Test
    using LinearAlgebra

    @testset "Pole management" begin
        basis = get_basis(Fermionic(), 100.0, 5.0, 1e-10)
        default_poles = default_omega_sampling_points(basis)
        @test length(default_poles) >= length(basis)
        @test all(abs.(default_poles) .<= 5.0)

        dlr = DiscreteLehmannRepresentation(basis)
        @test sampling_points(dlr) == dlr.poles
        @test npoles(dlr) == length(get_poles(dlr)) == length(dlr.poles)

        n_poles_custom = min(20, length(default_poles))
        dlr_custom = DiscreteLehmannRepresentation(basis, default_poles[1:n_poles_custom])
        @test length(dlr_custom) == npoles(dlr_custom) == n_poles_custom
    end

    @testset "Basis properties inheritance - $stat" for stat in (Fermionic(), Bosonic())
        β, ωmax, ε = 10000.0, 1.0, 1e-12
        basis = get_basis(stat, β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)
        @test SparseIR.statistics(dlr) == stat
        @test SparseIR.β(dlr) == β
        @test SparseIR.ωmax(dlr) == ωmax
        @test SparseIR.Λ(dlr) == β * ωmax
        @test SparseIR.accuracy(dlr) ≤ ε
        @test all(isone, SparseIR.significance(dlr))
        @test !SparseIR.iswellconditioned(dlr)
        @test SparseIR.basis(dlr) === basis
    end

    @testset "Edge cases" begin
        basis = get_basis(Fermionic(), 10.0, 1.0, 1e-3)
        @test length(basis) < 20
        dlr = DiscreteLehmannRepresentation(basis)
        @test length(dlr) > 0

        # Zeros stay zeros
        @test all(iszero, from_IR(dlr, zeros(length(basis))))
        @test all(iszero, to_IR(dlr, zeros(length(dlr))))

        # to_IR ∘ from_IR is a projection: applying it twice changes nothing
        p1 = to_IR(dlr, from_IR(dlr, ones(length(basis))))
        p2 = to_IR(dlr, from_IR(dlr, p1))
        @test norm(p2 - p1) <= 1e-10 * norm(p1)
    end
end
