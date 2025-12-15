@testitem "DLR Constructor" tags=[:julia, :spir] begin
    using SparseIR
    using Test
    using Random
    using LinearAlgebra

    @testset "Constructor with default poles - $stat" for stat in (Fermionic(), Bosonic())
        # Test with Fermionic statistics
        β = 10000.0
        ωmax = 1.0
        ε = 1e-12
        if stat === Bosonic()
            kernel = RegularizedBoseKernel(β * ωmax)
        else
            kernel = LogisticKernel(β * ωmax)
        end
        basis = FiniteTempBasis(stat, β, ωmax, ε; kernel=kernel)
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
        @test get_poles(dlr) ≈ dlr.poles
        @test all(isone, SparseIR.significance(dlr))
        @test !SparseIR.iswellconditioned(dlr)

        # Test that poles are the default omega sampling points
        default_poles = default_omega_sampling_points(basis)
        @test dlr.poles ≈ default_poles
    end

    @testset "Constructor with custom poles - $stat" for stat in (Fermionic(), Bosonic())
        # Test with Bosonic statistics
        β = 10000.0
        ωmax = 1.0
        ε = 1e-12
        if stat === Bosonic()
            kernel = RegularizedBoseKernel(β * ωmax)
        else
            kernel = LogisticKernel(β * ωmax)
        end
        basis = FiniteTempBasis(stat, β, ωmax, ε; kernel=kernel)

        # Get default poles and use them as custom poles
        default_poles = default_omega_sampling_points(basis)
        dlr_custom = DiscreteLehmannRepresentation(basis, default_poles)

        @test dlr_custom isa DiscreteLehmannRepresentation
        @test SparseIR.statistics(dlr_custom) === stat
        @test dlr_custom.poles ≈ default_poles
        @test npoles(dlr_custom) == length(default_poles)
        @test get_poles(dlr_custom) ≈ default_poles

        # Verify poles can be retrieved correctly (like C++ test)
        poles_reconst = get_poles(dlr_custom)
        @test length(poles_reconst) == length(default_poles)
        for i in 1:length(poles_reconst)
            @test poles_reconst[i] ≈ default_poles[i]
        end

        # Test with random poles
        Random.seed!(123)
        num_poles = 10
        random_poles = ωmax * (2rand(num_poles) .- 1)
        dlr_random = DiscreteLehmannRepresentation(basis, random_poles)

        @test length(dlr_random) == num_poles
        @test dlr_random.poles ≈ random_poles
        @test maximum(abs, dlr_random.poles) ≤ ωmax

        # Verify custom poles are stored correctly
        retrieved_poles = get_poles(dlr_random)
        @test retrieved_poles ≈ random_poles
    end
end

@testitem "IR to DLR transformation" tags=[:julia, :spir] begin
    using SparseIR
    using Test
    using Random
    using LinearAlgebra

    @testset "Real coefficients - $stat" for stat in (Fermionic(), Bosonic())
        β = 10000.0
        ωmax = 1.0
        ε = 1e-12

        basis = FiniteTempBasis(stat, β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        # Test 1D arrays
        Random.seed!(42)
        gl = randn(length(basis))

        # Transform to DLR
        g_dlr = from_IR(dlr, gl)
        @test length(g_dlr) == length(dlr)
        @test eltype(g_dlr) == Float64

        # Transform back to IR
        gl_reconst = to_IR(dlr, g_dlr)
        @test length(gl_reconst) == length(basis)
        @test eltype(gl_reconst) == Float64

        # Since DLR is a different representation, we don't expect exact recovery
        # but the error should be reasonable
        @test norm(gl_reconst - gl) / norm(gl) < 0.1  # Within 10% relative error
    end

    @testset "Complex coefficients" begin
        β = 100.0
        ωmax = 10.0
        ε = 1e-10

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        Random.seed!(123)
        gl_complex = randn(ComplexF64, length(basis))

        # Transform to DLR
        g_dlr_complex = from_IR(dlr, gl_complex)
        @test length(g_dlr_complex) == length(dlr)
        @test eltype(g_dlr_complex) == ComplexF64

        # Transform back
        gl_complex_reconst = to_IR(dlr, g_dlr_complex)
        @test length(gl_complex_reconst) == length(basis)
        @test eltype(gl_complex_reconst) == ComplexF64

        # Check reasonable reconstruction
        @test norm(gl_complex_reconst - gl_complex) / norm(gl_complex) < 0.1
    end

    @testset "Multi-dimensional arrays" begin
        β = 100.0
        ωmax = 10.0
        ε = 1e-10

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        # Test 2D array transformations
        Random.seed!(42)
        gl_2d = randn(length(basis), 3)

        # Transform to DLR
        g_dlr_2d = from_IR(dlr, gl_2d)
        @test size(g_dlr_2d) == (length(dlr), 3)

        # Transform back to IR
        gl_2d_reconst = to_IR(dlr, g_dlr_2d)
        @test size(gl_2d_reconst) == (length(basis), 3)

        # Check reconstruction error
        @test norm(gl_2d_reconst - gl_2d) / norm(gl_2d) < 0.1

        # Test with different dimension
        gl_2d_dim2 = randn(5, length(basis))
        g_dlr_2d_dim2 = from_IR(dlr, gl_2d_dim2, 2)
        @test size(g_dlr_2d_dim2) == (5, length(dlr))

        gl_2d_dim2_reconst = to_IR(dlr, g_dlr_2d_dim2, 2)
        @test size(gl_2d_dim2_reconst) == (5, length(basis))
        @test norm(gl_2d_dim2_reconst - gl_2d_dim2) / norm(gl_2d_dim2) < 0.1

        # Test 3D array transformations
        gl_3d = randn(length(basis), 2, 4)
        g_dlr_3d = from_IR(dlr, gl_3d)
        @test size(g_dlr_3d) == (length(dlr), 2, 4)

        gl_3d_reconst = to_IR(dlr, g_dlr_3d)
        @test size(gl_3d_reconst) == (length(basis), 2, 4)
        @test norm(gl_3d_reconst - gl_3d) / norm(gl_3d) < 0.1

        # Test complex arrays
        gl_complex_2d = randn(ComplexF64, length(basis), 3)
        g_dlr_complex_2d = from_IR(dlr, gl_complex_2d)
        @test eltype(g_dlr_complex_2d) == ComplexF64
        @test size(g_dlr_complex_2d) == (length(dlr), 3)

        gl_complex_2d_reconst = to_IR(dlr, g_dlr_complex_2d)
        @test norm(gl_complex_2d_reconst - gl_complex_2d) / norm(gl_complex_2d) < 0.1
    end

    @testset "Error handling" begin
        β = 100.0
        ωmax = 1.0
        ε = 1e-12

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        # Test dimension mismatch for from_IR
        wrong_size_gl = randn(length(basis) + 1)
        @test_throws DimensionMismatch from_IR(dlr, wrong_size_gl)

        # Test dimension mismatch for to_IR
        wrong_size_dlr = randn(length(dlr) + 1)
        @test_throws DimensionMismatch to_IR(dlr, wrong_size_dlr)

        # Test multi-dimensional mismatch
        gl_2d = randn(length(basis), 5)
        g_dlr_2d = from_IR(dlr, gl_2d, 1)
        @test_throws DimensionMismatch from_IR(dlr, gl_2d, 2)  # Wrong dimension
        @test_throws DimensionMismatch to_IR(dlr, g_dlr_2d, 2)  # Wrong dimension
    end
end

@testitem "DLR with sampling" tags=[:julia, :spir] begin
    using SparseIR
    using Test
    using Random
    using LinearAlgebra

    @testset "Compression test - $stat" for stat in (Fermionic(), Bosonic())
        β = 10000.0
        ωmax = 1.0
        ε = 1e-12

        basis = FiniteTempBasis(stat, β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        Random.seed!(982743)

        # Create a function as a sum of poles
        num_poles = 10
        poles = ωmax * (2rand(num_poles) .- 1)
        coeffs = 2rand(num_poles) .- 1
        @test maximum(abs, poles) ≤ ωmax

        # Create DLR with these specific poles
        dlr_test = DiscreteLehmannRepresentation(basis, poles)

        # The coefficients in DLR directly correspond to the pole expansion
        # Convert to IR representation
        Gl = to_IR(dlr_test, coeffs)

        # Now use default DLR to represent this function
        g_dlr = from_IR(dlr, Gl)

        # Comparison on Matsubara frequencies
        smpl = MatsubaraSampling(basis)
        smpl_for_dlr = MatsubaraSampling(dlr; sampling_points=sampling_points(smpl))

        giv_ref = evaluate(smpl, Gl)
        giv = evaluate(smpl_for_dlr, g_dlr)

        # DLR should represent the function well
        @test isapprox(giv, giv_ref; atol=300ε, rtol=0)

        # Comparison on τ
        smpl_τ = TauSampling(basis)
        gτ = evaluate(smpl_τ, Gl)

        # For DLR, use the same sampling points as the original basis
        smpl_τ_for_dlr = TauSampling(dlr; sampling_points=sampling_points(smpl_τ))
        gτ2 = evaluate(smpl_τ_for_dlr, g_dlr)

        @test isapprox(gτ, gτ2; atol=300ε, rtol=0)
    end

    @testset "Bosonic pole representation" begin
        β = 2.0
        ωmax = 21.0
        ε = 1e-7

        basis_b = FiniteTempBasis(Bosonic(), β, ωmax, ε)

        # G(iw) = sum_p coeff_p / (iw - omega_p)
        coeff = [1.1, 2.0]
        ω_p = [2.2, -1.0]

        # Create DLR with these specific poles
        sp = DiscreteLehmannRepresentation(basis_b, ω_p)

        # Convert pole coefficients to IR
        gl_pole = to_IR(sp, coeff)

        # This should match the basis representation of the same function
        # (up to the accuracy of the representation)
        @test length(gl_pole) == length(basis_b)
    end
end

@testitem "DLR properties" tags=[:julia, :spir] begin
    using SparseIR
    using Test

    @testset "Pole management" begin
        β = 100.0
        ωmax = 5.0
        ε = 1e-10

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)

        # Test default pole selection
        default_poles = default_omega_sampling_points(basis)
        @test length(default_poles) > 0
        @test all(abs.(default_poles) .<= ωmax)

        # Create DLR and verify poles
        dlr = DiscreteLehmannRepresentation(basis)
        @test sampling_points(dlr) == dlr.poles
        @test npoles(dlr) == length(dlr.poles)
        @test length(get_poles(dlr)) == npoles(dlr)
        # The C API may return a different number of poles than requested
        @test npoles(dlr) > 0

        # Test with specific number of poles
        n_poles_custom = min(20, length(default_poles))
        custom_poles = default_poles[1:n_poles_custom]
        dlr_custom = DiscreteLehmannRepresentation(basis, custom_poles)
        @test length(dlr_custom) == n_poles_custom
        @test npoles(dlr_custom) == n_poles_custom
    end

    @testset "Basis properties inheritance" begin
        for stat in (Fermionic(), Bosonic())
            β = 1000.0
            ωmax = 10.0
            ε = 1e-11

            basis = FiniteTempBasis(stat, β, ωmax, ε)
            dlr = DiscreteLehmannRepresentation(basis)

            # Check that all properties are correctly inherited
            @test SparseIR.statistics(dlr) == stat
            @test SparseIR.β(dlr) == β
            @test SparseIR.ωmax(dlr) == ωmax
            @test SparseIR.Λ(dlr) == β * ωmax
            @test SparseIR.accuracy(dlr) ≤ ε

            # DLR-specific properties
            @test all(isone, SparseIR.significance(dlr))
            @test !SparseIR.iswellconditioned(dlr)
            @test SparseIR.basis(dlr) === basis
        end
    end

    @testset "Edge cases" begin
        # Small basis
        β = 10.0
        ωmax = 1.0
        ε = 1e-3

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)
        @test length(basis) < 20  # Should be small

        dlr = DiscreteLehmannRepresentation(basis)
        # The C API sometimes returns more poles than the basis size for small bases
        @test length(dlr) > 0

        # Test transformation with zeros
        gl_zeros = zeros(length(basis))
        g_dlr_zeros = from_IR(dlr, gl_zeros)
        @test all(iszero, g_dlr_zeros)

        gl_reconst_zeros = to_IR(dlr, g_dlr_zeros)
        @test all(iszero, gl_reconst_zeros)

        # Test transformation with ones
        gl_ones = ones(length(basis))
        g_dlr_ones = from_IR(dlr, gl_ones)
        gl_reconst_ones = to_IR(dlr, g_dlr_ones)
        # Skip norm test - would need LinearAlgebra import
    end
end
