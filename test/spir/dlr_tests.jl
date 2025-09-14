@testitem "DLR Constructor" tags=[:julia, :spir] begin
    using SparseIR
    using Test
    using Random
    using LinearAlgebra

    @testset "Constructor with default poles" begin
        # Test with Fermionic statistics
        β = 10000.0
        ωmax = 1.0
        ε = 1e-12

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)

        #@test dlr isa DiscreteLehmannRepresentation
        #@test dlr isa SparseIR.AbstractBasis
        #@test SparseIR.statistics(dlr) isa Fermionic
        #@test SparseIR.β(dlr) == β
        #@test SparseIR.ωmax(dlr) == ωmax
        #@test SparseIR.Λ(dlr) == β * ωmax
        #@test SparseIR.accuracy(dlr) ≤ ε
        #@test length(dlr) == length(dlr.poles)
        #@test size(dlr) == (length(dlr),)
        #@test npoles(dlr) == length(dlr.poles)
        #@test get_poles(dlr) ≈ dlr.poles
        #@test all(isone, SparseIR.significance(dlr))
        #@test !SparseIR.iswellconditioned(dlr)

        ## Test that poles are the default omega sampling points
        #default_poles = default_omega_sampling_points(basis)
        #@test dlr.poles ≈ default_poles
    end

    #==
    @testset "Constructor with custom poles" begin
        # Test with Bosonic statistics
        β = 10000.0
        ωmax = 1.0
        ε = 1e-12

        basis = FiniteTempBasis(Bosonic(), β, ωmax, ε)

        # Get default poles and use them as custom poles
        default_poles = default_omega_sampling_points(basis)
        dlr_custom = DiscreteLehmannRepresentation(basis, default_poles)

        @test dlr_custom isa DiscreteLehmannRepresentation
        @test SparseIR.statistics(dlr_custom) isa Bosonic
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
    ==#
end