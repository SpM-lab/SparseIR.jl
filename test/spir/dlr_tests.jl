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

        #==
        @test dlr isa DiscreteLehmannRepresentation
        @test dlr isa SparseIR.AbstractBasis
        @test SparseIR.statistics(dlr) isa Fermionic
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
        ==#
    end
end