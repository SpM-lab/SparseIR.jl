@testitem "test1" tags=[:julia, :spir] begin
    using SparseIR
    using Test
    using Random
    using LinearAlgebra

    @testset "Constructor with default poles" begin
        β = 100.0
        ωmax = 1.0
        ε = 1e-12

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)
    end
end

@testitem "test2" tags=[:julia, :spir] begin
    using SparseIR
    using Test
    using Random
    using LinearAlgebra

    @testset "Constructor with default poles" begin
        β = 10000.0
        ωmax = 1.0
        ε = 1e-12

        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)
    end
end