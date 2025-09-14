@testitem "basis.jl" tags=[:julia, :lib] begin
    using SparseIR

    β = 2.0
    ωmax = 5.0
    ε = 1e-6
    Λ = β * ωmax
    @testset "FiniteTempBasis{S} for S=$(S)" for S in [Fermionic, Bosonic]
        basis = FiniteTempBasis(S(), β, ωmax, ε)
        @test true
    end

    @testset "FiniteTempBasis{S} for S=$(S)" for S in [Fermionic, Bosonic]
        kernel = LogisticKernel(10.0)
        basis = FiniteTempBasis(S(), β, ωmax, ε; kernel)
        @test true
    end

    @testset "FiniteTempBasis{S} for K=RegularizedBoseKernel" begin
        kernel = RegularizedBoseKernel(10.0)
        @test_throws ArgumentError("RegularizedBoseKernel is incompatible with Fermionic statistics") FiniteTempBasis(
            Fermionic(), β, ωmax, ε; kernel)

        kernel = RegularizedBoseKernel(10.0)
        basis = FiniteTempBasis(Bosonic(), β, ωmax, ε; kernel)
    end
end
