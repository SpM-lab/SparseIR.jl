@testitem "sve.jl" tags=[:julia, :lib] begin
    using Test
    using SparseIR

    @testset "sve_result/$(nameof(K))" for K in (LogisticKernel, RegularizedBoseKernel)
        sve = SparseIR.SVEResult(K(10), 1e-10)
        s = sve.s
        @test sve.kernel isa K
        @test length(s) > 5
        @test all(>(0), s)
        @test issorted(s; rev=true)
        # The SVE keeps singular values below the requested accuracy, so a basis
        # can be truncated at ε from it.
        @test last(s) / first(s) <= 1e-10
    end

    @testset "larger basis for smaller ε" begin
        sizes = [length(FiniteTempBasis(Fermionic(), 1.0, 42.0, ε)) for ε in (1e-4, 1e-8)]
        @test sizes[1] < sizes[2]
    end
end
