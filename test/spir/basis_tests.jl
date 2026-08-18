@testitem "basis.jl" tags=[:julia, :lib] begin
    using SparseIR

    β = 2.0
    ωmax = 5.0
    ε = 1e-6
    Λ = β * ωmax
    @testset "FiniteTempBasis{S} for S=$(S)" for S in [Fermionic, Bosonic]
        basis = FiniteTempBasis(S(), β, ωmax, ε)
        s_full = basis.sve_result.s
        @test length(s_full) > length(basis.s)
        @test SparseIR.accuracy(basis) == s_full[length(basis.s) + 1] / first(s_full)
        @test SparseIR.accuracy(basis) < ε <= last(basis.s) / first(basis.s)
    end

    @testset "accuracy with max_size" begin
        basis = FiniteTempBasis(Fermionic(), 1.0, 42.0, ε; max_size=5)
        s_full = basis.sve_result.s
        @test length(basis.s) == 5
        @test SparseIR.accuracy(basis) == s_full[6] / first(s_full)
        @test SparseIR.accuracy(basis) > ε
    end

    @testset "accuracy without an excluded singular value" begin
        basis = FiniteTempBasis(Fermionic(), 1e-3, 1e-3, 1e-100)
        s_full = basis.sve_result.s
        @test length(s_full) == length(basis.s)
        @test SparseIR.accuracy(basis) == last(s_full) / first(s_full)
    end

    @testset "FiniteTempBasis{S} for S=$(S)" for S in [Fermionic, Bosonic]
        kernel = LogisticKernel(10.0)
        basis = FiniteTempBasis(S(), β, ωmax, ε; kernel)
        @test true
    end

    #==
    @testset "FiniteTempBasis{S} for K=RegularizedBoseKernel" begin
        kernel = RegularizedBoseKernel(10.0)
        @test_throws ArgumentError("RegularizedBoseKernel is incompatible with Fermionic statistics") FiniteTempBasis(
            Fermionic(), β, ωmax, ε; kernel)

        kernel = RegularizedBoseKernel(10.0)
        basis = FiniteTempBasis(Bosonic(), β, ωmax, ε; kernel)
    end
    ==#
end
