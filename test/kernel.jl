using Test
using SparseIR

@testset "kernel.jl" begin
    @testset "accuracy with K = $K" for K in (LogisticKernel(9.0),
                                              RegularizedBoseKernel(8.0),
                                              LogisticKernel(120_000.0),
                                              RegularizedBoseKernel(127_500.0),
                                              SparseIR.get_symmetrized(LogisticKernel(40_000.0),
                                                                       -1),
                                              SparseIR.get_symmetrized(RegularizedBoseKernel(35_000.0),
                                                                       -1))
        T = Float32
        T_x = Float64

        rule = convert(SparseIR.Rule{T}, SparseIR.legendre(10))
        hints = SparseIR.sve_hints(K, eps(T_x))
        gauss_x = SparseIR.piecewise(rule, SparseIR.segments_x(hints))
        gauss_y = SparseIR.piecewise(rule, SparseIR.segments_y(hints))
        ϵ = eps(T)
        tiny = floatmin(T) / ϵ

        result = SparseIR.matrix_from_gauss(K, gauss_x, gauss_y)
        result_x = SparseIR.matrix_from_gauss(K, convert(SparseIR.Rule{T_x}, gauss_x),
                                              convert(SparseIR.Rule{T_x}, gauss_y))
        magn = maximum(abs, result_x)

        @test result≈result_x atol=2magn * ϵ rtol=0
        reldiff = @. ifelse(abs(result) < tiny, 1, result / result_x)
        @test all(x -> isapprox(x, 1; atol=100ϵ, rtol=0), reldiff)
    end

    @testset "singularity with Λ = $Λ" for Λ in (10, 42, 10_000), x in 2rand(10) .- 1
        K = RegularizedBoseKernel(Λ)
        @test K(x, 0.0) ≈ 1 / Λ
    end

    @testset "unit tests" begin
        K = LogisticKernel(42)
        K_symm = SparseIR.get_symmetrized(K, 1)
        @test !SparseIR.iscentrosymmetric(K_symm)
        @test_throws ErrorException SparseIR.get_symmetrized(K_symm, -1)

        K = RegularizedBoseKernel(99)
        hints = SparseIR.sve_hints(K, 1e-6)
        @test SparseIR.nsvals(hints) == 56
        @test SparseIR.ngauss(hints) == 10
        @test SparseIR.ypower(K) == 1
        @test SparseIR.ypower(SparseIR.get_symmetrized(K, -1)) == 1
        @test SparseIR.ypower(SparseIR.get_symmetrized(K, 1)) == 1
        @test SparseIR.conv_radius(K) == 40 * 99
        @test SparseIR.conv_radius(SparseIR.get_symmetrized(K, -1)) == 40 * 99
        @test SparseIR.conv_radius(SparseIR.get_symmetrized(K, 1)) == 40 * 99
    end
end
