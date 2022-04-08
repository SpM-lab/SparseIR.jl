@testset "kernel.jl" begin
    @testset "accuracy" begin
        for K in [LogisticKernel(9),
                  RegularizedBoseKernel(8),
                  LogisticKernel(120_000),
                  RegularizedBoseKernel(127_500),
                  get_symmetrized(LogisticKernel(40_000), -1),
                  get_symmetrized(RegularizedBoseKernel(35_000), -1)]
            T = Float32
            T_x = Float64

            rule = convert(Rule{T}, legendre(10))
            hints = sve_hints(K, 2.2e-16)
            gauss_x = piecewise(rule, segments_x(hints))
            gauss_y = piecewise(rule, segments_y(hints))
            ϵ = eps(T)
            tiny = floatmin(T) / ϵ

            result = matrix_from_gauss(K, gauss_x, gauss_y)
            result_x = matrix_from_gauss(K,
                                         convert(Rule{T_x}, gauss_x),
                                         convert(Rule{T_x}, gauss_y))
            magn = maximum(abs, result_x)

            @test result ≈ result_x atol = 2magn * ϵ rtol = 0
            reldiff = @. ifelse(abs(result) < tiny, 1, result / result_x)
            @test all(isapprox.(reldiff, 1, atol=100ϵ, rtol=0))
        end
    end
end