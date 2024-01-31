using Test
using SparseIR: Float64x2

logrange(x1, x2, length) = (exp10(y) for y in range(log10(x1), log10(x2); length))

@testset "_multifloat_funcs.jl" begin
    xx = collect(logrange(floatmin(Float64), 20.0, 1000))
    xx = [0.0] ∪ xx ∪ [Inf]
    xx = -xx ∪ xx
    xx = Float64x2.(xx)

    @testset "type inference and stability" begin
        x = rand(xx)
        @inferred sinh(x)
        @inferred cosh(x)

        @test sinh(x) isa Float64x2
        @test cosh(x) isa Float64x2
    end

    @testset "sinh(x) where x = $x" for x in xx
        @test sinh(x)≈sinh(big(x)) rtol=eps(Float64x2)
    end

    @testset "cosh(x) where x = $x" for x in xx
        @test cosh(x)≈cosh(big(x)) rtol=eps(Float64x2)
    end

    @test isnan(sinh(Float64x2(NaN)))
    @test isnan(cosh(Float64x2(NaN)))
end
