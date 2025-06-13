@testsnippet MultiFloatSetup begin
    logrange(x1, x2, length) = (exp10(y) for y in range(log10(x1), log10(x2); length))

    xx = collect(logrange(floatmin(Float64), 20.0, 1000))
    xx = [0.0] ∪ xx ∪ [Inf]
    xx = -xx ∪ xx
    xx = SparseIR.Float64x2.(xx)
end

@testitem "type inference and stability" setup=[MultiFloatSetup] begin
    x = rand(xx)
    @inferred sinh(x)
    @inferred cosh(x)

    @test sinh(x) isa SparseIR.Float64x2
    @test cosh(x) isa SparseIR.Float64x2
end

@testitem "sinh(x)" setup=[MultiFloatSetup] begin
    for x in xx
        @test sinh(x)≈sinh(big(x)) rtol=eps(SparseIR.Float64x2)
    end
end

@testitem "cosh(x)" setup=[MultiFloatSetup] begin
    for x in xx
        @test cosh(x)≈cosh(big(x)) rtol=eps(SparseIR.Float64x2)
    end
end

@testitem "Float64x2(NaN) sinh/cosh" setup=[MultiFloatSetup] begin
    @test isnan(sinh(SparseIR.Float64x2(NaN)))
    @test isnan(cosh(SparseIR.Float64x2(NaN)))
end
