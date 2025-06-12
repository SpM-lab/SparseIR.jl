@testsnippet ValidateGaussFunction begin
    function validategauss(rule)
        @test rule.a ≤ rule.b
        @test all(≤(rule.b), rule.x)
        @test all(≥(rule.a), rule.x)
        @test issorted(rule.x)
        @test length(rule.x) == length(rule.w)
        @test rule.x_forward ≈ rule.x .- rule.a
        @test rule.x_backward ≈ rule.b .- rule.x
    end
end

@testitem "collocate" begin
    using LinearAlgebra

    r = SparseIR.legendre(20)
    cmat = SparseIR.legendre_collocation(r)
    emat = SparseIR.legvander(r.x, length(r.x) - 1)
    @test emat * cmat ≈ I(20)
end

@testitem "gauss legendre" setup=[ValidateGaussFunction] begin
    rule = SparseIR.legendre(200)
    validategauss(rule)
    x, w = SparseIR.gauss(200)
    @test rule.x ≈ x
    @test rule.w ≈ w
end

@testitem "piecewise" setup=[ValidateGaussFunction] begin
    edges = [-4, -1, 1, 3]
    rule = SparseIR.piecewise(SparseIR.legendre(20), edges)
    validategauss(rule)
end

@testitem "scale" setup=[ValidateGaussFunction] begin
    rule = SparseIR.legendre(30)
    validategauss(SparseIR.scale(rule, 2))
end
