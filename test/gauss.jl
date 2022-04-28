using Test
using SparseIR
using SparseIR._SpecFuncs: legvander

using LinearAlgebra: I
using QuadGK: gauss

function validategauss(rule)
    @test rule.a ≤ rule.b
    @test all(≤(rule.b), rule.x)
    @test all(≥(rule.a), rule.x)
    @test issorted(rule.x)
    @test length(rule.x) == length(rule.w)
end

@testset "gauss.jl" begin
    @testset "collocate" begin
        r = legendre(20)
        cmat = legendre_collocation(r)
        emat = legvander(r.x, length(r.x) - 1)
        @test emat * cmat ≈ I(20)
    end

    @testset "gauss legendre" begin
        rule = legendre(200)
        validategauss(rule)
        x, w = gauss(200)
        @test rule.x ≈ x
        @test rule.w ≈ w
    end

    @testset "piecewise" begin
        edges = [-4, -1, 1, 3]
        rule = piecewise(legendre(20), edges)
        validategauss(rule)
    end

    @testset "integrals" begin
        r = legendre(40)
        f(x) = cos(5x) + cos(50x)
        @test quadrature(r, f) ≈ 2sin(5) / 5 + 2sin(50) / 50
        g(x) = sin(x^3)
        @test quadrature(reseat(r, 1, 4), g) ≈ 0.2042835844987353218
    end
end
