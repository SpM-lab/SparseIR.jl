using Test
using SparseIR
using SparseIR.LinearAlgebra
using SparseIR: gauss

function validategauss(rule)
    @test rule.a ≤ rule.b
    @test all(≤(rule.b), rule.x)
    @test all(≥(rule.a), rule.x)
    @test issorted(rule.x)
    @test length(rule.x) == length(rule.w)
    @test rule.x_forward ≈ rule.x .- rule.a
    @test rule.x_backward ≈ rule.b .- rule.x
end

@testset "gauss.jl" begin
    @testset "collocate" begin
        r = SparseIR.legendre(20)
        cmat = SparseIR.legendre_collocation(r)
        emat = SparseIR.legvander(r.x, length(r.x) - 1)
        @test emat * cmat ≈ I(20)
    end

    @testset "gauss legendre" begin
        rule = SparseIR.legendre(200)
        validategauss(rule)
        x, w = gauss(200)
        @test rule.x ≈ x
        @test rule.w ≈ w
    end

    @testset "piecewise" begin
        edges = [-4, -1, 1, 3]
        rule = SparseIR.piecewise(SparseIR.legendre(20), edges)
        validategauss(rule)
    end

    @testset "scale" begin
        rule = SparseIR.legendre(30)
        SparseIR.scale(rule, 2)
    end
end
