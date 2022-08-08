using Test
using SparseIR

include("_util.jl")

@testset "bessels.jl" begin
    @test typestable(SparseIR.sphericalbesselj, [Int, Float64])

    @testset "domain" begin
        @test_throws DomainError SparseIR.sphericalbesselj(-2, 0.3)
        @test !isnan(SparseIR.sphericalbesselj(4, 1e20))
    end

    @testset "special cases" begin
        @test isnan(SparseIR.sphericalbesselj(237, NaN))
        @test iszero(SparseIR.sphericalbesselj(42, Inf))
        @test isone(SparseIR.sphericalbesselj(0, 0.0))
        @test iszero(SparseIR.sphericalbesselj(99, 0.0))
    end

    @testset "small x" begin
        n = 11
        x = 1e0
        # Mathematica
        ref = 3.099551854790080034457495747083911933213405593516888829346e-12
        @test SparseIR.sphericalbesselj(n, x) ≈ ref
    end

    @testset "float n" begin
        # randomly chosen
        n = 23.2
        x = 221.34
        # Mathematica
        ref = -0.00329119519019341450255948137321395977016102685418180379
        @test SparseIR.sphericalbesselj(n, x) ≈ ref
    end

    @testset "large x" begin
        n = 11
        x = 2e7
        # Mathematica
        ref = 3.231408127574738307041647037980510188264714173990261929327e-8
        @test SparseIR.sphericalbesselj(n, x) ≈ ref
    end
end
