using Test
using SparseIR
using MathLink
sphericalbesselj_M(n, x) = weval(W"SphericalBesselJ"(W"n", W"x"); n, x)

@testset "_specfuncs.jl" begin
    @testset "domain" begin
        @test_throws DomainError SparseIR.sphericalbesselj(-2, 0.3)
        @test !isnan(SparseIR.sphericalbesselj(4, 1e20))
    end

    @testset "accuracy (with pn = $pn, px = $px)" for pn in 0:7, px in 0:25
        n = 2^pn
        x = float(2^px)
        @test SparseIR.sphericalbesselj(n, x) ≈ sphericalbesselj_M(n, x)
        @test SparseIR.sphericalbesselj(n + 1, x) ≈ sphericalbesselj_M(n + 1, x)
    end
end
