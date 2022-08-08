using MathLink
sphericalbesselj_M(n, x) = weval(W"SphericalBesselJ"(W"n", W"x"); n, x)

@testset "bessels_mathlink.jl" begin
    @testset "accuracy of sphericalbesselj(2^$pn, 2^$px)" for pn in 0:7, px in -1:25
        n = 2^pn
        x = 2.0^px
        @test SparseIR.sphericalbesselj(n, x) ≈ sphericalbesselj_M(n, x)
        @test SparseIR.sphericalbesselj(n + 1, x) ≈ sphericalbesselj_M(n + 1, x)
    end
end
