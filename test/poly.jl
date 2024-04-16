using Test
using SparseIR
using SparseIR.LinearAlgebra

isdefined(Main, :sve_logistic) || include("_conftest.jl")

@testset "poly.jl" begin
    @testset "shape" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        l = length(s)
        @test size(u) == (l,)

        @test size(u[4]) == ()
        @test size(u[3:5]) == (3,)
    end

    @testset "knots" begin
        u, s, v = sve_logistic[42]
        @test first(u[1].knots) == -1.0
        @test last(u[1].knots) == 1.0
    end

    @testset "slice" begin
        sve_result = sve_logistic[42]

        basis = FiniteTempBasis{Fermionic}(4.2, 10.0; sve_result)
        @test length(basis[1:4]) == 4
    end

    @testset "eval" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        l = length(s)

        # Evaluate
        @test u(0.4) == [u[i](0.4) for i in 1:l]
        @test u.([0.4, -0.2]) == [[u[i](x) for i in 1:l] for x in (0.4, -0.2)]
    end

    @testset "matrix_hat" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        uhat = SparseIR.PiecewiseLegendreFTVector(u, Fermionic())

        n = MatsubaraFreq.([1, 3, 5, -1, -3, 5])
        result1 = uhat[1](n)
        result = uhat(reshape(n, (3, 2)))
        result_iter = reshape(uhat(n), (:, 3, 2))
        @test size(result1) == (length(n),)
        @test size(result) == size(result_iter)
        @test result == result_iter
    end

    @testset "overlap with Λ = $Λ" for Λ in (10, 42, 10_000)
        atol = 1e-13
        u, s, v = SparseIR.part(sve_logistic[Λ])

        # Keep only even number of polynomials
        u, s, v = u[1:(end - end % 2)], s[1:(end - end % 2)], v[1:(end - end % 2)]

        @test overlap(u[1], u[1])≈1 rtol=0 atol=atol
        @test overlap(u[1], u[2])≈0 rtol=0 atol=atol
        @test overlap(u[1], u[1]; points=[0.1, 0.2])≈1 rtol=0 atol=atol

        ref = float.(eachindex(s) .== 1)
        @test all(isapprox.(overlap(u[1], u), ref; rtol=0, atol))
    end

    @testset "eval unique" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        û = SparseIR.PiecewiseLegendreFTVector(u, Fermionic())

        # evaluate
        res1 = û([1, 3, 3, 1])
        idx = [1, 2, 2, 1]
        res2 = û([1, 3])[:, idx]
        @test res1 == res2
    end

    @testset "unit tests" begin
        u, s, v = SparseIR.part(sve_logistic[42])

        @test size(u[1](rand(30))) == (30,)

        @test_throws DomainError u(SparseIR.xmax(u) + 123)
        @test_throws DomainError u(SparseIR.xmin(u) - 123)

        int_result, int_error = SparseIR.overlap(u[1], u[1]; return_error=true)

        @test int_error < eps()

        u_linearcombination = 2u[1] - 3u[2]
        @test SparseIR.overlap(u_linearcombination, u[2]) ≈ -3

        @test size(u(rand(2, 3, 4))) == (length(u), 2, 3, 4)

        @test (SparseIR.xmin(u), SparseIR.xmax(u)) === (-1.0, 1.0)
        @test SparseIR.knots(u) == u[end].knots
        @test SparseIR.Δx(u) == u[1].Δx
        @test SparseIR.symm(u)[2] == SparseIR.symm(u[2])
        @test all(<(eps()), SparseIR.overlap(u, sin)[1:2:end])
        @test all(<(eps()), SparseIR.overlap(u, cos)[2:2:end])

        û = SparseIR.PiecewiseLegendreFTVector(u, Fermionic())

        @test length(SparseIR.moments(û)[1]) == length(û)
        û_weird = SparseIR.PiecewiseLegendreFT(-9u[2] + u[3], Fermionic())
        @test_throws ErrorException SparseIR.func_for_part(û_weird)
        @test SparseIR.phase_stable(u, 5) ≈ SparseIR.phase_stable(u, 5.0)
    end
end
