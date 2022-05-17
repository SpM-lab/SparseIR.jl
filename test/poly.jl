using Test
using SparseIR

@testset "poly.jl" begin
    @testset "typestability" begin
        @test typestable(
            SparseIR._evaluate, [SparseIR.PiecewiseLegendrePoly{Float64}, Float64];
            checkonlyany=true,
        )
        @test typestable(
            SparseIR._evaluate,
            [SparseIR.PiecewiseLegendrePoly{Float64}, Vector{Float64}];
            checkonlyany=true,
        )
    end

    @testset "shape" begin
        u, s, v = sve_logistic[42]
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

        basis = DimensionlessBasis(fermion, 42.0; sve_result)
        @test length(basis[begin:5]) == 5

        basis = FiniteTempBasis(fermion, 4.2, 10.0; sve_result)
        @test length(basis[begin:4]) == 4
    end

    @testset "eval" begin
        u, s, v = sve_logistic[42]
        l = length(s)

        # Evaluate
        @test u(0.4) == [u[i](0.4) for i in 1:l]
        @test u.([0.4, -0.2]) == [[u[i](x) for i in 1:l] for x in (0.4, -0.2)]
    end

    @testset "matrix_hat" begin
        u, s, v = sve_logistic[42]
        uhat = SparseIR.hat.(u, :odd)

        n = [1, 3, 5, -1, -3, 5]
        result1 = uhat[1](n)
        result = uhat(reshape(n, (3, 2)))
        result_iter = reshape(uhat(n), (:, 3, 2))
        @test size(result1) == (length(n),)
        @test size(result) == size(result_iter)
        @test result == result_iter
    end

    @testset "overlap with Λ = $Λ, atol = $atol" for (Λ, atol) in
                                                     ((42, 1e-13), (10^4, 1e-13))
        u, s, v = sve_logistic[Λ]

        # Keep only even number of polynomials
        u, s, v = u[begin:(end - end % 2)],
        s[begin:(end - end % 2)],
        v[begin:(end - end % 2)]

        @test overlap(u[1], u[1]) ≈ 1 rtol = 0 atol = atol
        @test overlap(u[1], u[2]) ≈ 0 rtol = 0 atol = atol

        ref = float.(1:length(s) .== 1)
        @test all(isapprox.(overlap(u[1], u), ref, rtol=0, atol=atol))
    end

    @testset "eval unique" begin
        u, s, v = sve_logistic[42]
        û = SparseIR.hat.(u, :odd)

        # evaluate
        res1 = û([1, 3, 3, 1])
        idx = [1, 2, 2, 1]
        res2 = û([1, 3])[:, idx]
        @test res1 == res2
    end
end
