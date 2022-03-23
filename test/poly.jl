using Test

using SparseIR

@testset "poly.jl" begin
    @testset "shape" begin
        u, s, v = sve_logistic[42]
        @show u
        @show s
        @show v
        l = length(s)
        @test size(u) == (1,)
        # @test size(u[3]) == ()
        # @test size(u[2:5]) == (3,)
    end

    # @testset "slice" begin
    #     sve_result = sve_logistic[42]

    #     basis = IRBasis(:F, 42; sve_result)
    #     @test length(basis[begin:5]) == 5

    #     basis = FiniteTempBasis(:F, 4.2, 10; sve_result)
    #     @test length(basis[begin:4]) == 4
    # end

    # @testset "eval" begin
    #     u, s, v = sve_logistic[42]
    #     l = length(s)

    #     #Evaluate
    #     # TODO: do we need assert_array_almost_equal_nulp here?
    #     @test u(0.4) ≈ [u[i](0.4) for i in 1:l]
    #     @test u([0.4, -0.2]) ≈ [[u[i](x) for x in (0.4, -0.2)] for i in 1:l]
    # end

    # @testset "broadcast" begin
    #     u, s, v = sve_logistic[42]

    #     x = [0.3, 0.5]
    #     l = [2, 7]
    #     # TODO: see above
    #     @test value(u, l, x) ≈ [u[ll](xx) for (ll, xx) in zip(l, x)]
    # end

    # @testset "matrix_hat" begin
    #     u, s, v = sve_logistic[42]
    #     û = hat(u, :odd)

    #     n = [1, 3, 5, -1, -3, 5]
    #     result = û(reshape(n, (3, 2)))
    #     result_iter = reshape(û(n), (:, 3, 2))
    #     @test size(result) == size(result_iter)
    #     @test result ≈ result_iter
    # end

    # @testset "overlap" begin
    #     for (Λ, atol) in [(42, 1e-14), (10^4, 5e-13)]
    #         u, s, v = sve_logistic[Λ]

    #         # Keep only even number of polynomials
    #         u, s, v = u[begin:(end - length(u) % 2)], s[begin:(end - length(s) % 2)],
    #                   v[begin:(end - length(v) % 2)]
    #         npoly = length(s)

    #         @test overlap(first(u), first(u)) ≈ 1 rtol = 0 atol = atol

    #         ref = ones(length(s))
    #         ref[begin] = 0
    #         @test abs(overlap(u, first(u)) - 1) ≈ ref rtol = 0 atol = atol

    #         # Axis
    #         trans_f(x) = transpose(first(u)(x))
    #         @test abs(overlap(u, trans_f; axis=0) - 1) ≈ ref rtol = 0 atol = atol # TODO axis=0 ?

    #         @test overlap(u, u; axis=-1) ≈ Matrix(I, length(s), length(s)) rtol = 0 atol = atol # TODO
    #     end
    # end
end