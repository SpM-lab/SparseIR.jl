using Test
using SparseIR

@testset "poly.jl" begin
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

        basis = FiniteTempBasis(fermion, 4.2, 10; sve_result)
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
        û = hat.(u, :odd, 0:length(u)-1) # TODO: fix this

        n = [1, 3, 5, -1, -3, 5]
        result = û(reshape(n, (3, 2)))
        result_iter = reshape(û(n), (:, 3, 2))
        @test size(result) == size(result_iter)
        @test result == result_iter
    end

    @testset "overlap" begin
        for (Λ, atol) in [(42, 1e-13), (10^4, 1e-13)]
            u, s, v = sve_logistic[Λ]

            # Keep only even number of polynomials
            u, s, v = u[begin:(end - end % 2)], s[begin:(end - end % 2)],
                      v[begin:(end - end % 2)]

            @test overlap(u[1], u[1]) ≈ 1 rtol = 0 atol = atol
            @test overlap(u[1], u[2]) ≈ 0 rtol = 0 atol = atol

            ## TOO SLOW!
            # TODO: fix slowness (maybe I do need to write a custom adaptive integration routine)

            #=
            ref = float.(1:length(s) .== 1)
            @test overlap(u[1], u) ≈ ref rtol = 0 atol = atol

            function test(n)
                u, s, v = compute(LogisticKernel(42.0))

                # Keep only even number of polynomials
                u, s, v = u[begin:(end - end % 2)], s[begin:(end - end % 2)],
                          v[begin:(end - end % 2)]

                return overlap(u[1], u[1:n])
            end

            julia> @btime test(1)
              30.816 ms (930126 allocations: 18.97 MiB)
            1-element Vector{Float64}:
             0.9999999999999993

            julia> @btime test(2)
              31.050 ms (931701 allocations: 19.17 MiB)
            2-element Vector{Float64}:
             1.0000000000000002
             1.734723475976807e-18

            julia> @btime test(3)
              31.229 ms (934782 allocations: 19.47 MiB)
            3-element Vector{Float64}:
              1.0000000000000004
              2.2551405187698492e-17
             -8.500145032286355e-17

            julia> @btime test(4)
              33.982 ms (954388 allocations: 21.43 MiB)
            4-element Vector{Float64}:
              1.0000000000000002
             -1.3444106938820255e-17
             -1.0408340855860843e-16
              6.505213034913027e-19

            julia> @btime test(5)
              38.704 ms (989623 allocations: 24.90 MiB)
            5-element Vector{Float64}:
              0.9999999999999991
             -2.6020852139652106e-18
              1.249000902703301e-16
              1.3227266504323154e-17
             -3.329584871702984e-16

            julia> @btime test(6)
              41.473 ms (1070787 allocations: 33.31 MiB)
            6-element Vector{Float64}:
              0.9999999999999997
              1.0299920638612292e-17
              1.3010426069826053e-16
              9.974659986866641e-18
             -2.627021863932377e-16
              1.214306433183765e-17

            julia> @btime test(7)
              605.345 ms (6422321 allocations: 574.46 MiB)
            7-element Vector{Float64}:
              1.0
              2.003741140024773e-17
              4.333556083424561e-16
             -1.2996873542669984e-17
             -3.214117340333278e-16
             -1.620882247865829e-17
             -6.765760369488449e-16

            julia> @btime test(8)
            # aborted because it took too long
            =#
        end
    end

    @testset "eval unique" begin
        u, s, v = sve_logistic[42]
        û = hat.(u, :odd, 0:length(u)-1) # TODO: fix this

        # evaluate
        res1 = û([1, 3, 3, 1])
        idx = [1, 2, 2, 1]
        res2 = û([1, 3])[:, idx]
        @test res1 == res2
    end
end
