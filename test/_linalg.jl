using Test
using LinearAlgebra
using SparseIR
using SparseIR._LinAlg
using DoubleFloats

@testset "_linalg.jl" begin
    @testset "jacobi with T = $T" for T in (Float64, Double64)
        A = randn(T, 20, 10)
        U, S, V = svd_jacobi(A)
        @test U * Diagonal(S) * V' ≈ A
    end

    @testset "rrqr with T = $T" for T in (Float64, Double64)
        A = randn(T, 40, 30)
        A_eps = norm(A) * eps(eltype(A))
        A_qr, A_rank = rrqr(A)
        A_rec = A_qr.Q * A_qr.R * A_qr.P'
        @test isapprox(A_rec, A; rtol=0, atol=4 * A_eps)
        @test A_rank == 30
    end

    @testset "rrqr_trunc with T = $T" for T in (Float64, Double64)
        # Vandermonde matrix
        A = Vector{T}(-1:0.02:1) .^ Vector{T}(0:20)'
        m, n = size(A)
        A_qr, k = SparseIR._LinAlg.rrqr(A; rtol=1e-5)
        @test k < min(m, n)

        Q, R = SparseIR._LinAlg.truncate_qr_result(A_qr, k)
        A_rec = Q * R * A_qr.P'
        @test isapprox(A, A_rec, rtol=0, atol=1e-5 * norm(A))
    end

    @testset "tsvd with T = $T" for T in (Float64, Double64), tol in (1e-14, 1e-13)
        A = Vector{T}(-1:0.01:1) .^ Vector{T}(0:50)'
        U, S, V = SparseIR._LinAlg.tsvd(A; rtol=tol)
        k = length(S)

        @test U * Diagonal(S) * V' ≈ A rtol = 0 atol = tol * norm(A)
        @test U'U ≈ I
        @test V'V ≈ I
        @test issorted(S; rev=true)
        @test k < minimum(size(A))

        A_svd = svd(Float64.(A))
        @test S ≈ A_svd.S[1:k]
    end
end
