using Test
using LinearAlgebra
using SparseIR
using SparseIR._LinAlg

@testset "linalg" begin

@testset "jacobi" begin
    A = randn(20, 10)
    A_svd = svd_jacobi(A)
    A_rec = A_svd.U * (A_svd.S .* A_svd.Vt)
    @test isapprox(A_rec, A)
end

@testset "rrqr" begin
    A = randn(40, 30)
    A_eps = norm(A) * eps(eltype(A))
    A_qr, A_rank = rrqr(A)
    A_rec = A_qr.Q * A_qr.R * A_qr.P'
    @test isapprox(A_rec, A; rtol=0, atol=4 * A_eps)
    @test A_rank == 30
end

@testset "rrqr_trunc" begin
    # Vandermonde matrix
    A = Vector(-1:0.02:1) .^ Vector(0:20)'
    m, n = size(A)
    A_qr, k = SparseIR._LinAlg.rrqr(A, rtol=1e-5)
    @assert k < min(m, n)

    Q, R = SparseIR._LinAlg.truncate_qr_result(A_qr, k)
    A_rec = Q * R * A_qr.P'
    @test isapprox(A, A_rec, rtol=0, atol=1e-5 * norm(A))
end

end
