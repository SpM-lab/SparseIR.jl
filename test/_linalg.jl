using Test
import LinearAlgebra
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
    A_eps = LinearAlgebra.norm(A) * eps(eltype(A))
    A_qr, A_rank = rrqr(A)
    A_rec = A_qr.Q * A_qr.R * A_qr.P'
    @test isapprox(A_rec, A; rtol=0, atol=4 * A_eps)
    @test A_rank == 30
end

end
