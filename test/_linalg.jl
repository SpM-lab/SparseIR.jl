using Test
using SparseIR
using SparseIR._LinAlg

@testset "linalg" begin

@testset "jacobi" begin
    A = randn(20, 10)
    A_svd = svd_jacobi(A)
    A_rec = A_svd.U * (A_svd.S .* A_svd.Vt)
    @test isapprox(A_rec, A)
end

end
