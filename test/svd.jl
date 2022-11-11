using Test
using SparseIR
using SparseIR: Float64x2
using SparseIR.LinearAlgebra

@testset "svd.jl" begin
    mat64x2 = Float64x2.(rand(4, 6))
    @test_logs (:info, "n_sv_hint is set but will not be used in the current implementation!") SparseIR.compute_svd(mat64x2; n_sv_hint=2)
    @test_logs (:info, "strategy is set but will not be used in the current implementation!") SparseIR.compute_svd(mat64x2; strategy=:accurate)

    mat = rand(5, 6)
    @test_logs (:info, "n_sv_hint is set but will not be used in the current implementation!") SparseIR.compute_svd(mat; n_sv_hint=2)
    u, s, v = SparseIR.compute_svd(mat; strategy=:accurate)
    @test u * Diagonal(s) * v' ≈ mat
    @test_throws DomainError SparseIR.compute_svd(mat; strategy=:fast)
end