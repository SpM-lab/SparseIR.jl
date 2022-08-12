using Test
using SparseIR
using Documenter

DocMeta.setdocmeta!(SparseIR, :DocTestSetup, :(using SparseIR))

@testset "basis.jl" begin
    doctest(SparseIR; manual = false)

    @testset "consistency" begin
        β = 2.0
        wmax = 5.0
        ε = 1e-5
        basis_f, basis_b = SparseIR.finite_temp_bases(β, wmax, ε)
        bs = FiniteTempBasisSet(β, wmax, ε)

        @test length(bs.basis_f) == length(basis_f)
        @test length(bs.basis_b) == length(basis_b)
    end

    @testset "consistency2" begin
        β = 2.0
        wmax = 5.0
        ε = 1e-5

        sve_result = sve_logistic[β * wmax]
        basis_f, basis_b = SparseIR.finite_temp_bases(β, wmax, ε, sve_result)
        smpl_τ_f = TauSampling(basis_f)
        smpl_τ_b = TauSampling(basis_b)
        smpl_wn_f = MatsubaraSampling(basis_f)
        smpl_wn_b = MatsubaraSampling(basis_b)

        bs = FiniteTempBasisSet(β, wmax, ε; sve_result)
        @test smpl_τ_f.sampling_points == smpl_τ_b.sampling_points
        @test bs.smpl_tau_f.matrix == smpl_τ_f.matrix
        @test bs.smpl_tau_b.matrix == smpl_τ_b.matrix

        @test bs.smpl_wn_f.matrix == smpl_wn_f.matrix
        @test bs.smpl_wn_b.matrix == smpl_wn_b.matrix
    end
end
