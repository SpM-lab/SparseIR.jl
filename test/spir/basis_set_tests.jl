@testitem "basis_set.jl" tags=[:julia, :spir] begin
using Test
using SparseIR
import SparseIR as SparseIR

    @testset "consistency" begin
        β = 2.0
        ωmax = 5.0
        ε = 1e-5
        basis_f, basis_b = SparseIR.finite_temp_bases(β, ωmax, ε)
        bs = FiniteTempBasisSet(β, ωmax, ε)

        @test length(bs.basis_f) == length(basis_f)
        @test length(bs.basis_b) == length(basis_b)
    end

    @testset "consistency2" begin
        β = 2.0
        ωmax = 5.0
        ε = 1e-5

        sve_result = SparseIR.SVEResult(LogisticKernel(β * ωmax), ε)
        basis_f, basis_b = SparseIR.finite_temp_bases(β, ωmax, ε; sve_result)
        smpl_τ_f = TauSampling(basis_f)
        smpl_τ_b = TauSampling(basis_b)
        smpl_wn_f = MatsubaraSampling(basis_f)
        smpl_wn_b = MatsubaraSampling(basis_b)

        bs = FiniteTempBasisSet(β, ωmax, ε; sve_result)
        @test smpl_τ_f.sampling_points == smpl_τ_b.sampling_points
        #@test bs.smpl_tau_f.matrix == smpl_τ_f.matrix
        #@test bs.smpl_tau_b.matrix == smpl_τ_b.matrix

        #@test bs.smpl_wn_f.matrix == smpl_wn_f.matrix
        #@test bs.smpl_wn_b.matrix == smpl_wn_b.matrix
    end

    @testset "unit tests LogisticKernel" begin
        β = 23
        ωmax = 3e-2
        ε = 1e-5

        bset = FiniteTempBasisSet(β, ωmax, ε)

        basis_f = FiniteTempBasis{Fermionic}(β, ωmax, ε)
        basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε)
        @test SparseIR.β(bset) == β
        @test SparseIR.ωmax(bset) == ωmax
        @test bset.tau == SparseIR.sampling_points(TauSampling(basis_f))
        @test bset.wn_f == SparseIR.sampling_points(MatsubaraSampling(basis_f))
        @test bset.wn_b == SparseIR.sampling_points(MatsubaraSampling(basis_b))
        # @test bset.sve_result.s ≈ SparseIR.SVEResult(LogisticKernel(β * ωmax); ε).s
        @test :tau ∈ propertynames(bset)
        # SparseIR.finite_temp_bases(0.1, 0.2, 1e-3; kernel=RegularizedBoseKernel(0.1 * 0.2))
    end
end # testitem
