@testset "basis.jl" begin
    @testset "consistency" begin
        β = 2.0
        wmax = 5.0
        ε = 1e-5

        # sve_result = sve_logistic[10#= β * wmax =#]
        sve_result = compute(LogisticKernel(β * wmax))
        basis_f, basis_b = finite_temp_bases(β, wmax, ε, sve_result)
        # smpl_τ_f = TauSampling(basis_f)
        # smpl_τ_b = TauSampling(basis_b)
        # smpl_wn_f = MatsubaraSampling(basis_f)
        # smpl_wn_b = MatsubaraSampling(basis_b)

        # bs = FiniteTempBasisSet(β, wmax, ε; sve_result)
        # @test smpl_τ_f.sampling_points == smpl_τ_b.sampling_points
        # @test bs.smpl_tau_f.matrix.a == smpl_τ_f.matrix.a
        # @test bs.smpl_tau_b.matrix.a == smpl_τ_b.matrix.a

        # @test bs.smpl_wn_f.matrix.a == smpl_wn_f.matrix.a
        # @test bs.smpl_wn_b.matrix.a == smpl_wn_b.matrix.a
    end
end