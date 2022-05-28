@testset "spr.jl" begin
    @testset "transform with stat = $stat" for stat in (fermion, boson)
        beta = 10_000
        wmax = 1.0
        eps = 1e-12
        basis = FiniteTempBasis(
            stat, float(beta), wmax, eps; sve_result=sve_logistic[(beta, eps)]
        )
        spr = SparsePoleRepresentation(basis)

        Random.seed!(4711)

        num_poles = 10
        poles = wmax * (2 * rand(num_poles) .- 1)
        coeffs = 2 * rand(num_poles) .- 1
        @test maximum(abs, poles) ≤ wmax

        Gl = to_IR(SparsePoleRepresentation(basis, poles), coeffs)

        g_spr = from_IR(spr, Gl)

        # Comparison on Matsubara frequencies
        smpl = MatsubaraSampling(basis)
        smpl_for_spr = MatsubaraSampling(spr, smpl.sampling_points)
        giv = evaluate(smpl_for_spr, g_spr)
        giv_ref = evaluate(smpl, Gl; dim=1)
        @test isapprox(giv, giv_ref; atol=300 * eps, rtol=0)

        # Comparison on tau
        smpl_tau = TauSampling(basis)
        gtau = evaluate(smpl_tau, Gl)
        smpl_tau_for_spr = TauSampling(spr)
        gtau2 = evaluate(smpl_tau_for_spr, g_spr)

        @test isapprox(gtau, gtau2; atol=300 * eps, rtol=0)
    end
end
