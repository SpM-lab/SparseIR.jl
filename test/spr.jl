using Test
using SparseIR
using Random

include("_conftest.jl")

@testset "spr.jl" begin
    @testset "Compression with stat = $stat" for stat in (Fermionic(), Bosonic())
        β = 10_000
        ωmax = 1
        ε = 1e-12
        basis = FiniteTempBasis(stat, β, ωmax, ε; sve_result=sve_logistic[β * ωmax])
        spr = SparsePoleRepresentation(basis)

        Random.seed!(982743)

        num_poles = 10
        poles = ωmax * (2rand(num_poles) .- 1)
        coeffs = 2rand(num_poles) .- 1
        @test maximum(abs, poles) ≤ ωmax

        Gl = SparseIR.to_IR(SparsePoleRepresentation(basis, poles), coeffs)
        g_spr = SparseIR.from_IR(spr, Gl)

        # Comparison on Matsubara frequencies
        smpl = MatsubaraSampling(basis)
        smpl_for_spr = MatsubaraSampling(spr, SparseIR.sampling_points(smpl))

        giv_ref = evaluate(smpl, Gl; dim=1)
        giv = evaluate(smpl_for_spr, g_spr)

        @test isapprox(giv, giv_ref; atol=300ε, rtol=0)

        # Comparison on τ
        smpl_τ = TauSampling(basis)
        gτ = evaluate(smpl_τ, Gl)

        smpl_τ_for_spr = TauSampling(spr)
        gτ2 = evaluate(smpl_τ_for_spr, g_spr)

        @test isapprox(gτ, gτ2; atol=300ε, rtol=0)
    end

    @testset "Boson" begin
        β = 2
        ωmax = 21
        ε = 1e-7
        basis_b = FiniteTempBasis(Bosonic(), β, ωmax, ε; sve_result=sve_logistic[β * ωmax])

        # G(iw) = sum_p coeff_p U^{SPR}(iw, omega_p)
        coeff = [1.1, 2.0]
        ω_p = [2.2, -1.0]

        ρl_pole = basis_b.v(ω_p) * coeff
        gl_pole = -basis_b.s .* ρl_pole

        sp = SparsePoleRepresentation(basis_b, ω_p)
        gl_pole2 = SparseIR.to_IR(sp, coeff)

        @test isapprox(gl_pole, gl_pole2; atol=300ε, rtol=0)
    end
end
