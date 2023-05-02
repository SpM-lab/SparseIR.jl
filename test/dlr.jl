using Test
using SparseIR
using Random

isdefined(Main, :sve_logistic) || include("_conftest.jl")

@testset "dlr.jl" begin
    @testset "Compression with stat = $stat" for stat in (Fermionic(), Bosonic())
        β = 10_000
        ωmax = 1
        ε = 1e-12
        basis = FiniteTempBasis(stat, β, ωmax, ε; sve_result=sve_logistic[β * ωmax])
        dlr = DiscreteLehmannRepresentation(basis)

        Random.seed!(982743)

        num_poles = 10
        poles = ωmax * (2rand(num_poles) .- 1)
        coeffs = 2rand(num_poles) .- 1
        @test maximum(abs, poles) ≤ ωmax

        Gl = SparseIR.to_IR(DiscreteLehmannRepresentation(basis, poles), coeffs)
        g_dlr = SparseIR.from_IR(dlr, Gl)

        # Comparison on Matsubara frequencies
        smpl = MatsubaraSampling(basis)
        smpl_for_dlr = MatsubaraSampling(dlr;
                                         sampling_points=SparseIR.sampling_points(smpl))

        giv_ref = evaluate(smpl, Gl; dim=1)
        giv = evaluate(smpl_for_dlr, g_dlr)

        @test isapprox(giv, giv_ref; atol=300ε, rtol=0)

        # Comparison on τ
        smpl_τ = TauSampling(basis)
        gτ = evaluate(smpl_τ, Gl)

        smpl_τ_for_dlr = TauSampling(dlr)
        gτ2 = evaluate(smpl_τ_for_dlr, g_dlr)

        @test isapprox(gτ, gτ2; atol=300ε, rtol=0)
    end

    @testset "Boson" begin
        β = 2
        ωmax = 21
        ε = 1e-7
        basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε; sve_result=sve_logistic[β * ωmax])

        # G(iw) = sum_p coeff_p U^{DLR}(iw, omega_p)
        coeff = [1.1, 2.0]
        ω_p = [2.2, -1.0]

        ρl_pole = basis_b.v(ω_p) * coeff
        gl_pole = -basis_b.s .* ρl_pole

        sp = DiscreteLehmannRepresentation(basis_b, ω_p)
        gl_pole2 = SparseIR.to_IR(sp, coeff)

        @test isapprox(gl_pole, gl_pole2; atol=300ε, rtol=0)
    end

    @testset "unit tests" begin
        @testset "MatsubaraPoles" begin
            poles = [2.0, 3.3, 9.3]
            β = π

            n = rand(-12345:2:987, 100)
            mpb = SparseIR.MatsubaraPoles{Fermionic}(β, poles)
            @test mpb(n) ≈ @. 1 / (im * n' - poles)

            n = rand(-234:2:13898, 100)
            mbp = SparseIR.MatsubaraPoles{Bosonic}(β, poles)
            @test mbp(n) ≈ @. tanh(π / 2 * poles) / (im * n' - poles)
        end

        @testset "DiscreteLehmannRepresentation" for stat in (Fermionic(), Bosonic())
            β = 10_000
            ωmax = 1
            ε = 1e-12
            basis = FiniteTempBasis(stat, β, ωmax, ε; sve_result=sve_logistic[β * ωmax])
            dlr = DiscreteLehmannRepresentation(basis)

            @test all(isone, SparseIR.significance(dlr))
            @test SparseIR.β(dlr) == β
            @test SparseIR.ωmax(dlr) == ωmax
            @test SparseIR.Λ(dlr) == β * ωmax
            @test SparseIR.sampling_points(dlr) ==
                  SparseIR.default_omega_sampling_points(basis)
            @test SparseIR.accuracy(dlr) < ε
            @test SparseIR.default_matsubara_sampling_points(dlr) ==
                  SparseIR.default_matsubara_sampling_points(basis)
        end
    end
end
