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

        Gl = to_IR(SparsePoleRepresentation(basis, poles), coeffs)
        g_spr = from_IR(spr, Gl)

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
        gl_pole2 = to_IR(sp, coeff)

        @test isapprox(gl_pole, gl_pole2; atol=300ε, rtol=0)
    end

    @testset "unit tests" begin
        @testset "MatsubaraPoleBasis" begin
            poles = [2.0, 3.3, 9.3]
            β = π

            n = rand(-12345:2:987, 100)
            mpb = SparseIR.MatsubaraPoleBasis(Fermionic(), β, poles)
            @test mpb(n) ≈ @. 1 / (im * n' - poles)
            
            n = rand(-234:2:13898, 100)
            mbp = SparseIR.MatsubaraPoleBasis(Bosonic(), β, poles)
            @test mbp(n) ≈ @. tanh(π / 2 * poles) / (im * n' - poles)
        end

        @testset "SparsePoleRepresentation" for stat in (Fermionic(), Bosonic())
            β = 10_000
            ωmax = 1
            ε = 1e-12
            basis = FiniteTempBasis(stat, β, ωmax, ε; sve_result=sve_logistic[β * ωmax])
            spr = SparsePoleRepresentation(basis)

            io = IOBuffer()
            show(io, spr)
            @test occursin(r"SparsePoleRepresentation for", String(take!(io)))
            @test all(isone, SparseIR.significance(spr))
            @test SparseIR.β(spr) == β
            @test SparseIR.ωmax(spr) == ωmax
            @test SparseIR.Λ(spr) == β * ωmax
            @test SparseIR.sampling_points(spr) == SparseIR.default_omega_sampling_points(basis)
            @test SparseIR.accuracy(spr) < ε
            @test SparseIR.default_matsubara_sampling_points(spr) == SparseIR.default_matsubara_sampling_points(basis)
        end
    end
end
