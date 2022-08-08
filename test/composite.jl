using Test
using SparseIR

@testset "composite.jl" begin

    @testset "Augmented basis with stat = $stat" for stat in (fermion, boson)
        wmax = 2.0
        β = 1e3
        ϵ = 1e-6

        basis = FiniteTempBasis(stat, β, wmax, ϵ)
        basis_legg = LegendreBasis(stat, β, 2)
        basis_comp = CompositeBasis([basis_legg, basis])

        @test basis_comp.u(1.0) ≈ vcat(basis_legg.u(1.0), basis.u(1.0))

        v = 0.1:0.1:1.0
        @test basis_comp.u(v) ≈ vcat(basis_legg.u(v), basis.u(v))

        n = MatsubaraFreq(6 + SparseIR.zeta(stat))
        res = vcat(basis_legg.uhat(n), basis.uhat(n))
        @test basis_comp.uhat(n) ≈ res
        @test basis_comp.uhat(Integer(n)) ≈ res

        nn = n : n+4*pioverbeta
        res = vcat(basis_legg.uhat(nn), basis.uhat(nn))
        @test basis_comp.uhat(nn) ≈ res
        @test basis_comp.uhat(Integer.(nn)) ≈ res

        # G(τ) = c - e^{-τ*pole}/(1 - e^{-β*pole})
        pole = 1.0
        c = 1e-2
        τ_smpl = TauSampling(basis_comp)
        gτ = c .- exp.(-τ_smpl.sampling_points * pole) / (1 - exp(-β * pole))
        gl_from_τ = fit(τ_smpl, gτ)

        gτ_reconst = evaluate(τ_smpl, gl_from_τ)
        @test isapprox(gτ, gτ_reconst; atol=1e-14 * maximum(abs, gτ), rtol=0)

        sgn = SparseIR.significance(basis_comp)
        @test issorted(sgn; rev=true)
        @test all(<=(1), sgn)
        @test all(>=(ϵ), sgn)
    end

end
