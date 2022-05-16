using Test
using SparseIR

@testset "composite.jl" begin
    @testset "Augmented bosonic basis" begin
        wmax = 2
        β = 1e+3
        basis = FiniteTempBasis(boson, β, wmax, 1e-6)
        basis_legg = LegendreBasis(boson, β, 2)
        basis_comp = CompositeBasis([basis_legg, basis])

        # G(τ) = c - e^{-τ*pole}/(1 - e^{-β*pole})
        pole = 1.0
        c = 1e-2
        τ_smpl = TauSampling(basis_comp)
        τ_smpl = TauSampling(basis_comp)
        gτ = c .- exp.(-τ_smpl.sampling_points .* pole) ./ (1 - exp(-β * pole))
        gl_from_τ = fit(τ_smpl, gτ)

        gτ_reconst = evaluate(τ_smpl, gl_from_τ)
        @test isapprox(gτ, gτ_reconst; atol=1e-14 * maximum(abs, gτ), rtol=0)
    end
end
