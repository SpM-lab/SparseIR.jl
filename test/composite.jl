using Test
using SparseIR

@testset "composite.jl" begin
    """Augmented bosonic basis"""
    wmax = 2
    beta = 1e+3
    basis = FiniteTempBasis(boson, beta, wmax, 1e-6)
    basis_legg = LegendreBasis(boson, beta, 2)
    basis_comp = CompositeBasis([basis_legg, basis])

    # G(tau) = c - e^{-tau*pole}/(1 - e^{-beta*pole})
    pole = 1.0
    c = 1e-2
    tau_smpl = TauSampling(basis_comp)
    gtau = c .- exp.(-tau_smpl.sampling_points .* pole) ./ (1 - exp(-beta * pole))
    gl_from_tau = fit(tau_smpl, gtau)

    gtau_reconst = evaluate(tau_smpl, gl_from_tau)
    @test isapprox(gtau, gtau_reconst; atol=1e-14 * maximum(abs.(gtau)), rtol=0)
end
