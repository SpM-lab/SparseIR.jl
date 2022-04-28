using Test
using SparseIR

@testset "composite.jl" begin
    """Augmented bosonic basis"""
    wmax = 2
    beta = 1e+3
    #startt = time_ns()
    walltimes = Tuple{String,Int}[]
    #push!(walltimes, ("A", time_ns()))
    basis = FiniteTempBasis(boson, beta, wmax, 1e-6)
    #push!(walltimes, ("A.1", time_ns()))
    basis = FiniteTempBasis(boson, beta, wmax, 1e-6)
    #push!(walltimes, ("B", time_ns()))
    basis_legg = LegendreBasis(boson, beta, 2)
    #push!(walltimes, ("C", time_ns()))
    basis_comp = CompositeBasis([basis_legg, basis])
    #push!(walltimes, ("D", time_ns()))

    # G(tau) = c - e^{-tau*pole}/(1 - e^{-beta*pole})
    pole = 1.0
    c = 1e-2
    #push!(walltimes, ("E", time_ns()))
    tau_smpl = TauSampling(basis_comp)
    #push!(walltimes, ("E.1", time_ns()))
    tau_smpl = TauSampling(basis_comp)
    #push!(walltimes, ("F", time_ns()))
    gtau = c .- exp.(-tau_smpl.sampling_points .* pole) ./ (1 - exp(-beta * pole))
    #push!(walltimes, ("G", time_ns()))
    gl_from_tau = fit(tau_smpl, gtau)
    #push!(walltimes, ("H", time_ns()))

    gtau_reconst = evaluate(tau_smpl, gl_from_tau)
    @test isapprox(gtau, gtau_reconst; atol=1e-14 * maximum(abs.(gtau)), rtol=0)
    #push!(walltimes, ("I", time_ns()))

    #for t in walltimes
    #println(t[1], "      :    ", (t[2]-startt) * 1e-9)
    #end
end
