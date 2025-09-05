### A Pluto.jl notebook ###
# v0.20.10

using Markdown
using InteractiveUtils

# ╔═╡ 66ed6d9a-4e36-11f0-1f7c-59baebea7d28
begin
    using Pkg
    using Random

    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    using SparseIR
    import SparseIR as SparseIR
    using Plots
    gr() # USE GR backend
    using OMEinsum
    using LaTeXStrings
end

# ╔═╡ a8a7b92c-f9bd-4d47-b765-cfd267ac4644
begin
    function rho(omega)
        return abs.(omega) < 1 ? (2/π) .* sqrt.(1-omega^2) : 0.0
    end

    beta = 10000.0
    wmax = 1.0
    eps = 1e-15 # cutoff for SVD
    basis = FiniteTempBasis(Fermionic(), beta, wmax, eps)

    rhol = overlap(basis.v, rho)
    gl = - basis.s .* rhol

    ls = collect(0:(length(basis) - 1))
    p = plot(; marker=:x, yaxis=:log, ylabel=L"|g_l|", xlabel=L"l", ylims=(1e-15, 10))
    plot!(p, ls[1:2:end], abs.(gl[1:2:end]); marker=:x,
        yaxis=:log, ylabel=L"|g_l|", xlabel=L"l")
end

# ╔═╡ dbfea9c2-8464-4c66-991f-b42d08fbc44e
begin
    smpl_tau = TauSampling(basis)
    println("sampling times: ", smpl_tau.sampling_points)
    println("Condition number: ", SparseIR.cond(smpl_tau))
end

# ╔═╡ a825a7b9-0f17-4512-a697-f993120e7661
begin
    # Evaluate G(τ) on the sampling times
    gtau_smpl = evaluate(smpl_tau, gl)

    plot(smpl_tau.sampling_points, gtau_smpl; marker=:x, xlabel=L"\tau", ylabel=L"G(\tau)")
end

# ╔═╡ f8cffacc-100c-4735-9eb8-9e853e020c25
# Fit G(τ) on the sampling times
gl_reconst_from_tau = fit(smpl_tau, gtau_smpl)

# ╔═╡ 903b2c67-4a42-4ece-84af-03f549b5b091
begin
    smpl_matsu = MatsubaraSampling(basis)
    println("sampling frequencies: ", smpl_matsu.sampling_points)
    println("Condition number: ", SparseIR.cond(smpl_matsu))
end

# ╔═╡ 105bf4eb-1de7-49d3-a68f-8e34e4c2b571
begin
    # Evaluate G(iv) on the sampling frequencies
    giv_smpl = evaluate(smpl_matsu, gl)

    # `value` function evaluate the actual values of Matsubara frequencies
    plot(SparseIR.value.(smpl_matsu.ωn, beta), imag.(giv_smpl);
        marker=:x, xlabel=L"\nu", ylabel=L"\mathrm{Im} G(i\nu)")
end

# ╔═╡ 9976c89e-b860-4c22-8af0-e5f8eb3017b2
# Fit G(τ) on the sampling times
gl_reconst_from_matsu = fit(smpl_matsu, giv_smpl)

# ╔═╡ 92a4e88c-2bb7-4560-86d5-2dfa5e73d6d9
let
    p = plot(; xlabel=L"l", ylabel=L"g_l", ylims=(1e-17, 10), yaxis=:log)
    plot!(p, ls[1:2:end], abs.(gl[1:2:end]); marker=:none, label="Exact")
    plot!(p, ls[1:2:end], abs.(gl_reconst_from_tau[1:2:end]);
        marker=:x, label="from sampling times")
    plot!(p, ls[1:2:end], abs.(gl_reconst_from_matsu[1:2:end]);
        marker=:+, label="from sampling frequencies")
end

# ╔═╡ 3662a96e-818e-40f8-8ced-2b18f8bca932
let
    p = plot(; xlabel=L"L", ylabel=L"Error in $g_l$", ylims=(1e-18, 10), yaxis=:log)
    plot!(p, ls[1:2:end], abs.((gl_reconst_from_tau - gl)[1:2:end]);
        marker=:x, label="from sampling times")
    plot!(p, ls[1:2:end], abs.((gl_reconst_from_matsu - gl)[1:2:end]);
        marker=:+, label="from sampling frequencies")
end

# ╔═╡ Cell order:
# ╠═66ed6d9a-4e36-11f0-1f7c-59baebea7d28
# ╠═a8a7b92c-f9bd-4d47-b765-cfd267ac4644
# ╠═dbfea9c2-8464-4c66-991f-b42d08fbc44e
# ╠═a825a7b9-0f17-4512-a697-f993120e7661
# ╠═f8cffacc-100c-4735-9eb8-9e853e020c25
# ╠═903b2c67-4a42-4ece-84af-03f549b5b091
# ╠═105bf4eb-1de7-49d3-a68f-8e34e4c2b571
# ╠═9976c89e-b860-4c22-8af0-e5f8eb3017b2
# ╠═92a4e88c-2bb7-4560-86d5-2dfa5e73d6d9
# ╠═3662a96e-818e-40f8-8ced-2b18f8bca932
