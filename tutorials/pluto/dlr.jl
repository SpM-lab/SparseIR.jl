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
    using LaTeXStrings
end

# ╔═╡ a8a7b92c-f9bd-4d47-b765-cfd267ac4644
begin
    wmax = 1.0
    lambda_ = 1e+4
    beta = lambda_/wmax

    basis = FiniteTempBasis(Fermionic(), beta, wmax, 1e-15)
    print(length(basis))
end

# ╔═╡ dbfea9c2-8464-4c66-991f-b42d08fbc44e
begin
    rho(omega) = sqrt(1-omega^2)/sqrt(0.5*π)

    omega = LinRange(-wmax, wmax, 1000)
    plot(omega, rho.(omega); xlabel=latexstring("\\omega"), ylabel=latexstring("\\rho(\\omega)"))
end

# ╔═╡ a825a7b9-0f17-4512-a697-f993120e7661
begin
    rhol = overlap(basis.v, rho)
    ls = collect(1:length(basis))
    plot(ls[1:2:end], abs.(rhol)[1:2:end]; marker=:cross, yaxis=:log, ylims=(1e-5, 1))
end

# ╔═╡ f8cffacc-100c-4735-9eb8-9e853e020c25
begin
    gl = - basis.s .* rhol
    plot(ls[1:2:end], abs.(gl)[1:2:end]; marker=:cross, ylims=(1e-5, 10), yaxis=:log)
end

# ╔═╡ 4ed0d27c-ba70-4ba3-a0da-0a5e17fb6e4e
begin
    dlr = DiscreteLehmannRepresentation(basis)

    # To DLR
    g_dlr = SparseIR.from_IR(dlr, gl)

    plot(dlr.poles, g_dlr; marker=:cross)
end

# ╔═╡ 8228e5c2-a201-41d3-ab8c-8313845e3030
begin
    # Transform back to IR from DLR
    gl_reconst = SparseIR.to_IR(dlr, g_dlr)

    plot(
        [abs.(gl), abs.(gl_reconst), abs.(gl-gl_reconst)];
        label=["Exact" "Reconstructed from DLR" "error"],
        marker=[:cross :x :circle], line=(nothing, nothing, nothing), yaxis=:log,
        ylims=(1e-18, 10)
    )
end

# ╔═╡ 41ab2929-69b8-44ae-9856-0a6c82320f41
begin
    #v = 2 .* collect(-1000:10:1000) .+ 1
    v = FermionicFreq.(2 .* collect(-1000:10:1000) .+ 1)
    iv = SparseIR.valueim.(v, beta)

    newaxis = [CartesianIndex()]
    transmat = 1 ./ (iv[:, newaxis] .- dlr.poles[newaxis, :])
    giv = transmat * g_dlr

    smpl = MatsubaraSampling(basis; sampling_points=v)
    giv_exact = evaluate(smpl, gl)

    plot(
        imag.(iv), [imag.(giv_exact), imag.(giv)]; marker=[:cross :x], line=(
            nothing, nothing),
        xlabel=latexstring("\\nu"),
        ylabel=latexstring("G(\\mathrm{i}\\omega_n)")
    )
end

# ╔═╡ Cell order:
# ╠═66ed6d9a-4e36-11f0-1f7c-59baebea7d28
# ╠═a8a7b92c-f9bd-4d47-b765-cfd267ac4644
# ╠═dbfea9c2-8464-4c66-991f-b42d08fbc44e
# ╠═a825a7b9-0f17-4512-a697-f993120e7661
# ╠═f8cffacc-100c-4735-9eb8-9e853e020c25
# ╠═4ed0d27c-ba70-4ba3-a0da-0a5e17fb6e4e
# ╠═8228e5c2-a201-41d3-ab8c-8313845e3030
# ╠═41ab2929-69b8-44ae-9856-0a6c82320f41
