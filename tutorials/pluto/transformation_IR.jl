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

# ╔═╡ aa3e9399-14fb-4486-8d68-608a4fa42cef
begin
    beta = 15.0
    wmax = 10.0
    basis_b = FiniteTempBasis(Bosonic(), beta, wmax, 1e-7)

    coeff = [1.0]
    omega_p = [0.1]

    rhol_pole = ein"lp,p->l"(
        basis_b.v(omega_p),
        coeff ./ tanh.(0.5 * beta * omega_p)
    )
    gl_pole = -basis_b.s .* rhol_pole

    plot([abs.(rhol_pole), abs.(gl_pole)]; yaxis=:log,
        label=[latexstring("|\\rho_l|") latexstring("|g_l|")],
        xlabel=latexstring("l"), ylims=(1e-5, 1e+1))
end

# ╔═╡ ebbff3a0-2db1-4828-9d21-db421b1311ac
begin
    dlr = DiscreteLehmannRepresentation(basis_b, omega_p)
    gl_pole2 = to_IR(dlr, coeff ./ tanh.(0.5 * beta * omega_p))

    plot(
        [abs.(gl_pole2), abs.(gl_pole)];
        label=["from DLR" "exact"],
        yaxis=:log,
        xlabel=latexstring("l"), ylabel=latexstring("g_l"), ylims=(1e-5, 1e+1), marker=[:cross :circle])
end

# ╔═╡ be191751-2c1d-47af-bf13-9c1c9c0b587b
begin
    # Three Gaussian peaks (normalized to 1)
    gaussian(x, mu, sigma) = exp(-((x - mu) / sigma)^2) / (sqrt(π) * sigma)

    function rho(omega)
        0.2 * gaussian(omega, 0.0, 0.15) +
        0.4 * gaussian(omega, 1.0, 0.8) + 0.4 * gaussian(omega, -1.0, 0.8)
    end

    omegas = LinRange(-5, 5, 1000)
    plot(omegas, rho.(omegas); xlabel=latexstring("\\omega"),
        ylabel=latexstring("\\rho(\\omega)"), label="")
end

# ╔═╡ a0b245a7-cd71-4095-9e87-941f556ae876
begin
    basis = FiniteTempBasis(Fermionic(), beta, wmax, 1e-7)

    rhol = [overlap(basis.v[l], rho) for l in 1:length(basis)]
    gl = -basis.s .* rhol

    plot([abs.(rhol), abs.(gl)]; yaxis=:log, ylims=(1e-5, 1),
        marker=[:circle :diamond], line=nothing, xlabel=latexstring("l"),
        label=[latexstring("\\rho_l") latexstring("|g_l|")])
end

# ╔═╡ 696b71ee-b326-44b0-94d2-8e2335a988a4
begin
    rho_omega_reconst = transpose(basis.v(collect(omegas))) * rhol

    plot(omegas, rho_omega_reconst; xlabel=latexstring("\\omega"),
        ylabel=latexstring("\\rho(\\omega)"), label="")
end

# ╔═╡ fe181719-aa12-433b-8964-ec8c879bcb65
begin
    taus = collect(LinRange(0, beta, 1000))
    gtau1 = transpose(basis.u(taus)) * gl

    plot(taus, gtau1; xlabel=latexstring("\\tau"),
        ylabel=latexstring("G(\\tau)"), label=nothing)

    smpl = TauSampling(basis; sampling_points=taus)
    gtau2 = evaluate(smpl, gl)
    plot(taus, gtau2; xlabel=latexstring("\\tau"),
        ylabel=latexstring("G(\\tau)"), label=nothing, marker=:cross)
end

# ╔═╡ 49f92304-504c-4fc6-8f25-e224a7bedce1
begin
    function eval_gtau(taus)
        uval = basis.u(taus) #(nl, ntau)
        return transpose(uval) * gl
    end

    gl_reconst = [overlap(basis.u[l], eval_gtau) for l in 1:length(basis)]

    ls = collect(0:(length(basis) - 1))
    plot(
        ls[1:2:end],
        [abs.(gl_reconst[1:2:end]), abs.(gl[1:2:end]), abs.(gl_reconst - gl)[1:2:end]]; xlabel=latexstring("l"), label=["reconstructed" "exact" "error"],
        marker=[:+ :x :none], markersize=10, yaxis=:log)
end

# ╔═╡ e805d778-5b12-47a1-84e0-39f964c30aff
begin
    rng = Xoshiro(100)
    shape = (1, 2, 3)
    newaxis = [CartesianIndex()]
    gl_tensor = randn(rng, shape...)[:, :, :, newaxis] .* gl[newaxis, newaxis, newaxis, :]
    println("gl: ", size(gl))
    println("gl_tensor: ", size(gl_tensor))
end

# ╔═╡ 6bec1a40-f70a-4940-8498-d9ed780ad69f
begin
    smpl_matsu = MatsubaraSampling(basis)
    gtau_tensor = evaluate(smpl_matsu, gl_tensor; dim=4)
    print("gtau_tensor: ", size(gtau_tensor))
    gl_tensor_reconst = fit(smpl_matsu, gtau_tensor; dim=4)
    @assert isapprox(gl_tensor, gl_tensor_reconst)
end

# ╔═╡ Cell order:
# ╠═66ed6d9a-4e36-11f0-1f7c-59baebea7d28
# ╠═aa3e9399-14fb-4486-8d68-608a4fa42cef
# ╠═ebbff3a0-2db1-4828-9d21-db421b1311ac
# ╠═be191751-2c1d-47af-bf13-9c1c9c0b587b
# ╠═a0b245a7-cd71-4095-9e87-941f556ae876
# ╠═696b71ee-b326-44b0-94d2-8e2335a988a4
# ╠═fe181719-aa12-433b-8964-ec8c879bcb65
# ╠═49f92304-504c-4fc6-8f25-e224a7bedce1
# ╠═e805d778-5b12-47a1-84e0-39f964c30aff
# ╠═6bec1a40-f70a-4940-8498-d9ed780ad69f
