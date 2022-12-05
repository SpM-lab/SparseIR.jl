using SparseIR, Plots, LaTeXStrings

function main()
	basis, siω, Gl, Σl, β = compute()
	make_plot(basis, siω, Gl, Σl, β)
end

function compute(β = 10, ωmax = 8, ε = 1e-6)
    # Construct the IR basis and sparse sampling for fermionic propagators
    basis = FiniteTempBasis{Fermionic}(β, ωmax, ε)
    sτ = TauSampling(basis)
    siω = MatsubaraSampling(basis; positive_only=true)
    
    # Solve the single impurity Anderson model coupled to a bath with a
    # semicircular density of states with unit half bandwidth.
    U = 1.2
    ρ₀(ω) = 2/π * √(1 - clamp(ω, -1, +1)^2)
    
    # Compute the IR basis coefficients for the non-interacting propagator
    ρ₀l = overlap.(basis.v, ρ₀)
    G₀l = -basis.s .* ρ₀l
    
    # Self-consistency loop: alternate between second-order expression for the
    # self-energy and the Dyson equation until convergence.
    Gl = copy(G₀l)
    Gl_prev = zero(Gl)
    G₀iω = evaluate(siω, G₀l)
	Σl = 0
    while !isapprox(Gl, Gl_prev, atol=ε)
        Gl_prev = copy(Gl)
        Gτ = evaluate(sτ, Gl)
        Στ = @. U^2 * Gτ^3
        Σl = fit(sτ, Στ)
        Σiω = evaluate(siω, Σl)
        Giω = @. (G₀iω^-1 - Σiω)^-1
        Gl = fit(siω, Giω)
    end
	basis, siω, Gl, Σl, β
end

function make_plot(basis, siω, Gl, Σl, β)
	wsample = SparseIR.sampling_points(siω)
	box = collect(first(wsample):last(wsample))[1:43]
	siω_box = MatsubaraSampling(basis; positive_only=true, sampling_points=box)
	plot(SparseIR.value.(box, β), imag(evaluate(siω_box, Σl)); 
		ylims=(-0.13, 0), 
		marker=(:+, 5, 1.0, :red, stroke(2)), 
		line=(:dot, :red), 
		label=L"\operatorname{Im}\;\hat\Sigma\,(\mathrm{i}\omega)",
		xaxis=L"$\omega$ (Matsubara frequency)",
		legend=:topleft,
		background_color=RGBA(1,1,1,0),
		dpi=600)
	scatter!(SparseIR.value.(wsample, β)[1:end-1], imag(evaluate(siω, Σl));
		marker=(:x, 5, 1.0, :black, stroke(3)),
		label="sampling points")
	plot!(eachindex(Gl)[1:2:end], abs.(Gl[1:2:end] / first(Gl));
		inset=(1, bbox(0.05, 0.15, 0.5, 0.5, :bottom, :right)),
		subplot=2,
		label=L"\left| G_l / G_1 \right|",
		marker=(:+, 4, :blue, stroke(2)),
		line=(:dash, :blue),
		xaxis=(L"$l$ (expansion order)"),
	 	yminorgrid=true, yticks=[1e-6, 1e-4, 1e-2, 1])
	plot!(eachindex(Σl)[1:2:end], abs.(Σl[1:2:end] / first(Σl));
		subplot=2,
		label=L"\left|\Sigma_l/\Sigma_1\right|",
		marker=(:x, 4, :red, stroke(2)),
		line=(:dash, :red))
	plot!(eachindex(basis.s), basis.s / first(basis.s);
		subplot=2,
		ylims=(1e-6, 1),
		yaxis=:log,
		label=L"S_l / S_1",
		marker=(:+, 2, :black),
		line=(:dot, :black))
	savefig("readme_plot.png")
end

main()
