### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 66ed6d9a-4e36-11f0-1f7c-59baebea7d28
begin
	using Pkg
	using Revise
	using Random

	Pkg.activate(joinpath(@__DIR__, "..", ".."))
	using SparseIR
	import SparseIR as SparseIR
	
	using OMEinsum
	using FastGaussQuadrature
	using SparseIR: valueim, value, MatsubaraSampling64F, TauSampling64F
	using LaTeXStrings
	using Plots
	using LinearAlgebra
end

# ╔═╡ a8a7b92c-f9bd-4d47-b765-cfd267ac4644
begin
	"""
	Compute spectral function rho(omega)
	"""
	rho_omega(omega::Float64) = sqrt(4 - omega^2) / (2*pi)
	
	num_omega = 100
	omegas = collect(range(-2, 2, length=num_omega))
	rho = rho_omega.(omegas)
	p = plot(xlabel="\$\\omega\$", ylabel = "\$\\rho(\\omega)\$", size=(400,300))
	plot!(omegas, rho, label=nothing)
end

# ╔═╡ dbfea9c2-8464-4c66-991f-b42d08fbc44e
begin
	"""
	Gauss-Legendre quadrature rule of degree `deg` for the interval [xmin, xmax]
	"""
	function _gausslegendre(deg::Int, xmin::Float64, xmax::Float64)
	    @assert xmin < xmax
	    x, w = gausslegendre(deg)
	    dx = xmax - xmin
	    w_ = 0.5 * dx * w
	    x_ = (0.5 * dx) * (x .+ 1) .+ xmin
	    return x_, w_
	end
end

# ╔═╡ e774da2a-2465-474a-947a-69581c9d1302
"""
Solver
"""
struct IPTSolver
    U::Float64
    basis
    beta::Float64
    rho_omega::Function
    omega_range::Tuple{Float64,Float64}
    quad_rule::Tuple{Vector{Float64}, Vector{Float64}}
    smpl_matsu::MatsubaraSampling64F
    smpl_tau::TauSampling64F

    function IPTSolver(U::Float64, basis, rho::Function,
        omega_range::Tuple{Float64,Float64}, deg_leggaus::Int64=100)
        quad_rule = _gausslegendre(deg_leggaus, omega_range...)
        smpl_matsu = MatsubaraSampling(basis)
        smpl_tau = TauSampling(basis)

        new(U, basis, SparseIR.beta(basis), rho_omega, omega_range, quad_rule, smpl_matsu, smpl_tau)
    end
end

# ╔═╡ 32a4f2b2-19e8-45e3-b05e-e43001c0a9f7


# ╔═╡ 876a5253-5f09-460a-8d09-b80109d5701d
begin
	""" Solve Dyason equation: Compute G_loc(iv) from Sigma_iv """
	function compute_gloc_iv(solver::IPTSolver, sigma_iv::Vector{Complex{Float64}})
	    #iv = 1im .* solver.smpl_matsu.sampling_points .* π / solver.beta
	    iv = valueim.(solver.smpl_matsu.sampling_points, solver.beta)
	    newaxis = [CartesianIndex()]
	    gloc_iv = ein"q,q,wq->w"(
	            solver.quad_rule[2],
	            solver.rho_omega.(solver.quad_rule[1]),
	            1 ./ ((iv .- sigma_iv)[:, newaxis] .- solver.quad_rule[1][newaxis, :])
	        )
	    return gloc_iv
	end
	
	""" Compute IPT Sigma(iv) from mathcal{G}(iv)"""
	function compute_sigma_iv(solver::IPTSolver, g0_iv::Vector{ComplexF64})
	    g0_IR = fit(solver.smpl_matsu, g0_iv, dim=1)
	    g0_tau = evaluate(solver.smpl_tau, g0_IR)
	    sigma_tau = (solver.U)^2 .* (g0_tau).^2 .* (-1 .* g0_tau[end:-1:1])
	    sigma_IR = fit(solver.smpl_tau, sigma_tau)
	    return evaluate(solver.smpl_matsu, sigma_IR, dim=1)
	end
	
	""" Perform one self-consistent loop """
	function onestep(solver::IPTSolver, sigma_iv::Vector{Complex{Float64}})
	    gloc_iv = compute_gloc_iv(solver, sigma_iv)
	    g0_iv = 1 ./ ((1 ./ gloc_iv) .+ sigma_iv)
	    return compute_sigma_iv(solver, im .* imag.(g0_iv))
	end
end

# ╔═╡ b4002202-0c6d-499f-8bce-bf23eda0fe85
function renormalization_factor(solver::IPTSolver, sigma_iv0::Union{Nothing,Vector{ComplexF64}}, nitr::Int, mixing::Float64)
	# Initialize sigma_iv
	if isnothing(sigma_iv0)
		sigma_iv = zeros(ComplexF64,length(solver.smpl_matsu.sampling_points))
	else
		sigma_iv = copy(sigma_iv0)
	end

	# mixing sigma_iv with new_sigma_iv
	for i in 1:nitr
		new_sigma_iv = onestep(solver,sigma_iv)
		rdiff = sum(abs.(new_sigma_iv .- sigma_iv)) / sum(abs.(new_sigma_iv))
		if rdiff < 1e-14
			break
		end
		sigma_iv = mixing .* new_sigma_iv .+ (1-mixing) .* sigma_iv
	end

	center = Int(length(sigma_iv)/2) + 1 # julia
	delta_sigma_iv = sigma_iv[center+1]-sigma_iv[center]
	inv_z = (1 - imag(delta_sigma_iv)*solver.beta/(2*pi))

	return 1/inv_z, sigma_iv
end

# ╔═╡ 32b41074-6932-43c3-ab2f-77209cc6289a
begin
	function sweepU(basis, D::Float64, U_range::Vector{Float64}, rho_omega::Function, omega_range::Tuple{Float64, Float64}, nitr::Int, mixing::Float64)
	    Z = Float64[]
	    sigma_history = []
	    for iu in U_range
	        solver = IPTSolver(iu, basis, rho_omega, omega_range)
	        sigma_iv0 = length(sigma_history) > 0 ? sigma_history[end][1] : nothing
	        z, sigma_ = renormalization_factor(solver, sigma_iv0, nitr, mixing)
	        push!(sigma_history, (sigma_, solver.smpl_matsu.sampling_points))
	        if z > 0
	            push!(Z, z)
	        else
	            push!(Z,0.0)
	        end
	    end
	
	    return Z, sigma_history
	end
	
	# Setup parameters
	D = 2.0
	temp = 0.1 / D
	beta = 1 / temp
	wmax = 10.0 * D
	lambda = beta * wmax
	eps = 1e-15
	wmax = lambda / beta
	basis = FiniteTempBasis(Fermionic(), beta, wmax, eps)
	
	# Perform a sweep by increasing U
	num_U = 50
	U_min = 0.0 / D
	U_max = 7.5 / D
	U_range = 2 .* collect(range(U_min, U_max, num_U))
	omega_range = (-2.0, 2.0)
	nitr = 1000
	mixing = 0.25
	Z, sigma_history = sweepU(basis, D, U_range, rho_omega, omega_range, nitr, mixing);
end

# ╔═╡ eb8bc670-ce2d-4f3f-b432-652acc61be38


# ╔═╡ 1b585b75-f92c-4be7-91d4-7001cf018348
begin
	sigma = imag.(sigma_history[begin][1])
	v = value.(sigma_history[begin][2], beta)
	p1 = plot(xlabel="\$\\nu\$", ylabel = "\$\\mathrm{Im}\\ \\Sigma(\\mathrm{i}\\nu)\$",
	        title=latexstring("U=$(U_range[begin]),T=0.05"), size=(400,500))
	plot!(v, sigma, marker=:circle, label=nothing);
	
	sigma = imag.(sigma_history[end][1])
	v = value.(sigma_history[end][2], beta)
	p2 = plot(xlabel="\$\\nu\$", ylabel = "\$\\mathrm{Im}\\ \\Sigma(\\mathrm{i}\\nu)\$",
	        title=latexstring("U=$(U_range[end]),T=0.05"), size=(400,500))
	plot!(v, sigma, marker=:circle, label=nothing)
	
	plot(p1,p2,layout=(2,1))
end

# ╔═╡ 45b5dc82-88ed-427f-adb2-7489e8328b85
let
	p = plot(xlabel="\$U\$", ylabel = "\$z=(1-\\frac{\\partial \\Sigma}{\\partial \\omega})^{-1}\$",
	        title=latexstring("T=0.05"), size=(400,300))
	plot!(U_range, Z, marker=:circle, label="Ours", xlim=(1.0,7.5))
end

# ╔═╡ Cell order:
# ╠═66ed6d9a-4e36-11f0-1f7c-59baebea7d28
# ╠═a8a7b92c-f9bd-4d47-b765-cfd267ac4644
# ╠═dbfea9c2-8464-4c66-991f-b42d08fbc44e
# ╠═e774da2a-2465-474a-947a-69581c9d1302
# ╠═32a4f2b2-19e8-45e3-b05e-e43001c0a9f7
# ╠═876a5253-5f09-460a-8d09-b80109d5701d
# ╠═b4002202-0c6d-499f-8bce-bf23eda0fe85
# ╠═32b41074-6932-43c3-ab2f-77209cc6289a
# ╠═eb8bc670-ce2d-4f3f-b432-652acc61be38
# ╠═1b585b75-f92c-4be7-91d4-7001cf018348
# ╠═45b5dc82-88ed-427f-adb2-7489e8328b85
