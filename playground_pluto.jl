### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 842c7aec-b17a-11ec-127b-3f91b4687627
begin
	using Revise
	using Pkg
	Pkg.activate(".")
	using SparseIR
end

# ╔═╡ c92e8377-29e9-4b29-ae11-e0423459722d
using LinearAlgebra, Random

# ╔═╡ 36915eda-1b70-4fea-95f4-f252df56c060
# ╠═╡ disabled = true
#=╠═╡
using Plots, PlutoUI
  ╠═╡ =#

# ╔═╡ b8c2c8c7-9a42-4394-a390-aecf9a1c6f34
basis = IRBasis(boson, 10)

# ╔═╡ 76e9300e-5464-4910-bc2a-02c7521920ac
extrema(basis.uhat[1])

# ╔═╡ 8886718d-414b-4d87-a85d-0c5638e440c7
@bind n Slider(eachindex(basis.u))

# ╔═╡ 48dc95f8-fa62-4857-9d4a-b336832677d0
let
	x = -1:0.0001:1
	p = plot()
	poly = basis.u[n]
	plot!(p, x, poly, label="poly")
	for i in 1:6#basis.u.polyorder
		poly = deriv(poly)
		plot!(p, x, x -> poly(x) / (2basis.kernel.Λ)^i, label="derivative $i")
	end
	p
end

# ╔═╡ 60508545-94c1-4af0-b5ed-a128c8ef8aef
let
	x = -1:0.0001:1
	p = plot()
	poly = basis.u[n]
	plot!(p, x, x -> 10deriv(poly, 5)(x))
	plot!(p, x, deriv(poly, 6))
	p
end

# ╔═╡ Cell order:
# ╠═842c7aec-b17a-11ec-127b-3f91b4687627
# ╠═c92e8377-29e9-4b29-ae11-e0423459722d
# ╠═36915eda-1b70-4fea-95f4-f252df56c060
# ╠═b8c2c8c7-9a42-4394-a390-aecf9a1c6f34
# ╠═76e9300e-5464-4910-bc2a-02c7521920ac
# ╠═8886718d-414b-4d87-a85d-0c5638e440c7
# ╠═48dc95f8-fa62-4857-9d4a-b336832677d0
# ╠═60508545-94c1-4af0-b5ed-a128c8ef8aef
