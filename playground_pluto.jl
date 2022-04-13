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

# ╔═╡ 745d4cf2-0d33-4b24-9546-8553fc0b253e
using Tullio

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

# ╔═╡ f22a0ee2-0b79-4a2a-b35c-5c42954b8b79
a = rand(10, 11)

# ╔═╡ 36700b9e-4551-455d-b38d-00ef2255e60b
b = rand(13, 11, 14)

# ╔═╡ d09cea97-18c6-4db3-8de4-03ce7c659c33
size(a), size(b)

# ╔═╡ 84275989-ac41-4710-864e-4ba9a2030269
#    @tullio u_data[l, i, s] := cmat[l, x] * u_x[i, x, s]

# ╔═╡ e80d7334-740f-48d0-9657-d1b965619db4
@tullio c_ref[l, i, s] := a[l, x] * b[i, x, s];

# ╔═╡ 10cd6c57-0c8e-460b-afc6-7f5580c6c4a8
c = reshape(a * reshape(permutedims(b, (2, 1, 3)), (size(b, 2), :)), (size(a, 1), size(b, 1), size(b, 3)))

# ╔═╡ ae243a67-0be1-4eea-95dc-6af62e3faf6a
c == c_ref

# ╔═╡ 0c65dd2e-8ba2-4501-b5d8-1272c6979501
[a * transpose(b[:, :, i]) for i in 1:14]

# ╔═╡ 607cabef-5595-4daf-bac0-a697b88ff353
size(a .* permutedims(b, (1, 3, 2)))

# ╔═╡ 934b70ce-1675-416a-854d-49e77b641215
a .* permutedims(b, (2, 1, 3))

# ╔═╡ Cell order:
# ╠═842c7aec-b17a-11ec-127b-3f91b4687627
# ╠═c92e8377-29e9-4b29-ae11-e0423459722d
# ╠═36915eda-1b70-4fea-95f4-f252df56c060
# ╠═b8c2c8c7-9a42-4394-a390-aecf9a1c6f34
# ╠═76e9300e-5464-4910-bc2a-02c7521920ac
# ╠═8886718d-414b-4d87-a85d-0c5638e440c7
# ╠═48dc95f8-fa62-4857-9d4a-b336832677d0
# ╠═60508545-94c1-4af0-b5ed-a128c8ef8aef
# ╠═f22a0ee2-0b79-4a2a-b35c-5c42954b8b79
# ╠═36700b9e-4551-455d-b38d-00ef2255e60b
# ╠═d09cea97-18c6-4db3-8de4-03ce7c659c33
# ╠═84275989-ac41-4710-864e-4ba9a2030269
# ╠═745d4cf2-0d33-4b24-9546-8553fc0b253e
# ╠═e80d7334-740f-48d0-9657-d1b965619db4
# ╠═10cd6c57-0c8e-460b-afc6-7f5580c6c4a8
# ╠═ae243a67-0be1-4eea-95dc-6af62e3faf6a
# ╠═0c65dd2e-8ba2-4501-b5d8-1272c6979501
# ╠═607cabef-5595-4daf-bac0-a697b88ff353
# ╠═934b70ce-1675-416a-854d-49e77b641215
