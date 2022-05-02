### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                                           "AbstractPlutoDingetjes")].Bonds.initial_value
        catch
            b -> missing
        end
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
using Plots, PlutoUI

# ╔═╡ 8d00dcf9-e24e-41d7-af8a-a5e7e4ab3939
using PyCall

# ╔═╡ b8c2c8c7-9a42-4394-a390-aecf9a1c6f34
#basis = DimensionlessBasis(boson, 10)

# ╔═╡ 19d59ed7-aaea-4e47-9708-606acb03cdbb
begin
    ε = 1e-15
    stat = boson

    Λ = 1e4
    wmax = 1.0
    pole = 0.1 * wmax
    β = Λ / wmax
end

# ╔═╡ 1cb616f4-7957-4f9d-8942-c881b87b94bc
basis = FiniteTempBasis(stat, β, wmax, ε)

# ╔═╡ 8886718d-414b-4d87-a85d-0c5638e440c7
@bind n Slider(eachindex(basis.u))

# ╔═╡ 48dc95f8-fa62-4857-9d4a-b336832677d0
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ a9ce84bb-7f44-40f0-9558-6ba930d9fb6d
# ╠═╡ disabled = true
#=╠═╡
@bind d Slider(0:basis.u.polyorder, show_value=true)
  ╠═╡ =#

# ╔═╡ 60508545-94c1-4af0-b5ed-a128c8ef8aef
# ╠═╡ disabled = true
#=╠═╡
let
	x = -1:0.0001:1
	poly = basis.u[n]
	plot(x, deriv(poly, d))
end
  ╠═╡ =#

# ╔═╡ 2cd01891-3cdf-48f4-a60c-e2052d2c5a59
basis_py = pyimport("sparse_ir").FiniteTempBasis("B", β, wmax, ε)

# ╔═╡ 00bb8304-c307-4a2b-b5dc-9fb4991116a0
for i in 1:104
    #basis.uhat[i].model.moments .= basis_py.uhat[i-1]._model.moments
end

# ╔═╡ 6fa0439c-198f-4802-9e81-e25ac6fc580b
basis.uhat[2].model.moments

# ╔═╡ 23a0ff60-e431-40c9-b066-bbd787238f8f
basis_py.uhat.__getitem__(0:1)._model.moments

# ╔═╡ 53b9baba-4651-4d96-adf5-6122bdb41e97
vec(basis_py.uhat[0]._model.moments)

# ╔═╡ d7e3a562-8f93-4791-9ab8-d57d6bb6bc6a
begin
    stat_shift = (stat == fermion) ? 1 : 0
    weight = (stat == fermion) ? 1 : 1 / tanh(0.5 * Λ * pole / wmax)
    gl = -basis.s .* basis.v(pole) * weight
    func_G(n) = 1 / (im * (2n + stat_shift) * π / β - pole)
end

# ╔═╡ af735714-2fff-41de-aa2e-51c563313bb5
begin
    # Compute G(iwn) using unl
    matsu_test = Int[-1, 0, 1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
    prj_w = transpose(basis.uhat(2matsu_test .+ stat_shift))
    Giwn_t = prj_w * gl
end

# ╔═╡ e7c4cc78-57d8-4a9a-8282-051779375abe
# Compute G(iwn) from analytic expression
Giwn_ref = func_G.(matsu_test)

# ╔═╡ 682d942c-6442-44d8-99af-60324cdd85ae
basis.uhat[1:2](Int(4e5))

# ╔═╡ 7f17ea66-1c6c-49f0-87c6-dc226f15af59
magnitude = maximum(abs, Giwn_ref)

# ╔═╡ 7140a21f-ee18-4e33-86da-d622f69e9956
diff = abs.(Giwn_t - Giwn_ref)

# ╔═╡ 0f1b8564-b891-4910-a3df-4decb0ae7d96
tol = max(10 * last(basis.s) / first(basis.s), 1e-10)

# ╔═╡ e2898e3b-0aac-4909-a4bb-fbc6161fcc41
diff ./ magnitude

# ╔═╡ 27a6ffed-2e5f-4ee2-8368-f49e9226a13e
maximum(diff ./ magnitude)

# ╔═╡ 6ac780bd-d5dc-4230-96f3-8a6b71cf15cb
diff ./ Giwn_ref

# ╔═╡ ffb9a88b-7b65-47cb-9ee6-9bb31e433e2c
maximum(abs, diff ./ Giwn_ref)

# ╔═╡ Cell order:
# ╠═842c7aec-b17a-11ec-127b-3f91b4687627
# ╠═c92e8377-29e9-4b29-ae11-e0423459722d
# ╠═36915eda-1b70-4fea-95f4-f252df56c060
# ╠═b8c2c8c7-9a42-4394-a390-aecf9a1c6f34
# ╠═8886718d-414b-4d87-a85d-0c5638e440c7
# ╠═48dc95f8-fa62-4857-9d4a-b336832677d0
# ╠═a9ce84bb-7f44-40f0-9558-6ba930d9fb6d
# ╠═60508545-94c1-4af0-b5ed-a128c8ef8aef
# ╠═19d59ed7-aaea-4e47-9708-606acb03cdbb
# ╠═1cb616f4-7957-4f9d-8942-c881b87b94bc
# ╠═2cd01891-3cdf-48f4-a60c-e2052d2c5a59
# ╠═00bb8304-c307-4a2b-b5dc-9fb4991116a0
# ╠═6fa0439c-198f-4802-9e81-e25ac6fc580b
# ╠═23a0ff60-e431-40c9-b066-bbd787238f8f
# ╠═53b9baba-4651-4d96-adf5-6122bdb41e97
# ╠═d7e3a562-8f93-4791-9ab8-d57d6bb6bc6a
# ╠═af735714-2fff-41de-aa2e-51c563313bb5
# ╠═e7c4cc78-57d8-4a9a-8282-051779375abe
# ╠═682d942c-6442-44d8-99af-60324cdd85ae
# ╠═7f17ea66-1c6c-49f0-87c6-dc226f15af59
# ╠═7140a21f-ee18-4e33-86da-d622f69e9956
# ╠═0f1b8564-b891-4910-a3df-4decb0ae7d96
# ╠═e2898e3b-0aac-4909-a4bb-fbc6161fcc41
# ╠═27a6ffed-2e5f-4ee2-8368-f49e9226a13e
# ╠═6ac780bd-d5dc-4230-96f3-8a6b71cf15cb
# ╠═ffb9a88b-7b65-47cb-9ee6-9bb31e433e2c
# ╠═8d00dcf9-e24e-41d7-af8a-a5e7e4ab3939
