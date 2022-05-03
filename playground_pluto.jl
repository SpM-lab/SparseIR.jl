### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

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

# ╔═╡ 227eb6c4-d35f-4d6c-b79e-dc494088fc97
using BenchmarkTools

# ╔═╡ 953d4311-7ce9-4359-a7a2-b5f5f3192c65
md"""
```python
import sparse_ir as ir, numpy as np
basis = ir.FiniteTempBasis('F', 10, 4, 1e-6)
U = 0.5
def rhow(w):
  return np.sqrt(1 - w.clip(-1,1)**2) / np.pi
rho0l = basis.v.overlap(rhow)
G0l = -basis.s * rho0l
Gl = G0l
stau = ir.TauSampling(basis)
siw = ir.MatsubaraSampling(basis)
for _ in range(30):
  Gtau = stau.evaluate(Gl)
  Sigmatau = U**2 * Gtau**3
  Sigmal = stau.fit(Sigmatau)
  Sigmaiw = siw.evaluate(Sigmal)
  G0iw = siw.evaluate(G0l)
  Giw = 1/(1/G0iw - Sigmaiw)
  Gl = siw.fit(Giw)
```
"""

# ╔═╡ 0af86ef4-0740-4101-b209-2eb3d23002e3
basis = FiniteTempBasis(fermion, 10000.0, 4.0, 1e-6)

# ╔═╡ f636e647-c321-4625-9923-3ff0b8a37a06
U = 0.5

# ╔═╡ d6e782e5-5ec9-4cdf-b308-647c4ae37ce0
rhow(w) = √(1 - clamp(w, -1, 1)^2) / π

# ╔═╡ 353eedbc-664b-463d-b795-6f9887a81dc0
rho0l = overlap.(basis.v, rhow)

# ╔═╡ 1a2b33e9-6d3a-4c7b-b05a-253a13509dfe
G0l = - basis.s .* rho0l

# ╔═╡ a12f1f1e-1f21-4cbd-ad8a-be0f2ed579bc
size(basis)

# ╔═╡ f4f0a07d-a579-4ed7-ada5-7e882cdaed55
begin
	Gl = G0l
	stau = TauSampling(basis);
	siw = MatsubaraSampling(basis);
	for _ in 1:30
		Gtau = evaluate(stau, Gl)
		Sigmatau = U^2 * Gtau.^3
		Sigmal = fit(stau, Sigmatau)
		Sigmaiw = evaluate(siw, Sigmal)
		G0iw = evaluate(siw, G0l)
		Giw = 1 ./ (1 ./ G0iw - Sigmaiw)
		Glold = copy(Gl)
		Gl = fit(siw, Giw)
		println(norm(Gl - Glold))
	end
	real.(Gl)
end

# ╔═╡ d1eed5b7-4fdb-4a4e-a08f-4beb7f709dc9
moments(polyFTs::SparseIR.PiecewiseLegendreFTArray) = collect(eachrow(reduce(hcat, p.model.moments for p in polyFTs)))

# ╔═╡ 88ea79e3-03bd-4edc-9f04-7682a5e9d36d
function moments2(polyFTs::SparseIR.PiecewiseLegendreFTArray)
	n = length(first(polyFTs).model.moments)
	[[p.model.moments[i] for p in polyFTs] for i in 1:n]
end

# ╔═╡ fcc7578b-29b6-44dc-8ea0-cb2a0b96c6b3
function moments3(polyFTs::SparseIR.PiecewiseLegendreFTArray)
	n = length(first(polyFTs).model.moments)
	#[[p.model.moments[i] for p in polyFTs] for i in 1:n]
	mat = reduce(hcat, p.model.moments for p in polyFTs)
	eachrow(mat)
end

# ╔═╡ 9666b21b-4ef5-4588-9d41-1d43ccc1eadf
size(basis.uhat)

# ╔═╡ 1773939a-4ea4-4856-a769-e66e9018b3e6
@btime moments(basis.uhat)

# ╔═╡ b934f8e1-1ba1-4281-8e68-b927e10c2973
@btime moments2(basis.uhat)

# ╔═╡ 951a8aa2-0e42-4822-afa0-b5746842f3df
@btime moments3(basis.uhat)

# ╔═╡ e95e6459-83d2-4769-936c-c368e00dcc65
@time basis.uhat(10000003)

# ╔═╡ 2be2f599-6152-4a8f-bb7c-ef4293e2de9e
@time [û(10000003) for û in basis.uhat]

# ╔═╡ Cell order:
# ╠═842c7aec-b17a-11ec-127b-3f91b4687627
# ╠═c92e8377-29e9-4b29-ae11-e0423459722d
# ╠═36915eda-1b70-4fea-95f4-f252df56c060
# ╟─953d4311-7ce9-4359-a7a2-b5f5f3192c65
# ╠═0af86ef4-0740-4101-b209-2eb3d23002e3
# ╠═f636e647-c321-4625-9923-3ff0b8a37a06
# ╠═d6e782e5-5ec9-4cdf-b308-647c4ae37ce0
# ╠═353eedbc-664b-463d-b795-6f9887a81dc0
# ╠═1a2b33e9-6d3a-4c7b-b05a-253a13509dfe
# ╠═a12f1f1e-1f21-4cbd-ad8a-be0f2ed579bc
# ╠═f4f0a07d-a579-4ed7-ada5-7e882cdaed55
# ╠═227eb6c4-d35f-4d6c-b79e-dc494088fc97
# ╠═d1eed5b7-4fdb-4a4e-a08f-4beb7f709dc9
# ╠═88ea79e3-03bd-4edc-9f04-7682a5e9d36d
# ╠═fcc7578b-29b6-44dc-8ea0-cb2a0b96c6b3
# ╠═9666b21b-4ef5-4588-9d41-1d43ccc1eadf
# ╠═1773939a-4ea4-4856-a769-e66e9018b3e6
# ╠═b934f8e1-1ba1-4281-8e68-b927e10c2973
# ╠═951a8aa2-0e42-4822-afa0-b5746842f3df
# ╠═e95e6459-83d2-4769-936c-c368e00dcc65
# ╠═2be2f599-6152-4a8f-bb7c-ef4293e2de9e
