### A Pluto.jl notebook ###
# v0.19.0

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
using Plots

# ╔═╡ a548888b-89fd-44da-be55-d12564eeaa1f
basis = IRBasis(fermion, 10.0)

# ╔═╡ 4902d7ce-4b3e-4f5b-81e9-3ec4aef5fc3e
A = randn(20, 30)

# ╔═╡ 714cf6f4-e3aa-4dda-8956-97cf3d8f2626
Ad = svd(A)

# ╔═╡ 94a62b3d-ad9a-48a9-8bbb-556a7fd1dabb
Matrix(Ad) ≈ A

# ╔═╡ 7cff054d-cb5b-456a-bf33-2e193cc2fefe
x = randn(30)

# ╔═╡ 5b3bdd16-1588-4fae-a950-08c6170216fc
A * x

# ╔═╡ 360d2c48-0e8d-41bb-9b44-356b7b54cfc4
y = randn(20)

# ╔═╡ 995778b3-15dd-4742-932f-469fcc589357
A * (A \ y)

# ╔═╡ 4fd9b33c-57c3-45c9-9077-c90ec34545c1
isapprox(A, Matrix(Ad), atol=1.9621954815178372e-14, rtol=0)

# ╔═╡ 02b3c94f-f84d-4776-81fc-9c8171aeee86
all(isapprox.(A, Matrix(Ad), atol=1.9621954815178372e-14, rtol=0))

# ╔═╡ 0c9fd9ea-1b18-4457-9cec-0d5e329f3058
minimum(1:10000) do i
	Random.seed!(i)
	A = randn(49, 39)
	Ad = svd(A)
	norm_A = first(Ad.S) / last(Ad.S)

	norm(A - Matrix(Ad)) / (1e-15 * norm_A)
end

# ╔═╡ 193cfa91-8d07-4f58-b9c2-209269c2bd6b
norm(A - Matrix(Ad))

# ╔═╡ f95ddc02-4d15-426f-9c50-05176127ab0d
let 
	a = rand(10)
	display(a)
	insert!(a, length(a) ÷ 2 + 1, 9998898)
	a
end

# ╔═╡ Cell order:
# ╠═842c7aec-b17a-11ec-127b-3f91b4687627
# ╠═c92e8377-29e9-4b29-ae11-e0423459722d
# ╠═36915eda-1b70-4fea-95f4-f252df56c060
# ╠═a548888b-89fd-44da-be55-d12564eeaa1f
# ╠═4902d7ce-4b3e-4f5b-81e9-3ec4aef5fc3e
# ╠═714cf6f4-e3aa-4dda-8956-97cf3d8f2626
# ╠═94a62b3d-ad9a-48a9-8bbb-556a7fd1dabb
# ╠═7cff054d-cb5b-456a-bf33-2e193cc2fefe
# ╠═5b3bdd16-1588-4fae-a950-08c6170216fc
# ╠═360d2c48-0e8d-41bb-9b44-356b7b54cfc4
# ╠═995778b3-15dd-4742-932f-469fcc589357
# ╠═4fd9b33c-57c3-45c9-9077-c90ec34545c1
# ╠═02b3c94f-f84d-4776-81fc-9c8171aeee86
# ╠═0c9fd9ea-1b18-4457-9cec-0d5e329f3058
# ╠═193cfa91-8d07-4f58-b9c2-209269c2bd6b
# ╠═f95ddc02-4d15-426f-9c50-05176127ab0d
