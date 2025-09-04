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
	gr()
	using RSCG
	using SparseArrays
	using LinearAlgebra
	
	using SparseIR: valueim
end

# ╔═╡ c895a51b-2597-4dbd-98a4-97d619f81700
begin
	# Check if a given function called with given types is type stable
	function typestable(@nospecialize(f), @nospecialize(t))
	    v = code_typed(f, t)
	    stable = true
	    for vi in v
	        for (name, ty) in zip(vi[1].slotnames, vi[1].slottypes)
	            !(ty isa Type) && continue
	            if ty === Any
	                stable = false
	                println("Type instability is detected! the variable is $(name) ::$ty")
	            end
	        end
	    end
	    return stable
	end
	
	function make_x_plushop(Nx,Ny,BC)
	    N = Nx*Ny
	    Txhop = spzeros(Int64,N,N)
	    for ix=1:Nx
	        for iy=1:Ny
	            i = (iy-1)*Nx + ix
	            jx = ix + 1
	            jy = iy
	            if BC == "PBC"
	                jx += ifelse(jx > Nx,-Nx,0)
	            elseif BC == "OBC"
	            else
	                error("BC = $BC is not supported")
	            end
	            if 1 <= jx <= Nx
	                j = (jy-1)*Nx + jx
	                Txhop[i,j] = 1
	            end
	        end
	    end
	    return Txhop
	end
	
	function make_x_minushop(Nx,Ny,BC)
	    N = Nx*Ny
	    Txhop = spzeros(Int64,N,N)
	    for ix=1:Nx
	        for iy=1:Ny
	            i = (iy-1)*Nx + ix
	            jx = ix - 1
	            jy = iy
	            if BC == "PBC"
	                jx += ifelse(jx < 1,Nx,0)
	            elseif BC == "OBC"
	            else
	                error("BC = $BC is not supported")
	            end
	            if 1 <= jx <= Nx
	                j = (jy-1)*Nx + jx
	                Txhop[i,j] = 1
	            end
	        end
	    end
	    return Txhop
	end
	
	function make_y_plushop(Nx,Ny,BC)
	    N = Nx*Ny
	    Tyhop = spzeros(Int64,N,N)
	    for ix=1:Nx
	        for iy=1:Ny
	            i = (iy-1)*Nx + ix
	            jx = ix 
	            jy = iy + 1
	            if BC == "PBC"
	                jy += ifelse(jy > Ny,-Ny,0)
	            elseif BC == "OBC"
	            else
	                error("BC = $BC is not supported")
	            end
	            if 1 <= jy <= Ny
	                j = (jy-1)*Nx + jx
	                Tyhop[i,j] = 1
	            end
	        end
	    end
	    return Tyhop
	end
	
	function make_y_minushop(Nx,Ny,BC)
	    N = Nx*Ny
	    Tyhop = spzeros(Int64,N,N)
	    for ix=1:Nx
	        for iy=1:Ny
	            i = (iy-1)*Nx + ix
	            jx = ix 
	            jy = iy - 1
	            if BC == "PBC"
	                jy += ifelse(jy < 1,Ny,0)
	            elseif BC == "OBC"
	            else
	                error("BC = $BC is not supported")
	            end
	            if 1 <= jy <= Ny
	                j = (jy-1)*Nx + jx
	                Tyhop[i,j] = 1
	            end
	        end
	    end
	    return Tyhop
	end
	
	function make_H_normal(Nx,Ny,μ,BC)
	    N = Nx*Ny
	    Tx_plushop = make_x_plushop(Nx,Ny,BC)
	    Tx_minushop = make_x_minushop(Nx,Ny,BC)
	    Ty_plushop = make_y_plushop(Nx,Ny,BC)
	    Ty_minushop = make_y_minushop(Nx,Ny,BC)
	    HN = sparse(I,N,N)*(-μ)
	    t = 1.0
	    
	    HN += -t*(Tx_plushop + Tx_minushop + Ty_plushop + Ty_minushop)
	    return HN
	end
	
	@assert typestable(make_x_plushop,(Int64,Int64,String))
	@assert typestable(make_x_minushop,(Int64,Int64,String))
	@assert typestable(make_y_plushop,(Int64,Int64,String))
	@assert typestable(make_y_minushop,(Int64,Int64,String))
end

# ╔═╡ 1786efb3-9df1-48ff-bf89-024bc3ca0ec7
begin
	function make_Δ(Δ)
	    Nx,Ny = size(Δ)
	    N = Nx*Ny
	    Δmat = spzeros(ComplexF64,N,N)
	    for ix=1:Nx
	        for iy=1:Ny
	            i = (iy-1)*Nx + ix
	            Δmat[i,i] = Δ[ix,iy]
	        end
	    end
	    return Δmat
	end
	@assert typestable(make_Δ,(Matrix{ComplexF64},))
end

# ╔═╡ 8ff8d624-3446-43cc-addd-865c69f2c43d
begin
	function make_H_sc(Nx,Ny,μ,Δ,BC)
	    HN = make_H_normal(Nx,Ny,μ,BC)
	    matΔ = make_Δ(Δ)
	    H = [
	        HN matΔ
	        matΔ' -conj.(HN)
	    ]
	    return H
	end
	@assert typestable(make_H_sc,(Int64,Int64,Float64,Matrix{ComplexF64},String))
end

# ╔═╡ fac94fce-082b-40c2-99ef-6c7057ac89a6
begin
	function update_H_sc!(H,Δ)
	    matΔ = make_Δ(Δ)
	    Nx,Ny = size(Δ)
	    N = Nx*Ny
	    H[1:N,1+N:2N] = matΔ
	    H[1+N:2N,1:N] = matΔ'
	end
	
	@assert typestable(update_H_sc!,(AbstractMatrix{ComplexF64},Matrix{ComplexF64}))
end

# ╔═╡ 75ec114a-9447-4735-9b9b-c734320bb6ae
begin
	function calc_ωn(T,ωc)
	    M = Int((round(ωc/(T*π)))/2-1)
	    println("num. of Matsubara freq: ",2M)
	    ωn = zeros(ComplexF64,2M)
	    for n=1:2M
	        ωn[n] = π*T*(2.0*(n-M-1)+1)*im
	    end
	    return ωn
	end
	@assert typestable(calc_ωn,(Float64,Float64))
end

# ╔═╡ 66703f90-87cc-4982-831c-e365456bba2e
begin
	ωc = 100π
	T = 0.01
	ωn =  calc_ωn(T,ωc)
end

# ╔═╡ 7a79af91-abc7-4fdc-825f-32acf4fa8aa4
begin
	function calc_Δi!(i,N,H,Δold,T,U,ωn;mixratio = 0.5)
	    j = i + N
	    Gij = greensfunctions(i,j,ωn,H)
	    Δi = U*T*sum(Gij)
	    Δi = (1-mixratio)*Δold[i] + mixratio*Δi
	    return Δi
	end
	
	@assert typestable(calc_Δi!,(Int64,Int64,AbstractMatrix{ComplexF64},Matrix{ComplexF64},Float64,Float64,Vector{ComplexF64}))
	
	function calc_Δ!(Δnew,H,Δold,T,U,ωn;mixratio = 0.5)
	    Nx,Ny = size(Δold)
	    N = Nx*Ny
	    map!(i -> calc_Δi!(i,N,H,Δold,T,U,ωn,mixratio = mixratio),Δnew,1:N) #If you use pmap! instead of map!, you can do the parallel computation.
	    return
	end
	
	@assert typestable(calc_Δi!,(AbstractMatrix{ComplexF64},Matrix{ComplexF64},Float64,Float64,Vector{ComplexF64}))
end

# ╔═╡ 50d8bf09-9c5b-4e77-b2c2-4dde7395e399
begin
	Nx = 8
	Ny = 8
	Δ = ones(ComplexF64,Nx,Ny)
	Δold = copy(Δ)
	Δnew = zero(Δ)
	BC = "OBC" #open boundary condition
	#BC = "PBC" #periodic boundary condition
	U  =-2
	μ = -0.2
	
	H = make_H_sc(Nx,Ny,μ,Δ,BC)
end

# ╔═╡ b82db8c0-e2bf-4260-bc57-dea9f7ad3114
begin
	itemax = 100
	ix = Nx ÷ 2
	iy = Ny ÷ 2
	for ite = 1:itemax
	    calc_Δ!(Δnew,H,Δold,T,U,ωn)
	    update_H_sc!(H,Δnew)
	    eps = sum(abs.(Δnew-Δold))/sum(abs.(Δold))
	    println("$ite $eps ",Δnew[ix,iy])
	    Δold .= Δnew
	    if eps < 1e-3
	        break
	    end
	end
end

# ╔═╡ 5e12c321-c89f-434e-8b2a-19934d0cd23f
begin
	wmax = 10.0
	beta = 1/T
	basis = FiniteTempBasis(Fermionic(), beta, wmax, 1e-7)
	smpl = MatsubaraSampling(basis)
	ωn_s = valueim.(smpl.sampling_points, beta)
	println("num. of Matsubara freqs. ", length(ωn_s))
	smpl_beta = TauSampling(basis; sampling_points=[beta])
end

# ╔═╡ 7e57d2a3-2dda-44f2-b8d8-766367fcb122
begin
	function fit_ir(Gij,smpl_Matsubara,smpl_beta)
	    gl = fit(smpl_Matsubara, Gij)
	    G0 = evaluate(smpl_beta, gl)
	    return -G0[1]
	end
	
	@assert typestable(fit_ir,(Vector{ComplexF64},typeof(smpl),typeof(smpl_beta)))
	
	function calc_Δi_ir!(i,N,H,Δold,T,U,ωn,smpl_Matsubara,smpl_beta;mixratio = 0.5)
	    j = i + N
	    Gij = greensfunctions(i,j,ωn,H)
	    G0 = fit_ir(Gij,smpl_Matsubara,smpl_beta)
	    Δi = U*G0
	    Δi = (1-mixratio)*Δold[i] + mixratio*Δi
	    return Δi
	end
	
	@assert typestable(calc_Δi_ir!,(Int64,Int64,AbstractMatrix{ComplexF64},Matrix{ComplexF64},Float64,
	            Float64,Vector{ComplexF64},typeof(smpl),typeof(smpl_beta)))
	
	
	function calc_Δ_ir!(Δnew,H,Δold,T,U,ωn,smpl_Matsubara,smpl_beta;mixratio = 0.5)
	    Nx,Ny = size(Δold)
	    N = Nx*Ny
	    map!(i -> calc_Δi_ir!(i,N,H,Δold,T,U,ωn,smpl_Matsubara,smpl_beta,mixratio = mixratio),Δnew,1:N) #If you use pmap! instead of map!, you can do the parallel computation.
	    return
	end
	
	@assert typestable(calc_Δ_ir!,(Matrix{ComplexF64},AbstractMatrix{ComplexF64},Matrix{ComplexF64},Float64,
	            Float64,Vector{ComplexF64},typeof(smpl),typeof(smpl_beta)))
end

# ╔═╡ 224d09f8-f5aa-478e-b6c5-72e0772a668c
let
	Nx = 8
	Ny = 8
	Δ = ones(ComplexF64,Nx,Ny)
	Δold = copy(Δ)
	Δnew = zero(Δ)
	BC = "OBC" #open boundary condition
	#BC = "PBC" #periodic boundary condition
	U  =-2
	μ = -0.2
	
	H = make_H_sc(Nx,Ny,μ,Δ,BC)
	
	for ite = 1:itemax
	    calc_Δ_ir!(Δnew,H,Δold,T,U,ωn_s,smpl,smpl_beta)
	    update_H_sc!(H,Δnew)
	
	    eps = sum(abs.(Δnew-Δold))/sum(abs.(Δold))
	    println("$ite $eps ",Δnew[ix,iy])
	    Δold .= Δnew
	    if eps < 1e-4
	        break
	    end
	end
end

# ╔═╡ c8dd593b-7317-4a50-8a9f-1cbf4ac779ef
plot(1:Nx,1:Ny,abs.(Δnew),st=:contourf,
    xlabel = "ix",
    ylabel = "iy",
    zlabel = "|Delta|")

# ╔═╡ bd988b30-c3c3-4311-865a-3f26b66232be
begin
	M = 1000
	σ = zeros(ComplexF64,M)
	η = 0.05
	σmin = -4.0 + im*η
	σmax = 4.0+ im*η
	for i=1:M
	    σ[i] = (i-1)*(σmax-σmin)/(M-1) + σmin
	end
	
	i = (iy-1)*Nx + ix
	j = i
	
	Gij1 = greensfunctions(i,j,σ,H) 
	plot(real.(σ),(-1/π)*imag.(Gij1),
	    xlabel = "Energy [t]",
	    ylabel = "Local DOS",)
end

# ╔═╡ ad4cd793-02fb-4f80-bcd9-9ff3d4eb491e
begin
	const σ0 = [1 0
	0 1]
	const σx = [0 1
	1 0]
	const σy = [0 -im
	im 0]
	const σz = [1 0
	0 -1]
end

# ╔═╡ 72af392d-9d89-4085-901f-4c97083480d5
begin
	function make_Htsc_normal(Nx,Ny,μ,BC,h,α)
	    N = Nx*Ny
	    Tx_plushop = make_x_plushop(Nx,Ny,BC)
	    Tx_minushop = make_x_minushop(Nx,Ny,BC)
	    Ty_plushop = make_y_plushop(Nx,Ny,BC)
	    Ty_minushop = make_y_minushop(Nx,Ny,BC)
	    HN = kron(sparse(I,N,N)*(-μ),σ0) 
	    HN += kron(sparse(I,N,N)*(-h),σz) #Zeeman magnetic field
	    t = 1.0
	    
	    HN += kron(-t*(Tx_plushop + Tx_minushop + Ty_plushop + Ty_minushop),σ0)
	    
	    Hax = kron((α/(2im))*(Tx_plushop - Tx_minushop ) ,σy)
	    HN += Hax 
	    Hay = kron((α/(2im))*(Ty_plushop - Ty_minushop ) ,σx)
	    HN += Hay 
	    
	    return HN
	end
	
	@assert typestable(make_Htsc_normal,(Int64,Int64,Float64,String,Float64,Float64))
end

# ╔═╡ d255afba-da17-4272-9206-23dd982ba8b7
begin
	function make_Δtsc(Δ)
	    Nx,Ny = size(Δ)
	    N = Nx*Ny
	    Δmat = spzeros(ComplexF64,N,N)
	    for ix=1:Nx
	        for iy=1:Ny
	            i = (iy-1)*Nx + ix
	            Δmat[i,i] = Δ[ix,iy]
	        end
	    end
	    return kron(Δmat,im*σy)
	end
	
	@assert typestable(make_Δtsc,(Matrix{ComplexF64},))
end

# ╔═╡ 65030056-50bd-4ea9-8fc0-9a3f2f14b5d2
begin
	function make_Htsc_sc(Nx,Ny,μ,Δ,BC,h,α)
	    HN = make_Htsc_normal(Nx,Ny,μ,BC,h,α)
	    matΔ = make_Δtsc(Δ)
	    H = [
	        HN matΔ
	        matΔ' -conj.(HN)
	    ]
	    return H
	end
	
	@assert typestable( make_Htsc_sc,(Int64,Int64,Float64,Matrix{ComplexF64},String,Float64,Float64))
end

# ╔═╡ 5dc554b2-bf57-4548-81bd-9ecd066ef405
begin
	function update_Htsc_sc!(H,Δ)
	    matΔ = make_Δtsc(Δ)
	    N,_ = size(matΔ)
	    H[1:N,1+N:2N] = matΔ
	    H[1+N:2N,1:N] = matΔ'
	end
	
	@assert typestable(update_Htsc_sc!,(AbstractMatrix{ComplexF64},Matrix{ComplexF64}))
end

# ╔═╡ 27a4799b-d2ea-41bc-be35-f49ec6180a5c
begin
	function calc_Δitsc_ir!(i,N,H,Δold,T,U,ωn,smpl_Matsubara,smpl_beta;mixratio = 0.5)
	    ispin = 1
	    ii = (i-1)*2 + ispin
	    jspin = 2
	    jj = (i-1)*2 + jspin + N
	    
	    Gij = greensfunctions(ii,jj,ωn,H) 
	    G0 = fit_ir(Gij,smpl_Matsubara,smpl_beta)            
	    Δi = U*G0
	    Δi = (1-mixratio)*Δold[i] + mixratio*Δi   
	    return Δi
	end
	
	@assert typestable(calc_Δitsc_ir!,(Int64,Int64,AbstractMatrix{ComplexF64},Matrix{ComplexF64},Float64,
	            Float64,Vector{ComplexF64},typeof(smpl),typeof(smpl_beta)))
	
	
	function calc_Δtsc_ir!(Δnew,H,Δold,T,U,ωn,smpl_Matsubara,smpl_beta,;mixratio = 0.5)
	    Nx,Ny = size(Δold)
	    N = Nx*Ny*2
	    map!(i -> calc_Δitsc_ir!(i,N,H,Δold,T,U,ωn,smpl_Matsubara,smpl_beta,mixratio = mixratio),Δnew,1:Nx*Ny) #If you use pmap! instead of map!, you can do the parallel computation.
	end
	
	@assert typestable(calc_Δtsc_ir!,(Matrix{ComplexF64},AbstractMatrix{ComplexF64},Matrix{ComplexF64},Float64,
	            Float64,Vector{ComplexF64},typeof(smpl),typeof(smpl_beta)))
end

# ╔═╡ b88cdd8c-6a91-4d7b-ba03-5aa8d4cef8b5
Htsc = let
	T = 0.01
	
	beta = 1/T
	wmax = 10.0
	
	basis = FiniteTempBasis(Fermionic(), beta, wmax, 1e-5)
	smpl = MatsubaraSampling(basis)
	ωn = valueim.(smpl.sampling_points, beta)
	println("num. of Matsubara freqs. ", length(ωn))
	smpl_beta = TauSampling(basis; sampling_points=[beta])
	
	U  =-5.6
	itemax = 1000
	μ = 3.5
	
	Nx = 16
	Ny = 16
	Δ = 3*ones(ComplexF64,Nx,Ny)
	Δold = copy(Δ)
	Δnew = zero(Δ)
	BC = "OBC"
	h = 1
	α = 1
	Htsc =  make_Htsc_sc(Nx,Ny,μ,Δold,BC,h,α)
	
	ix = Nx ÷ 2
	iy = Ny ÷ 2
	i = (iy-1)*Nx + ix
	j = i
	
	
	for ite = 1:itemax
	    calc_Δtsc_ir!(Δnew,Htsc,Δold,T,U,ωn,smpl,smpl_beta,mixratio =1)
	    update_Htsc_sc!(Htsc,Δnew)
	
	    eps = sum(abs.(Δnew-Δold))/sum(abs.(Δold))
	    println("$ite $eps ",Δnew[ix,iy]," ave: ",sum(abs.(Δnew))/length(Δnew))
	
	    Δold .= Δnew
	    if eps < 1e-2
	        break
	    end
	end
end

# ╔═╡ bdfd4b46-a601-401f-8eba-799268d173b6
plot(1:Nx,1:Ny,abs.(Δnew),st=:contourf,
    xlabel = "ix",
    ylabel = "iy",
    zlabel = "|Delta|")

# ╔═╡ 63367890-a6c8-4552-9f89-1877bf841b7e
let
	M = 1000
	σ = zeros(ComplexF64,M)
	η = 0.01
	σmin = -4.0 + im*η
	σmax = 4.0+ im*η
	for i=1:M
	    σ[i] = (i-1)*(σmax-σmin)/(M-1) + σmin
	end
	
	
	ix = Nx ÷ 2
	iy = Ny ÷ 2
	ispin  =1
	
	i = (( iy-1)*Nx + ix-1)*2 + ispin
	j = i
	Gij1 = greensfunctions(i,j,σ,Htsc) 
	
	ispin  =2
	i = (( iy-1)*Nx + ix-1)*2 + ispin
	j = i
	Gij2 = greensfunctions(i,j,σ,Htsc) 
	plot(real.(σ),(-1/π)*imag.(Gij1 .+ Gij2),
	    xlabel = "Energy [t]",
	    ylabel = "Local DOS",)
end

# ╔═╡ c782238e-6a89-4b18-8cea-961ca7b4d910
let
	ix = 1
	iy = Ny ÷ 2
	ispin  =1
	
	i = (( iy-1)*Nx + ix-1)*2 + ispin
	j = i
	Gij1 = greensfunctions(i,j,σ,Htsc) 
	
	ispin  =2
	i = (( iy-1)*Nx + ix-1)*2 + ispin
	j = i
	Gij2 = greensfunctions(i,j,σ,Htsc) 
	plot(real.(σ),(-1/π)*imag.(Gij1 .+ Gij2),
	    xlabel = "Energy [t]",
	    ylabel = "Local DOS",)
end

# ╔═╡ Cell order:
# ╠═66ed6d9a-4e36-11f0-1f7c-59baebea7d28
# ╠═c895a51b-2597-4dbd-98a4-97d619f81700
# ╠═1786efb3-9df1-48ff-bf89-024bc3ca0ec7
# ╠═8ff8d624-3446-43cc-addd-865c69f2c43d
# ╠═fac94fce-082b-40c2-99ef-6c7057ac89a6
# ╠═75ec114a-9447-4735-9b9b-c734320bb6ae
# ╠═66703f90-87cc-4982-831c-e365456bba2e
# ╠═7a79af91-abc7-4fdc-825f-32acf4fa8aa4
# ╠═50d8bf09-9c5b-4e77-b2c2-4dde7395e399
# ╠═b82db8c0-e2bf-4260-bc57-dea9f7ad3114
# ╠═5e12c321-c89f-434e-8b2a-19934d0cd23f
# ╠═7e57d2a3-2dda-44f2-b8d8-766367fcb122
# ╠═224d09f8-f5aa-478e-b6c5-72e0772a668c
# ╠═c8dd593b-7317-4a50-8a9f-1cbf4ac779ef
# ╠═bd988b30-c3c3-4311-865a-3f26b66232be
# ╠═ad4cd793-02fb-4f80-bcd9-9ff3d4eb491e
# ╠═72af392d-9d89-4085-901f-4c97083480d5
# ╠═d255afba-da17-4272-9206-23dd982ba8b7
# ╠═65030056-50bd-4ea9-8fc0-9a3f2f14b5d2
# ╠═5dc554b2-bf57-4548-81bd-9ecd066ef405
# ╠═27a4799b-d2ea-41bc-be35-f49ec6180a5c
# ╠═b88cdd8c-6a91-4d7b-ba03-5aa8d4cef8b5
# ╠═bdfd4b46-a601-401f-8eba-799268d173b6
# ╠═63367890-a6c8-4552-9f89-1877bf841b7e
# ╠═c782238e-6a89-4b18-8cea-961ca7b4d910
