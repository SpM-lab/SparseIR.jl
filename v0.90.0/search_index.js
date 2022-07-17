var documenterSearchIndex = {"docs":
[{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"\\newcommand{\\var}{\\mathrm{Var}}","category":"page"},{"location":"guide/#Example-usage-and-detailed-explanation","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"","category":"section"},{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"We will explain the inner workings of SparseIR.jl by means of an example use case, adapted from the sparse-ir paper. <!– TODO: link to paper once it's released –>","category":"page"},{"location":"guide/#Problem-statement","page":"Example usage and detailed explanation","title":"Problem statement","text":"","category":"section"},{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"Let us perform self-consistent second-order perturbation theory for the single impurity Anderson model at finite temperature. Its Hamiltonian is given by $     H = U c^\\dagger\\uparrow c^\\dagger\\downarrow c\\downarrow c\\uparrow + \\sum{p\\sigma} \\big(V{p\\sigma}  f{p\\sigma}^\\dagger c\\sigma + V{p\\sigma}^* c\\sigma^\\dagger c\\sigma^\\dagger\\big) + \\sum{p\\sigma} \\epsilon{p} f{p\\sigma}^\\dagger f{p\\sigma} $ where U is the electron interaction strength, c\\sigma$ annihilates an electron on the impurity, f_psigma annihilates an electron in the bath, dagger denotes the Hermitian conjugate, pinmathbb R is bath momentum, and sigmainuparrow downarrow is spin. The hybridization strength V_psigma and bath energies epsilon_p are chosen such that the non-interacting density of states is semi-elliptic with a half-bandwidth of one, rho_0(omega) = frac2pisqrt1-omega^2, U=12, beta=10, and the system is assumed to be half-filled.","category":"page"},{"location":"guide/#Treatment","page":"Example usage and detailed explanation","title":"Treatment","text":"","category":"section"},{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"We first import SparseIR and construct an appropriate basis (omega_mathrmmax = 8 should be more than enough for this example):","category":"page"},{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"julia> using SparseIR\n\njulia> basis = FiniteTempBasis(fermion, 10, 8)\nFiniteTempBasis{LogisticKernel, Float64}(fermion, 10.0, 8.0)","category":"page"},{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"There's quite a lot happening behind the scenes in this first innocuous-looking statement, so let's break it down: Because we did not specify otherwise, the constructor chose the analytic continuation kernel for fermions, LogisticKernel(Λ=80.0), defined by $     K(x, y) = \\frac{e^{-Λ y (x + 1) / 2}}{1 + e^{-Λ y}}, $ for us.","category":"page"},{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"Central is the singular value expansion's (SVE) computation, which is handled by the function compute_sve:","category":"page"},{"location":"guide/","page":"Example usage and detailed explanation","title":"Example usage and detailed explanation","text":"It first constructs ","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SparseIR","category":"page"},{"location":"#SparseIR","page":"Home","title":"SparseIR","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SparseIR.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SparseIR]\nPrivate = false","category":"page"},{"location":"#SparseIR.SparseIR","page":"Home","title":"SparseIR.SparseIR","text":"Intermediate representation (IR) for many-body propagators\n\n\n\n\n\n","category":"module"},{"location":"#SparseIR.CompositeBasisFunction","page":"Home","title":"SparseIR.CompositeBasisFunction","text":"CompositeBasisFunction\n\nUnion of several basis functions in the imaginary-time/real-frequency domain domains\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.CompositeBasisFunction-Tuple{Real}","page":"Home","title":"SparseIR.CompositeBasisFunction","text":"(::CompositeBasisFunction)(x::Real)\n\nEvaluate basis function at position x\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.CompositeBasisFunctionFT","page":"Home","title":"SparseIR.CompositeBasisFunctionFT","text":"CompositeBasisFunctionFT\n\nUnion of several basis functions in the imaginary-frequency domain\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.CompositeBasisFunctionFT-Tuple{Union{Int64, Vector{Int64}}}","page":"Home","title":"SparseIR.CompositeBasisFunctionFT","text":"Evaluate basis function at frequency n\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.DimensionlessBasis","page":"Home","title":"SparseIR.DimensionlessBasis","text":"DimensionlessBasis(statistics, Λ, ε=nothing; kernel=LogisticKernel(Λ), sve_result=compute_sve(kernel; ε))\n\nConstruct an IR basis suitable for the given statistics and cutoff Λ.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.DimensionlessBasis-2","page":"Home","title":"SparseIR.DimensionlessBasis","text":"DimensionlessBasis <: AbstractBasis\n\nIntermediate representation (IR) basis in reduced variables.\n\nFor a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to imaginary time, τ ∈ [0, β], this type stores the truncated singular value expansion or IR basis:\n\nK(x, y) ≈ sum(u[l](x) * s[l] * v[l](y) for l in range(L))\n\nThe functions are given in reduced variables, x = 2τ/β - 1 and y = ω/ωmax, which scales both sides to the interval [-1, 1].  The kernel then only depends on a cutoff parameter Λ = β * ωmax.\n\nExamples\n\nThe following example code assumes the spectral function is a single pole at x = 0.2. We first compute an IR basis suitable for fermions and β*W ≤ 42. Then we get G(iw) on the first few Matsubara frequencies:\n\njulia> using SparseIR\n\njulia> basis = DimensionlessBasis(fermion, 42);\n\njulia> gl = basis.s .* basis.v(0.2);\n\njulia> giw = transpose(basis.uhat([1, 3, 5, 7])) * gl\n\nFields\n\nu::PiecewiseLegendrePolyVector: Set of IR basis functions on the reduced imaginary time (x) axis. These functions are stored as piecewise Legendre polynomials.\nTo obtain the value of all basis functions at a point or a array of points x, you can call the function u(x).  To obtain a single basis function, a slice or a subset l, you can use u[l].\nuhat::PiecewiseLegendreFTVector: Set of IR basis functions on the Matsubara frequency (wn) axis.\n\nThese objects are stored as a set of Bessel functions.\n\nTo obtain the value of all basis functions at a Matsubara frequency   or a array of points wn, you can call the function uhat(wn).   Note that we expect reduced frequencies, which are simply even/odd   numbers for bosonic/fermionic objects. To obtain a single basis   function, a slice or a subset l, you can use uhat[l].\n\ns: Vector of singular values of the continuation kernel\nv::PiecewiseLegendrePolyVector: Set of IR basis functions on the reduced real frequency (y) axis.\n\nThese functions are stored as piecewise Legendre polynomials.\n\nTo obtain the value of all basis functions at a point or a array of   points y, you can call the function v(y).  To obtain a single   basis function, a slice or a subset l, you can use v[l].\n\nSee also FiniteTempBasis for a basis directly in time/frequency.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasis","page":"Home","title":"SparseIR.FiniteTempBasis","text":"FiniteTempBasis(statistics, β, wmax, ε=nothing; kernel=LogisticKernel(β * wmax), sve_result=compute_sve(kernel; ε))\n\nConstruct a finite temperature basis suitable for the given statistics and cutoffs β and wmax.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasis-2","page":"Home","title":"SparseIR.FiniteTempBasis","text":"FiniteTempBasis <: AbstractBasis\n\nIntermediate representation (IR) basis for given temperature.\n\nFor a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to imaginary time, τ ∈ [0, beta], this type stores the truncated singular value expansion or IR basis:\n\nK(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 1:L)\n\nThis basis is inferred from a reduced form by appropriate scaling of the variables.\n\nExamples\n\nThe following example code assumes the spectral function is a single pole at ω = 2.5. We first compute an IR basis suitable for fermions and β = 10, W ≤ 4.2. Then we get G(iw) on the first few Matsubara frequencies:\n\njulia> using SparseIR\n\njulia> basis = FiniteTempBasis(fermion, 42, 4.2);\n\njulia> gl = basis.s .* basis.v(2.5);\n\njulia> giw = transpose(basis.uhat([1, 3, 5, 7])) * gl\n\nFields\n\nu::PiecewiseLegendrePolyVector: Set of IR basis functions on the imaginary time (tau) axis. These functions are stored as piecewise Legendre polynomials.\nTo obtain the value of all basis functions at a point or a array of points x, you can call the function u(x).  To obtain a single basis function, a slice or a subset l, you can use u[l].\nuhat::PiecewiseLegendreFT: Set of IR basis functions on the Matsubara frequency (wn) axis. These objects are stored as a set of Bessel functions.\nTo obtain the value of all basis functions at a Matsubara frequency or a array of points wn, you can call the function uhat(wn). Note that we expect reduced frequencies, which are simply even/odd numbers for bosonic/fermionic objects. To obtain a single basis function, a slice or a subset l, you can use uhat[l].\ns: Vector of singular values of the continuation kernel\nv::PiecewiseLegendrePoly: Set of IR basis functions on the real frequency (w) axis. These functions are stored as piecewise Legendre polynomials.\nTo obtain the value of all basis functions at a point or a array of points w, you can call the function v(w).  To obtain a single basis function, a slice or a subset l, you can use v[l].\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasisSet","page":"Home","title":"SparseIR.FiniteTempBasisSet","text":"FiniteTempBasisSet\n\nType for holding IR bases and sparse-sampling objects.\n\nAn object of this type holds IR bases for fermions and bosons and associated sparse-sampling objects.\n\nFields\n\nbasis_f::FiniteTempBasis: Fermion basis\nbasis_b::FiniteTempBasis: Boson basis\nbeta::Float64: Inverse temperature\nwmax::Float64: Cut-off frequency\ntau::Vector{Float64}: Sampling points in the imaginary-time domain\nwn_f::Vector{Int}: Sampling fermionic frequencies\nwn_b::Vector{Int}: Sampling bosonic frequencies\nsmpltauf::TauSampling: Sparse sampling for tau & fermion\nsmpltaub::TauSampling: Sparse sampling for tau & boson\nsmplwnf::MatsubaraSampling: Sparse sampling for Matsubara frequency & fermion\nsmplwnb::MatsubaraSampling: Sparse sampling for Matsubara frequency & boson\nsve_result::Tuple{PiecewiseLegendrePoly,Vector{Float64},PiecewiseLegendrePoly}: Results of SVE\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasisSet-Tuple{AbstractFloat, AbstractFloat, Any}","page":"Home","title":"SparseIR.FiniteTempBasisSet","text":"FiniteTempBasisSet(β, wmax, ε; sve_result=compute_sve(LogisticKernel(β * wmax); ε))\n\nCreate basis sets for fermion and boson and associated sampling objects. Fermion and bosonic bases are constructed by SVE of the logistic kernel.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.LegendreBasis","page":"Home","title":"SparseIR.LegendreBasis","text":"Legendre basis\n\nIn the original paper [L. Boehnke et al., PRB 84, 075145 (2011)], they used:\n\nG(\\tau) = \\sum_{l=0} \\sqrt{2l+1} P_l[x(\\tau)] G_l/beta,\n\nwhere P_l[x] is the l-th Legendre polynomial.\n\nIn this type, the basis functions are defined by\n\nU_l(\\tau) \\equiv c_l (\\sqrt{2l+1}/beta) * P_l[x(\\tau)],\n\nwhere cl are additional l-dependent constant factors. By default, we take cl = 1, which reduces to the original definition.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.LogisticKernel","page":"Home","title":"SparseIR.LogisticKernel","text":"LogisticKernel <: AbstractKernel\n\nFermionic/bosonic analytical continuation kernel.\n\nIn dimensionless variables x = 2 τβ - 1, y = β ωΛ, the integral kernel is a function on -1 1  -1 1:\n\n    K(x y) = frace^-Λ y (x + 1)  21 + e^-Λ y\n\nLogisticKernel is a fermionic analytic continuation kernel. Nevertheless, one can model the τ dependence of a bosonic correlation function as follows:\n\n     frace^-Λ y (x + 1)  21 - e^-Λ y ρ(y) dy =  K(x y) ρ(y) dy\n\nwith\n\n    ρ(y) = w(y) ρ(y)\n\nwhere the weight function is given by\n\n    w(y) = frac1tanh(Λ y2)\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.MatsubaraConstBasis","page":"Home","title":"SparseIR.MatsubaraConstBasis","text":"Constant term in matsubara-frequency domain\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.MatsubaraSampling","page":"Home","title":"SparseIR.MatsubaraSampling","text":"MatsubaraSampling(basis[, sampling_points])\n\nConstruct a MatsubaraSampling object. If not given, the sampling_points are chosen as  the (discrete) extrema of the highest-order basis function in Matsubara. This turns out  to be close to optimal with respect to conditioning for this size (within a few percent).\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.MatsubaraSampling-2","page":"Home","title":"SparseIR.MatsubaraSampling","text":"MatsubaraSampling <: AbstractSampling\n\nSparse sampling in Matsubara frequencies.\n\nAllows the transformation between the IR basis and a set of sampling points in (scaled/unscaled) imaginary frequencies.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.RegularizedBoseKernel","page":"Home","title":"SparseIR.RegularizedBoseKernel","text":"RegularizedBoseKernel <: AbstractKernel\n\nRegularized bosonic analytical continuation kernel.\n\nIn dimensionless variables x = 2 τβ - 1, y = β ωΛ, the fermionic integral kernel is a function on -1 1  -1 1:\n\n    K(x y) = y frace^-Λ y (x + 1)  2e^-Λ y - 1\n\nCare has to be taken in evaluating this expression around y = 0.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.SparsePoleRepresentation","page":"Home","title":"SparseIR.SparsePoleRepresentation","text":"Sparse pole representation\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.TauSampling","page":"Home","title":"SparseIR.TauSampling","text":"TauSampling(basis[, sampling_points])\n\nConstruct a TauSampling object. If not given, the sampling_points are chosen as  the extrema of the highest-order basis function in imaginary time. This turns out  to be close to optimal with respect to conditioning for this size (within a few percent).\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.TauSampling-2","page":"Home","title":"SparseIR.TauSampling","text":"TauSampling <: AbstractSampling\n\nSparse sampling in imaginary time.\n\nAllows the transformation between the IR basis and a set of sampling points in (scaled/unscaled) imaginary time.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.evaluate!-Tuple{AbstractArray, SparseIR.AbstractSampling, Any}","page":"Home","title":"SparseIR.evaluate!","text":"evaluate!(buffer::AbstractArray{T,N}, sampling, al; dim=1) where {T,N}\n\nLike evaluate, but write the result to buffer. Please use dim = 1 or N to avoid allocating large temporary arrays internally.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.evaluate-Union{Tuple{N}, Tuple{T}, Tuple{Tmat}, Tuple{S}, Tuple{SparseIR.AbstractSampling{S, Tmat}, AbstractArray{T, N}}} where {S, Tmat, T, N}","page":"Home","title":"SparseIR.evaluate","text":"evaluate(sampling, al; dim=1)\n\nEvaluate the basis coefficients al at the sparse sampling points.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.fit!-Union{Tuple{N}, Tuple{T}, Tuple{S}, Tuple{Array{S, N}, SparseIR.AbstractSampling, Array{T, N}}} where {S, T, N}","page":"Home","title":"SparseIR.fit!","text":"fit!(buffer, sampling, al::Array{T,N}; dim=1)\n\nLike fit, but write the result to buffer. Please use dim = 1 or N to avoid allocating large temporary arrays internally. The length of workarry cannot be smaller than the returned value of workarrlengthfit.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.fit-Union{Tuple{N}, Tuple{T}, Tuple{Tmat}, Tuple{S}, Tuple{SparseIR.AbstractSampling{S, Tmat}, AbstractArray{T, N}}} where {S, Tmat, T, N}","page":"Home","title":"SparseIR.fit","text":"fit(sampling, al::AbstractArray{T,N}; dim=1)\n\nFit basis coefficients from the sparse sampling points Please use dim = 1 or N to avoid allocating large temporary arrays internally.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.from_IR","page":"Home","title":"SparseIR.from_IR","text":"From IR to SPR\n\ngl:     Expansion coefficients in IR\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.overlap-Union{Tuple{T}, Tuple{SparseIR.PiecewiseLegendrePoly{T}, Any}} where T","page":"Home","title":"SparseIR.overlap","text":"overlap(poly::PiecewiseLegendrePoly, f)\n\nEvaluate overlap integral of poly with arbitrary function f.\n\nGiven the function f, evaluate the integral::\n\n∫ dx * f(x) * poly(x)\n\nusing adaptive Gauss-Legendre quadrature.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.to_IR","page":"Home","title":"SparseIR.to_IR","text":"From SPR to IR\n\ng_spr:     Expansion coefficients in SPR\n\n\n\n\n\n","category":"function"}]
}