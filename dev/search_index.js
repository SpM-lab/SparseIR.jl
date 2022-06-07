var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SparseIR","category":"page"},{"location":"#SparseIR","page":"Home","title":"SparseIR","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SparseIR.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SparseIR]","category":"page"},{"location":"#SparseIR.SparseIR","page":"Home","title":"SparseIR.SparseIR","text":"Intermediate representation (IR) for many-body propagators\n\n\n\n\n\n","category":"module"},{"location":"#Core.Union-Tuple{Integer}","page":"Home","title":"Core.Union","text":"(polyFT::PiecewiseLegendreFT)(n)\n\nObtain Fourier transform of polynomial for given frequency index n.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.AbstractKernel","page":"Home","title":"SparseIR.AbstractKernel","text":"(kernel::AbstractKernel)(x, y[, x₊, x₋])\n\nEvaluate kernel at point (x, y).\n\nThe parameters x₊ and x₋, if given, shall contain the values of x - xₘᵢₙ and xₘₐₓ - x, respectively.  This is useful if either difference is to be formed and cancellation expected.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.AbstractKernel-2","page":"Home","title":"SparseIR.AbstractKernel","text":"AbstractKernel\n\nIntegral kernel K(x, y).\n\nAbstract base type for an integral kernel, i.e. a AbstractFloat binary function K(x y) used in a Fredhold integral equation of the first kind:\n\n    u(x) =  K(x y) v(y) dy\n\nwhere x  x_mathrmmin x_mathrmmax and  y  y_mathrmmin y_mathrmmax.  For its SVE to exist, the kernel must be square-integrable, for its singular values to decay exponentially, it must be smooth.\n\nIn general, the kernel is applied to a scaled spectral function ρ(y) as:\n\n     K(x y) ρ(y) dy\n\nwhere ρ(y) = w(y) ρ(y).\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.AbstractSVEHints","page":"Home","title":"SparseIR.AbstractSVEHints","text":"AbstractSVEHints\n\nDiscretization hints for singular value expansion of a given kernel.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.AbstractSampling","page":"Home","title":"SparseIR.AbstractSampling","text":"AbstractSampling\n\nAbstract class for sparse sampling.\n\nEncodes the \"basis transformation\" of a propagator from the truncated IR basis coefficients G_ir[l] to time/frequency sampled on sparse points G(x[i]) together with its inverse, a least squares fit:\n\n     ________________                   ___________________\n    |                |    evaluate     |                   |\n    |     Basis      |---------------->|     Value on      |\n    |  coefficients  |<----------------|  sampling points  |\n    |________________|      fit        |___________________|\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.CentrosymmSVE","page":"Home","title":"SparseIR.CentrosymmSVE","text":"CentrosymmSVE <: AbstractSVE\n\nSVE of centrosymmetric kernel in block-diagonal (even/odd) basis.\n\nFor a centrosymmetric kernel K, i.e., a kernel satisfying: K(x, y) == K(-x, -y), one can make the following ansatz for the singular functions:\n\nu[l](x) = ured[l](x) + sign[l] * ured[l](-x)\nv[l](y) = vred[l](y) + sign[l] * ured[l](-y)\n\nwhere sign[l] is either +1 or -1.  This means that the singular value expansion can be block-diagonalized into an even and an odd part by (anti-)symmetrizing the kernel:\n\nK_even = K(x, y) + K(x, -y)\nK_odd  = K(x, y) - K(x, -y)\n\nThe lth basis function, restricted to the positive interval, is then the singular function of one of these kernels.  If the kernel generates a Chebyshev system [1], then even and odd basis functions alternate.\n\n[1]: A. Karlin, Total Positivity (1968).\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.CompositeBasisFunction","page":"Home","title":"SparseIR.CompositeBasisFunction","text":"Union of several basis functions in the imaginary-time/real-frequency domain domains\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.CompositeBasisFunction-Tuple{Real}","page":"Home","title":"SparseIR.CompositeBasisFunction","text":"Evaluate basis function at position x\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.CompositeBasisFunctionFT","page":"Home","title":"SparseIR.CompositeBasisFunctionFT","text":"Union of several basis functions in the imaginary-frequency domain domains\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.CompositeBasisFunctionFT-Tuple{Union{Int64, Vector{Int64}}}","page":"Home","title":"SparseIR.CompositeBasisFunctionFT","text":"Evaluate basis function at frequency n\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.DimensionlessBasis","page":"Home","title":"SparseIR.DimensionlessBasis","text":"DimensionlessBasis(statistics, Λ, ε=nothing; kernel=LogisticKernel(Λ), sve_result=compute_sve(kernel; ε))\n\nConstruct an IR basis suitable for the given statistics and cutoff Λ.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.DimensionlessBasis-2","page":"Home","title":"SparseIR.DimensionlessBasis","text":"DimensionlessBasis <: AbstractBasis\n\nIntermediate representation (IR) basis in reduced variables.\n\nFor a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to imaginary time, τ ∈ [0, β], this class stores the truncated singular value expansion or IR basis:\n\nK(x, y) ≈ sum(u[l](x) * s[l] * v[l](y) for l in range(L))\n\nThe functions are given in reduced variables, x = 2τ/β - 1 and y = ω/ωmax, which scales both sides to the interval [-1, 1].  The kernel then only depends on a cutoff parameter Λ = β * ωmax.\n\nExamples\n\nThe following example code assumes the spectral function is a single pole at x = 0.2. We first compute an IR basis suitable for fermions and β*W ≤ 42. Then we get G(iw) on the first few Matsubara frequencies:\n\njulia> using SparseIR\n\njulia> basis = DimensionlessBasis(fermion, 42);\n\njulia> gl = basis.s .* basis.v(0.2);\n\njulia> giw = transpose(basis.uhat([1, 3, 5, 7])) * gl\n\nFields\n\nu::PiecewiseLegendrePolyVector: Set of IR basis functions on the reduced imaginary time (x) axis. These functions are stored as piecewise Legendre polynomials.\nTo obtain the value of all basis functions at a point or a array of points x, you can call the function u(x).  To obtain a single basis function, a slice or a subset l, you can use u[l].\nuhat::PiecewiseLegendreFTVector: Set of IR basis functions on the Matsubara frequency (wn) axis.\n\nThese objects are stored as a set of Bessel functions.\n\nTo obtain the value of all basis functions at a Matsubara frequency   or a array of points wn, you can call the function uhat(wn).   Note that we expect reduced frequencies, which are simply even/odd   numbers for bosonic/fermionic objects. To obtain a single basis   function, a slice or a subset l, you can use uhat[l].\n\ns: Vector of singular values of the continuation kernel\nv::PiecewiseLegendrePolyVector: Set of IR basis functions on the reduced real frequency (y) axis.\n\nThese functions are stored as piecewise Legendre polynomials.\n\nTo obtain the value of all basis functions at a point or a array of   points y, you can call the function v(y).  To obtain a single   basis function, a slice or a subset l, you can use v[l].\n\nSee also FiniteTempBasis for a basis directly in time/frequency.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasis","page":"Home","title":"SparseIR.FiniteTempBasis","text":"FiniteTempBasis(statistics, β, wmax, ε=nothing; kernel=LogisticKernel(β * wmax), sve_result=compute_sve(kernel; ε))\n\nConstruct a finite temperature basis suitable for the given statistics and cutoffs β and wmax.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasis-2","page":"Home","title":"SparseIR.FiniteTempBasis","text":"FiniteTempBasis <: AbstractBasis\n\nIntermediate representation (IR) basis for given temperature.\n\nFor a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to imaginary time, τ ∈ [0, beta], this class stores the truncated singular value expansion or IR basis:\n\nK(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 1:L)\n\nThis basis is inferred from a reduced form by appropriate scaling of the variables.\n\nExamples\n\nThe following example code assumes the spectral function is a single pole at ω = 2.5. We first compute an IR basis suitable for fermions and β = 10, W ≤ 4.2. Then we get G(iw) on the first few Matsubara frequencies:\n\njulia> using SparseIR\n\njulia> basis = FiniteTempBasis(fermion, 42, 4.2);\n\njulia> gl = basis.s .* basis.v(2.5);\n\njulia> giw = transpose(basis.uhat([1, 3, 5, 7])) * gl\n\nFields\n\nu::PiecewiseLegendrePolyVector: Set of IR basis functions on the imaginary time (tau) axis. These functions are stored as piecewise Legendre polynomials.\nTo obtain the value of all basis functions at a point or a array of points x, you can call the function u(x).  To obtain a single basis function, a slice or a subset l, you can use u[l].\nuhat::PiecewiseLegendreFT: Set of IR basis functions on the Matsubara frequency (wn) axis. These objects are stored as a set of Bessel functions.\nTo obtain the value of all basis functions at a Matsubara frequency or a array of points wn, you can call the function uhat(wn). Note that we expect reduced frequencies, which are simply even/odd numbers for bosonic/fermionic objects. To obtain a single basis function, a slice or a subset l, you can use uhat[l].\ns: Vector of singular values of the continuation kernel\nv::PiecewiseLegendrePoly: Set of IR basis functions on the real frequency (w) axis. These functions are stored as piecewise Legendre polynomials.\nTo obtain the value of all basis functions at a point or a array of points w, you can call the function v(w).  To obtain a single basis function, a slice or a subset l, you can use v[l].\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasisSet","page":"Home","title":"SparseIR.FiniteTempBasisSet","text":"FiniteTempBasisSet\n\nClass for holding IR bases and sparse-sampling objects.\n\nAn object of this class holds IR bases for fermions and bosons and associated sparse-sampling objects.\n\nFields\n\nbasis_f::FiniteTempBasis: Fermion basis\nbasis_b::FiniteTempBasis: Boson basis\nbeta::Float64: Inverse temperature\nwmax::Float64: Cut-off frequency\ntau::Vector{Float64}: Sampling points in the imaginary-time domain\nwn_f::Vector{Int}: Sampling fermionic frequencies\nwn_b::Vector{Int}: Sampling bosonic frequencies\nsmpltauf::TauSampling: Sparse sampling for tau & fermion\nsmpltaub::TauSampling: Sparse sampling for tau & boson\nsmplwnf::MatsubaraSampling: Sparse sampling for Matsubara frequency & fermion\nsmplwnb::MatsubaraSampling: Sparse sampling for Matsubara frequency & boson\nsve_result::Tuple{PiecewiseLegendrePoly,Vector{Float64},PiecewiseLegendrePoly}: Results of SVE\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.FiniteTempBasisSet-Tuple{AbstractFloat, AbstractFloat, Any}","page":"Home","title":"SparseIR.FiniteTempBasisSet","text":"FiniteTempBasisSet(β, wmax, ε; sve_result=compute_sve(LogisticKernel(β * wmax); ε))\n\nCreate basis sets for fermion and boson and associated sampling objects. Fermion and bosonic bases are constructed by SVE of the logistic kernel.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.LegendreBasis","page":"Home","title":"SparseIR.LegendreBasis","text":"Legendre basis\n\nIn the original paper [L. Boehnke et al., PRB 84, 075145 (2011)], they used:\n\nG(\\tau) = \\sum_{l=0} \\sqrt{2l+1} P_l[x(\\tau)] G_l/beta,\n\nwhere P_l[x] is the l-th Legendre polynomial.\n\nIn this class, the basis functions are defined by\n\nU_l(\\tau) \\equiv c_l (\\sqrt{2l+1}/beta) * P_l[x(\\tau)],\n\nwhere cl are additional l-depenent constant factors. By default, we take cl = 1, which reduces to the original definition.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.LogisticKernel","page":"Home","title":"SparseIR.LogisticKernel","text":"LogisticKernel <: AbstractKernel\n\nFermionic/bosonic analytical continuation kernel.\n\nIn dimensionless variables x = 2 τβ - 1, y = β ωΛ, the integral kernel is a function on -1 1  -1 1:\n\n    K(x y) = frace^-Λ y (x + 1)  21 + e^-Λ y\n\nLogisticKernel is a fermionic analytic continuation kernel. Nevertheless, one can model the τ dependence of a bosonic correlation function as follows:\n\n     frace^-Λ y (x + 1)  21 - e^-Λ y ρ(y) dy =  K(x y) ρ(y) dy\n\nwith\n\n    ρ(y) = w(y) ρ(y)\n\nwhere the weight function is given by\n\n    w(y) = frac1tanh(Λ y2)\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.LogisticKernelOdd","page":"Home","title":"SparseIR.LogisticKernelOdd","text":"LogisticKernelOdd <: AbstractReducedKernel\n\nFermionic analytical continuation kernel, odd.\n\nIn dimensionless variables x = 2τβ - 1, y = βωΛ, the fermionic integral kernel is a function on -1 1  -1 1:\n\n    K(x y) = -fracsinh(Λ x y  2)cosh(Λ y  2)\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.MatsubaraConstBasis","page":"Home","title":"SparseIR.MatsubaraConstBasis","text":"Constant term in matsubara-frequency domain\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.MatsubaraSampling","page":"Home","title":"SparseIR.MatsubaraSampling","text":"MatsubaraSampling(basis[, sampling_points])\n\nConstruct a MatsubaraSampling object. If not given, the sampling_points are chosen as  the (discrete) extrema of the highest-order basis function in Matsubara. This turns out  to be close to optimal with respect to conditioning for this size (within a few percent).\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.MatsubaraSampling-2","page":"Home","title":"SparseIR.MatsubaraSampling","text":"MatsubaraSampling <: AbstractSampling\n\nSparse sampling in Matsubara frequencies.\n\nAllows the transformation between the IR basis and a set of sampling points in (scaled/unscaled) imaginary frequencies.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.NestedRule","page":"Home","title":"SparseIR.NestedRule","text":"NestedRule{T}\n\nNested quadrature rule.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.PiecewiseLegendreFT","page":"Home","title":"SparseIR.PiecewiseLegendreFT","text":"PiecewiseLegendreFT <: Function\n\nFourier transform of a piecewise Legendre polynomial.\n\nFor a given frequency index n, the Fourier transform of the Legendre function is defined as:\n\n    p̂(n) == ∫ dx exp(im * π * n * x / (xmax - xmin)) p(x)\n\nThe polynomial is continued either periodically (freq=:even), in which case n must be even, or antiperiodically (freq=:odd), in which case n must be odd.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.PiecewiseLegendrePoly","page":"Home","title":"SparseIR.PiecewiseLegendrePoly","text":"PiecewiseLegendrePoly <: Function\n\nPiecewise Legendre polynomial.\n\nModels a function on the interval xmin xmax as a set of segments on the intervals Si = ai ai+1, where on each interval the function is expanded in scaled Legendre polynomials.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.PiecewiseLegendrePolyVector","page":"Home","title":"SparseIR.PiecewiseLegendrePolyVector","text":"PiecewiseLegendrePolyVector{T}\n\nAlias for Vector{PiecewiseLegendrePoly{T}}.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.PowerModel","page":"Home","title":"SparseIR.PowerModel","text":"PowerModel\n\nModel from a high-frequency series expansion::\n\nA(iω) == sum(A[n] / (iω)^(n+1) for n in 1:N)\n\nwhere iω == i * π2 * wn is a reduced imaginary frequency, i.e., wn is an odd/even number for fermionic/bosonic frequencies.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.ReducedKernel","page":"Home","title":"SparseIR.ReducedKernel","text":"ReducedKernel\n\nRestriction of centrosymmetric kernel to positive interval.\n\nFor a kernel K on -1 1  -1 1 that is centrosymmetric, i.e. K(x y) = K(-x -y), it is straight-forward to show that the left/right singular vectors can be chosen as either odd or even functions.\n\nConsequentially, they are singular functions of a reduced kernel K_mathrmred on 0 1  0 1 that is given as either:\n\n    K_mathrmred(x y) = K(x y) pm K(x -y)\n\nThis kernel is what this class represents.  The full singular functions can be reconstructed by (anti-)symmetrically continuing them to the negative axis.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.RegularizedBoseKernel","page":"Home","title":"SparseIR.RegularizedBoseKernel","text":"RegularizedBoseKernel <: AbstractKernel\n\nRegularized bosonic analytical continuation kernel.\n\nIn dimensionless variables x = 2 τβ - 1, y = β ωΛ, the fermionic integral kernel is a function on -1 1  -1 1:\n\n    K(x y) = y frace^-Λ y (x + 1)  2e^-Λ y - 1\n\nCare has to be taken in evaluating this expression around y = 0.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.RegularizedBoseKernelOdd","page":"Home","title":"SparseIR.RegularizedBoseKernelOdd","text":"RegularizedBoseKernelOdd <: AbstractReducedKernel\n\nBosonic analytical continuation kernel, odd.\n\nIn dimensionless variables x = 2 τ  β - 1, y = β ω  Λ, the fermionic integral kernel is a function on -1 1  -1 1:\n\n    K(x y) = -y fracsinh(Λ x y  2)sinh(Λ y  2)\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.Rule","page":"Home","title":"SparseIR.Rule","text":"Rule{T<:AbstractFloat}\n\nQuadrature rule.\n\nApproximation of an integral over [a, b] by a sum over discrete points x with weights w:\n\n     f(x) ω(x) dx  _i f(x_i) w_i\n\nwhere we generally have superexponential convergence for smooth f(x) in  the number of quadrature points.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.SamplingSVE","page":"Home","title":"SparseIR.SamplingSVE","text":"SamplingSVE <: AbstractSVE\n\nSVE to SVD translation by sampling technique [1].\n\nMaps the singular value expansion (SVE) of a kernel kernel onto the singular value decomposition of a matrix A.  This is achieved by choosing two sets of Gauss quadrature rules: (x, wx) and (y, wy) and approximating the integrals in the SVE equations by finite sums.  This implies that the singular values of the SVE are well-approximated by the singular values of the following matrix:\n\nA[i, j] = √(wx[i]) * K(x[i], y[j]) * √(wy[j])\n\nand the values of the singular functions at the Gauss sampling points can be reconstructed from the singular vectors u and v as follows:\n\nu[l,i] ≈ √(wx[i]) u[l](x[i])\nv[l,j] ≈ √(wy[j]) u[l](y[j])\n\n[1] P. Hansen, Discrete Inverse Problems, Ch. 3.1\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.SparsePoleRepresentation","page":"Home","title":"SparseIR.SparsePoleRepresentation","text":"Sparse pole representation\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.TauSampling","page":"Home","title":"SparseIR.TauSampling","text":"TauSampling(basis[, sampling_points])\n\nConstruct a TauSampling object. If not given, the sampling_points are chosen as  the extrema of the highest-order basis function in imaginary time. This turns out  to be close to optimal with respect to conditioning for this size (within a few percent).\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR.TauSampling-2","page":"Home","title":"SparseIR.TauSampling","text":"TauSampling <: AbstractSampling\n\nSparse sampling in imaginary time.\n\nAllows the transformation between the IR basis and a set of sampling points in (scaled/unscaled) imaginary time.\n\n\n\n\n\n","category":"type"},{"location":"#SparseIR._canonicalize!-Tuple{Any, Any}","page":"Home","title":"SparseIR._canonicalize!","text":"canonicalize!(u, v)\n\nCanonicalize basis.\n\nEach SVD (u[l], v[l]) pair is unique only up to a global phase, which may differ from implementation to implementation and also platform. We fix that gauge by demanding u[l](1) > 0. This ensures a diffeomorphic connection to the Legendre polynomials as Λ → 0.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR._choose_accuracy-Tuple{Any, Any}","page":"Home","title":"SparseIR._choose_accuracy","text":"choose_accuracy(ε, Twork)\n\nChoose work type and accuracy based on specs and defaults\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR._compute_unl_inner-Tuple{SparseIR.PiecewiseLegendrePoly, Any}","page":"Home","title":"SparseIR._compute_unl_inner","text":"_compute_unl_inner(poly, wn)\n\nCompute piecewise Legendre to Matsubara transform.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR._get_tnl-Tuple{Any, Any}","page":"Home","title":"SparseIR._get_tnl","text":"_get_tnl(l, w)\n\nFourier integral of the l-th Legendre polynomial::\n\nTₗ(ω) == ∫ dx exp(iωx) Pₗ(x)\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR._phase_stable-Tuple{Any, Any}","page":"Home","title":"SparseIR._phase_stable","text":"_phase_stable(poly, wn)\n\nPhase factor for the piecewise Legendre to Matsubara transform.\n\nCompute the following phase factor in a stable way:\n\nexp.(iπ/2 * wn * cumsum(poly.Δx))\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR._shift_xmid-Tuple{Any, Any}","page":"Home","title":"SparseIR._shift_xmid","text":"_shift_xmid(knots, Δx)\n\nReturn midpoint relative to the nearest integer plus a shift.\n\nReturn the midpoints xmid of the segments, as pair (diff, shift), where shift is in (0, 1, -1) and diff is a float such that xmid == shift + diff to floating point accuracy.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR._split-Tuple{Any, Number}","page":"Home","title":"SparseIR._split","text":"_split(poly, x)\n\nSplit segment.\n\nFind segment of poly's domain that covers x.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.check_domain-Tuple{Any, Any, Any}","page":"Home","title":"SparseIR.check_domain","text":"check_domain(kernel, x, y)\n\nCheck that (x, y) lies within kernel's domain and return it.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.check_reduced_matsubara-Tuple{Integer}","page":"Home","title":"SparseIR.check_reduced_matsubara","text":"check_reduced_matsubara(n[, ζ])\n\nChecks that n is a reduced Matsubara frequency.\n\nCheck that the argument is a reduced Matsubara frequency, which is an integer obtained by scaling the freqency ω[n] as follows:\n\nβ / π * ω[n] == 2n + ζ\n\nNote that this means that instead of a fermionic frequency (ζ == 1), we expect an odd integer, while for a bosonic frequency (ζ == 0), we expect an even one.  If ζ is omitted, any one is fine.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.compute_sve-Tuple{SparseIR.AbstractKernel}","page":"Home","title":"SparseIR.compute_sve","text":"compute_sve(kernel; \n    ε=nothing, n_sv=typemax(Int), n_gauss=nothing, T=Float64, Twork=nothing,\n    sve_strat=iscentrosymmetric(kernel) ? CentrosymmSVE : SamplingSVE,\n    svd_strat=nothing)\n\nPerform truncated singular value expansion of a kernel.\n\nPerform a truncated singular value expansion (SVE) of an integral kernel kernel : [xmin, xmax] x [ymin, ymax] -> ℝ:\n\nkernel(x, y) == sum(s[l] * u[l](x) * v[l](y) for l in (1, 2, 3, ...)),\n\nwhere s[l] are the singular values, which are ordered in non-increasing fashion, u[l](x) are the left singular functions, which form an orthonormal system on [xmin, xmax], and v[l](y) are the right singular functions, which form an orthonormal system on [ymin, ymax].\n\nThe SVE is mapped onto the singular value decomposition (SVD) of a matrix by expanding the kernel in piecewise Legendre polynomials (by default by using a collocation).\n\nArguments\n\neps::AbstractFloat:  Relative cutoff for the singular values.\nn_sv::Integer: Maximum basis size. If given, only at most the n_sv most\n\nsignificant singular values and associated singular functions are returned.\n\nn_gauss::Integer: Order of Legendre polynomials. Defaults to hinted value\n\nby the kernel.\n\nT: Data type of the result.\nTwork: Working data type. Defaults to a data type with\n\nmachine epsilon of at least eps^2, or otherwise most accurate data type available.\n\nsve_strat: SVE to SVD translation strategy. Defaults to SamplingSVE.\nsvd_strat: SVD solver. Defaults to fast (ID/RRQR) based solution \n\nwhen accuracy goals are moderate, and more accurate Jacobi-based  algorithm otherwise.\n\nReturn value\n\nReturn tuple (u, s, v), where:\n\nu::PiecewiseLegendrePoly: the left singular functions\ns::Vector: singular values\nv::PiecewiseLegendrePoly: the right singular functions\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.conv_radius-Tuple{SparseIR.AbstractKernel}","page":"Home","title":"SparseIR.conv_radius","text":"conv_radius(kernel)\n\nConvergence radius of the Matsubara basis asymptotic model.\n\nFor improved relative numerical accuracy, the IR basis functions on the Matsubara axis uhat(basis, n) can be evaluated from an asymptotic expression for abs(n) > conv_radius.  If isinf(conv_radius), then  the asymptotics are unused (the default).\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.default_matsubara_sampling_points-Tuple{SparseIR.AbstractBasis}","page":"Home","title":"SparseIR.default_matsubara_sampling_points","text":"_default_matsubara_sampling_points(basis; mitigate=true)\n\nDefault sampling points on the imaginary frequency axis.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.default_omega_sampling_points-Tuple{SparseIR.AbstractBasis}","page":"Home","title":"SparseIR.default_omega_sampling_points","text":"default_omega_sampling_points(basis)\n\nDefault sampling points on the real-frequency axis.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.default_tau_sampling_points-Tuple{SparseIR.AbstractBasis}","page":"Home","title":"SparseIR.default_tau_sampling_points","text":"default_tau_sampling_points(basis)\n\nDefault sampling points on the imaginary time/x axis.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.deriv-Tuple{SparseIR.PiecewiseLegendrePoly}","page":"Home","title":"SparseIR.deriv","text":"deriv(poly)\n\nGet polynomial for the derivative.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.eval_matrix-Tuple{Type{TauSampling}, Any, Any}","page":"Home","title":"SparseIR.eval_matrix","text":"eval_matrix(T, basis, x)\n\nReturn evaluation matrix from coefficients to sampling points. T <: AbstractSampling.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.evaluate!-Tuple{AbstractArray, SparseIR.AbstractSampling, Any}","page":"Home","title":"SparseIR.evaluate!","text":"evaluate!(buffer::AbstractArray{T,N}, sampling, al; dim=1) where {T,N}\n\nLike evaluate, but write the result to buffer. Please use dim = 1 or N to avoid allocating large temporary arrays internally.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.evaluate-Union{Tuple{N}, Tuple{T}, Tuple{Tmat}, Tuple{S}, Tuple{SparseIR.AbstractSampling{S, Tmat}, AbstractArray{T, N}}} where {S, Tmat, T, N}","page":"Home","title":"SparseIR.evaluate","text":"evaluate(sampling, al; dim=1)\n\nEvaluate the basis coefficients al at the sparse sampling points.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.findextrema","page":"Home","title":"SparseIR.findextrema","text":"findextrema(polyFT::PiecewiseLegendreFT, part=nothing, grid=_DEFAULT_GRID)\n\nObtain extrema of fourier-transformed polynomial.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.finite_temp_bases","page":"Home","title":"SparseIR.finite_temp_bases","text":"finite_temp_bases(β, wmax, ε, sve_result=compute_sve(LogisticKernel(β * wmax); ε))\n\nConstruct FiniteTempBasis objects for fermion and bosons using the same LogisticKernel instance.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.fit!-Union{Tuple{N}, Tuple{T}, Tuple{S}, Tuple{Array{S, N}, SparseIR.AbstractSampling, Array{T, N}}} where {S, T, N}","page":"Home","title":"SparseIR.fit!","text":"fit!(buffer, sampling, al::Array{T,N}; dim=1)\n\nLike fit, but write the result to buffer. Please use dim = 1 or N to avoid allocating large temporary arrays internally. The length of workarry cannot be smaller than the returned value of workarrlengthfit.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.fit-Union{Tuple{N}, Tuple{T}, Tuple{Tmat}, Tuple{S}, Tuple{SparseIR.AbstractSampling{S, Tmat}, AbstractArray{T, N}}} where {S, Tmat, T, N}","page":"Home","title":"SparseIR.fit","text":"fit(sampling, al::AbstractArray{T,N}; dim=1)\n\nFit basis coefficients from the sparse sampling points Please use dim = 1 or N to avoid allocating large temporary arrays internally.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.from_IR","page":"Home","title":"SparseIR.from_IR","text":"From IR to SPR\n\ngl:     Expansion coefficients in IR\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.get_symmetrized-Tuple{SparseIR.AbstractKernel, Any}","page":"Home","title":"SparseIR.get_symmetrized","text":"get_symmetrized(kernel, sign)\n\nConstruct a symmetrized version of kernel, i.e. kernel(x, y) + sign * kernel(x, -y).\n\nwarning: Beware!\nBy default, this returns a simple wrapper over the current instance which naively performs the sum.  You may want to override this to avoid cancellation.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.getwmax-Tuple{FiniteTempBasis}","page":"Home","title":"SparseIR.getwmax","text":"getwmax(basis::FiniteTempBasis)\n\nReal frequency cutoff.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.giw-Tuple{Any, Integer}","page":"Home","title":"SparseIR.giw","text":"giw(polyFT, wn)\n\nReturn model Green's function for reduced frequencies\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.iscentrosymmetric-Tuple{SparseIR.AbstractKernel}","page":"Home","title":"SparseIR.iscentrosymmetric","text":"is_centrosymmetric(kernel)\n\nReturn true if kernel(x, y) == kernel(-x, -y) for all values of x and y  in range. This allows the kernel to be block-diagonalized, speeding up the singular value expansion by a factor of 4.  Defaults to false.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.iswellconditioned-Tuple{DimensionlessBasis}","page":"Home","title":"SparseIR.iswellconditioned","text":"iswellconditioned(basis)\n\nReturn true if the sampling is expected to be well-conditioned.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.joinrules-Union{Tuple{AbstractArray{SparseIR.Rule{T}, 1}}, Tuple{T}} where T<:AbstractFloat","page":"Home","title":"SparseIR.joinrules","text":"joinrules(rules)\n\nJoin multiple Gauss quadratures together.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.legder-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"Home","title":"SparseIR.legder","text":"legder\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.legendre-Tuple{Any}","page":"Home","title":"SparseIR.legendre","text":"legendre(n[, T])\n\nGauss-Legendre quadrature with n points on [-1, 1].\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.legendre_collocation","page":"Home","title":"SparseIR.legendre_collocation","text":"legendre_collocation(rule, n=length(rule.x))\n\nGenerate collocation matrix from Gauss-Legendre rule.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.legvander-Union{Tuple{T}, Tuple{AbstractVector{T}, Integer}} where T","page":"Home","title":"SparseIR.legvander","text":"legvander(x, deg)\n\nPseudo-Vandermonde matrix of degree deg.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.matop!-Union{Tuple{N}, Tuple{T}, Tuple{S}, Tuple{AbstractArray{S, N}, Any, AbstractArray{T, N}, Any, Any}} where {S, T, N}","page":"Home","title":"SparseIR.matop!","text":"matop!(buffer, mat, arr::AbstractArray, op, dim)\n\nApply the operator op to the matrix mat and to the array arr along the first dimension (dim=1) or the last dimension (dim=N).\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.matop_along_dim!-Union{Tuple{N}, Tuple{T}, Tuple{Any, Any, AbstractArray{T, N}, Any, Any}} where {T, N}","page":"Home","title":"SparseIR.matop_along_dim!","text":"matop_along_dim!(buffer, mat, arr::AbstractArray, dim::Integer, op)\n\nApply the operator op to the matrix mat and to the array arr along the dimension dim, writing the result to buffer.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.matrices-Tuple{SparseIR.SamplingSVE}","page":"Home","title":"SparseIR.matrices","text":"matrices(sve::AbstractSVE)\n\nSVD problems underlying the SVE.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.matrix_from_gauss-Tuple{Any, Any, Any}","page":"Home","title":"SparseIR.matrix_from_gauss","text":"matrix_from_gauss(kernel, gauss_x, gauss_y)\n\nCompute matrix for kernel from Gauss rules.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.movedim-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T, N}, Pair}} where {T, N}","page":"Home","title":"SparseIR.movedim","text":"movedim(arr::AbstractArray, src => dst)\n\nMove arr's dimension at src to dst while keeping the order of the remaining dimensions unchanged.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.ngauss-Tuple{SparseIR.SVEHintsLogistic}","page":"Home","title":"SparseIR.ngauss","text":"ngauss(hints)\n\nGauss-Legendre order to use to guarantee accuracy.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.nsvals","page":"Home","title":"SparseIR.nsvals","text":"nsvals(hints)\n\nUpper bound for number of singular values.\n\nUpper bound on the number of singular values above the given threshold, i.e. where s[l] ≥ ε * first(s).\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.overlap-Union{Tuple{T}, Tuple{SparseIR.PiecewiseLegendrePoly{T}, Any}} where T","page":"Home","title":"SparseIR.overlap","text":"overlap(poly::PiecewiseLegendrePoly, f)\n\nEvaluate overlap integral of poly with arbitrary function f.\n\nGiven the function f, evaluate the integral::\n\n∫ dx * f(x) * poly(x)\n\nusing adaptive Gauss-Legendre quadrature.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.piecewise-Tuple{Any, Vector}","page":"Home","title":"SparseIR.piecewise","text":"piecewise(rule, edges)\n\nPiecewise quadrature with the same quadrature rule, but scaled.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.postprocess","page":"Home","title":"SparseIR.postprocess","text":"postprocess(sve::AbstractSVE, u, s, v, T=nothing)\n\nConstruct the SVE result from the SVD.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.quadrature-Tuple{Any, Any}","page":"Home","title":"SparseIR.quadrature","text":"quadrature(rule, f)\n\nApproximate f's integral.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.reseat-Tuple{SparseIR.Rule, Any, Any}","page":"Home","title":"SparseIR.reseat","text":"reseat(rule, a, b)\n\nReseat quadrature rule to new domain.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.roots-Tuple{SparseIR.PiecewiseLegendrePoly}","page":"Home","title":"SparseIR.roots","text":"roots(poly)\n\nFind all roots of the piecewise polynomial poly.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.scale-Tuple{Any, Any}","page":"Home","title":"SparseIR.scale","text":"scale(rule, factor)\n\nScale weights by factor.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.segments_x","page":"Home","title":"SparseIR.segments_x","text":"segments_x(kernel)\n\nSegments for piecewise polynomials on the x axis.\n\nList of segments on the x axis for the associated piecewise polynomial. Should reflect the approximate position of roots of a high-order singular function in x.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.segments_y","page":"Home","title":"SparseIR.segments_y","text":"segments_y(kernel)\n\nSegments for piecewise polynomials on the y axis.\n\nList of segments on the y axis for the associated piecewise polynomial. Should reflect the approximate position of roots of a high-order singular function in y.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.sve_hints","page":"Home","title":"SparseIR.sve_hints","text":"sve_hints(kernel, ε)\n\nProvide discretisation hints for the SVE routines.\n\nAdvises the SVE routines of discretisation parameters suitable in tranforming the (infinite) SVE into an (finite) SVD problem.\n\nSee also AbstractSVEHints.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.to_IR","page":"Home","title":"SparseIR.to_IR","text":"From SPR to IR\n\ng_spr:     Expansion coefficients in SPR\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.truncate","page":"Home","title":"SparseIR.truncate","text":"truncate(u, s, v[, rtol][, lmax])\n\nTruncate singular value expansion.\n\nArguments\n\n- `u`, `s`, `v`: Thin singular value expansion\n- `rtol` : If given, only singular values satisfying `s[l]/s[0] > rtol` are retained.\n- `lmax` : If given, at most the `lmax` most significant singular values are retained.\n\n\n\n\n\n","category":"function"},{"location":"#SparseIR.weight_func-Tuple{SparseIR.AbstractKernel, SparseIR.Statistics}","page":"Home","title":"SparseIR.weight_func","text":"weight_func(kernel, statistics::Statistics)\n\nReturn the weight function for the given statistics.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.workarrlength-Tuple{SparseIR.AbstractSampling, AbstractArray}","page":"Home","title":"SparseIR.workarrlength","text":"workarrlength(smpl::AbstractSampling, al; dim=1)\n\nReturn length of workarr for fit!.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.xrange-Tuple{SparseIR.AbstractKernel}","page":"Home","title":"SparseIR.xrange","text":"xrange(kernel)\n\nReturn a tuple (x_mathrmmin x_mathrmmax) delimiting the range  of allowed x values.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.ypower-Tuple{SparseIR.AbstractKernel}","page":"Home","title":"SparseIR.ypower","text":"ypower(kernel)\n\nPower with which the y coordinate scales.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.yrange-Tuple{SparseIR.AbstractKernel}","page":"Home","title":"SparseIR.yrange","text":"yrange(kernel)\n\nReturn a tuple (y_mathrmmin y_mathrmmax) delimiting the range  of allowed y values.\n\n\n\n\n\n","category":"method"},{"location":"#SparseIR.Λ-Tuple{DimensionlessBasis}","page":"Home","title":"SparseIR.Λ","text":"Λ(basis)\n\nBasis cutoff parameter Λ = β * ωmax.\n\n\n\n\n\n","category":"method"}]
}
