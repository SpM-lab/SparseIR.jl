"Intermediate representation (IR) for many-body propagators"
module SparseIR

using IntervalRootFinding: roots as roots_irf, Interval, isunique, interval, mid, Newton
using LinearAlgebra: svd, SVD, QRIteration
using LowRankApprox: psvd
using QuadGK: gauss, kronrod, quadgk
using SpecialFunctions: sphericalbesselj as sphericalbesselj_sf

export fermion, boson
export DimensionlessBasis, FiniteTempBasis, finite_temp_bases
export SparsePoleRepresentation, to_IR, from_IR
export PiecewiseLegendrePoly, PiecewiseLegendrePolyArray, roots, hat, overlap, deriv
export LegendreBasis, MatsubaraConstBasis
export FiniteTempBasisSet
export legendre, legendre_collocation, Rule, piecewise, quadrature, reseat
export LogisticKernel, RegularizedBoseKernel, get_symmetrized
export CompositeBasis, CompositeBasisFunction, CompositeBasisFunctionFT
export TauSampling, MatsubaraSampling, evaluate, fit

@enum Statistics fermion boson

include("_specfuncs.jl")

include("svd.jl")
include("gauss.jl")
include("poly.jl")
include("kernel.jl")
include("basis.jl")
include("sve.jl")
include("augment.jl")
include("composite.jl")
include("sampling.jl")
include("spr.jl")
include("basis_set.jl")

end # module
