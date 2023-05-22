"""
Intermediate representation (IR) for many-body propagators.
"""
module SparseIR

export Fermionic, Bosonic
export MatsubaraFreq, BosonicFreq, FermionicFreq, pioverbeta
export FiniteTempBasis, FiniteTempBasisSet
export DiscreteLehmannRepresentation
export overlap
export LogisticKernel, RegularizedBoseKernel
export AugmentedBasis, TauConst, TauLinear, MatsubaraConst
export TauSampling, MatsubaraSampling, evaluate, fit, evaluate!, fit!,
       MatsubaraSampling64F, MatsubaraSampling64B, TauSampling64, sampling_points

using MultiFloats: Float64x2
using LinearAlgebra: LinearAlgebra, cond, dot, svd, SVD, QRIteration, mul!
using LinearAlgebra.BLAS: gemm!
using Logging: with_logger, NullLogger
using QuadGK: gauss, quadgk
using Bessels: sphericalbesselj
using PrecompileTools

# FIXME: These are piracy, but needed to make MultiFloats work for us.
Base.sinh(x::Float64x2) = 0.5 * (exp(x) - exp(-x))
Base.cosh(x::Float64x2) = 0.5 * (exp(x) + exp(-x))
Base.Math.hypot(x::Float64x2, y::Float64x2) = Base.Math._hypot(x, y)

include("_linalg.jl")
include("_roots.jl")
include("_specfuncs.jl")
using ._LinAlg: tsvd

include("freq.jl")
include("abstract.jl")
include("svd.jl")
include("gauss.jl")
include("poly.jl")
include("kernel.jl")
include("sve.jl")
include("basis.jl")
include("augment.jl")
include("sampling.jl")
include("dlr.jl")
include("basis_set.jl")

@static if VERSION ≥ v"1.9-" # 1.9 adds support for object caching
    # Precompile
    with_logger(Base.NullLogger()) do
        @compile_workload begin
            basis = FiniteTempBasis(Fermionic(), 1e-1, 1e-1, 1e-5)
            basis = FiniteTempBasis(Fermionic(), 1e-1, 1e-1)

            τ_smpl = TauSampling(basis)
            iω_smpl = MatsubaraSampling(basis)

            Gτ = evaluate(τ_smpl, basis.s)
            Giω = evaluate(iω_smpl, basis.s)

            fit(τ_smpl, Gτ)
            fit(iω_smpl, Giω)
        end
    end
end

end # module
