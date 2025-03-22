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
       MatsubaraSampling64F, MatsubaraSampling64B, TauSampling64, sampling_points,
       basis

using MultiFloats: MultiFloats, Float64x2, _call_big
# If we call MultiFloats.use_bigfloat_transcendentals() like MultiFloats
# recommends, we get an error during precompilation:
for name in (:exp, :sinh, :cosh, :expm1)
    eval(:(function Base.$name(x::Float64x2)
        Float64x2(_call_big($name, x, 107 + 20)) # 107 == precision(Float64x2)
    end))
end
using LinearAlgebra: LinearAlgebra, cond, dot, svd, SVD, QRIteration, mul!, norm
using QuadGK: gauss, quadgk
using Bessels: sphericalbesselj
using PrecompileTools: PrecompileTools, @compile_workload

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
    @compile_workload begin
        basis = FiniteTempBasis(Fermionic(), 1e-1, 1e-1)
        basis = FiniteTempBasis(Fermionic(), 1e-1, 1e-1, 1e-5)

        τ_smpl = TauSampling(basis)
        iω_smpl = MatsubaraSampling(basis)

        Gτ = evaluate(τ_smpl, basis.s)
        Giω = evaluate(iω_smpl, basis.s)

        fit(τ_smpl, Gτ)
        fit(iω_smpl, Giω)
    end
end

end # module
