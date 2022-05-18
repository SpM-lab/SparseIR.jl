"Intermediate representation (IR) for many-body propagators"
module SparseIR

using MultiFloats: Float64x2
Base.sinh(x::Float64x2) = Float64x2(sinh(big(x)))
Base.cosh(x::Float64x2) = Float64x2(cosh(big(x)))
Base.Math.hypot(x::Float64x2, y::Float64x2) = Base.Math._hypot(x, y) # TODO: only needed until MultiFloats is fixed

using IntervalRootFinding: IntervalRootFinding, Interval, isunique, interval, mid, Newton
using LinearAlgebra: dot, svd, SVD, QRIteration
using QuadGK: gauss, kronrod, quadgk
using SpecialFunctions: SpecialFunctions

export fermion, boson
export DimensionlessBasis, FiniteTempBasis
export SparsePoleRepresentation, to_IR, from_IR
export overlap
export LegendreBasis, MatsubaraConstBasis
export FiniteTempBasisSet
export LogisticKernel, RegularizedBoseKernel
export CompositeBasis, CompositeBasisFunction, CompositeBasisFunctionFT
export TauSampling, MatsubaraSampling, evaluate, fit

@enum Statistics boson fermion

include("_specfuncs.jl")
include("_linalg.jl")
using ._LinAlg: tsvd

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

# Precompile
precompile(FiniteTempBasis, (Statistics, Float64, Float64, Float64))
for cls in [:TauSampling, :MatsubaraSampling]
    for func in [:fit, :evaluate]
        for vartype in [:Float64, :ComplexF64]
            for dim in [:1, :2, :3, :4, :5, :6, :7]
                @eval precompile($(func), ($(cls), Array{$(vartype),$(dim)}))
            end
        end
    end
end

end # module
