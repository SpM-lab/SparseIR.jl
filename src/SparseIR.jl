module SparseIR

include("C_API.jl") # libsparseir
using .C_API

import LinearAlgebra
using LinearAlgebra: cond
using QuadGK: quadgk

export Fermionic, Bosonic
export MatsubaraFreq, BosonicFreq, FermionicFreq, pioverbeta
export FiniteTempBasis, FiniteTempBasisSet
export DiscreteLehmannRepresentation
export overlap
export LogisticKernel, RegularizedBoseKernel
export iscentrosymmetric
export xrange, yrange
export conv_radius
export weight_func
export sve_hints, segments_x, segments_y, nsvals, ngauss
export _get_ptr
export AugmentedBasis, TauConst, TauLinear, MatsubaraConst
export TauSampling, MatsubaraSampling, evaluate, fit, evaluate!, fit!,
       sampling_points, npoints
export from_IR, to_IR, npoles, get_poles, default_omega_sampling_points

function _is_column_major_contiguous(A::AbstractArray)
    strides(A) == cumprod((1, size(A)...)[1:(end - 1)])
end

include("freq.jl")
include("abstract.jl")
include("kernel.jl")
include("custom_kernel.jl")
include("sve.jl")
include("poly.jl")
include("basis.jl")
include("sampling.jl")
include("dlr.jl")
include("basis_set.jl")
include("augment.jl")

end # module SparseIR
