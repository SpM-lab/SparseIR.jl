module SparseIR

export legendre, legvander, legendre_collocation, Rule, piecewise, quadrature, reseat, LogisticKernel, RegularizedBoseKernel, sve_hints, segments_x, segments_y, matrix_from_gauss, get_symmetrized

include("gauss.jl")
include("kernel.jl")

end # module
