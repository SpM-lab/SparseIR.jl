using Test
using SparseIR

include("conftest.jl")

@testset "SparseIR.jl" begin
    include("gauss.jl")
    include("kernel.jl")
    include("poly.jl")
    include("sve.jl")
    include("basis_set.jl")
    include("sampling.jl")
    include("matsubara.jl")
end
