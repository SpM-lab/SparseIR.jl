using Test
using SparseIR

@testset "SparseIR.jl" begin
    include("conftest.jl")
    include("gauss.jl")
    include("kernel.jl")
    include("poly.jl")
    # include("sve.jl")
end
