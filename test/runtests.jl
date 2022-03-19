using Test
using SparseIR

@testset "SparseIR.jl" begin
    include("gauss.jl")
    include("kernel.jl")
end
