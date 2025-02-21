using SparseIR
using Test
using Aqua

@testset verbose=true "SparseIR tests" begin
    @testset verbose=true "Aqua" begin
        Aqua.test_all(SparseIR; ambiguities=false, piracies=false)
    end

    @testset verbose=true "Actual code" begin
        include("_conftest.jl")

        include("freq.jl")
        include("gauss.jl")
        include("kernel.jl")
        include("poly.jl")
        include("sve.jl")
        include("svd.jl")
        include("basis.jl")
        include("sampling.jl")
        include("augment.jl")
        include("dlr.jl")
        include("_linalg.jl")
        include("_roots.jl")
        include("_specfuncs.jl")
        include("_multifloat_funcs.jl")
        include("performance.jl")
    end
end
