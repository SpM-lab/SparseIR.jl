@testitem "Aqua.jl" begin
    using Aqua

    Aqua.test_all(SparseIR; piracies=false)
end
