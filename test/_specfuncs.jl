@testitem "_specfuncs - legder" begin
    @test SparseIR.legder(rand(10, 20), 99) == zeros((1, 20))
    @test SparseIR.legder([1 2 3 4]') == [6 9 20]'
    @test SparseIR.legder([1 2 3 4]', 2) == [9 60]'
    @test SparseIR.legder([1 2 3 4]', 3) == [60]'
end
