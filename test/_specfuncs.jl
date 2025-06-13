@testitem "_specfuncs - legder" begin
    @test SparseIR.legder(rand(10, 20), 99) == zeros((1, 20))
    @test SparseIR.legder([1 2 3 4]') == [6 9 20]'
    @test SparseIR.legder([1 2 3 4]', 2) == [9 60]'
    @test SparseIR.legder([1 2 3 4]', 3) == [60]'
end

@testitem "_specfuncs - legder edge case: cnt >= n" begin
    # Test line 56: when derivative order >= number of rows, return zeros(T, (1, m))
    c = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2 matrix

    # cnt = n (equal case)
    result = SparseIR.legder(c, 3)
    @test result == zeros(1, 2)
    @test size(result) == (1, 2)

    # cnt > n (greater case)
    result = SparseIR.legder(c, 5)
    @test result == zeros(1, 2)
    @test size(result) == (1, 2)
end
