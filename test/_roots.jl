using Test
using SparseIR

@testset "_roots.jl" begin
    @testset "refine_grid" begin
        @testset "Basic refinement" begin
            # Test with α = 2
            grid = [0.0, 1.0, 2.0]
            refined = SparseIR.refine_grid(grid, Val(2))
            @test length(refined) == 5  # α * (n-1) + 1 = 2 * (3-1) + 1 = 5
            @test refined ≈ [0.0, 0.5, 1.0, 1.5, 2.0]

            # Test with α = 3
            refined3 = SparseIR.refine_grid(grid, Val(3))
            @test length(refined3) == 7  # α * (n-1) + 1 = 3 * (3-1) + 1 = 7
            @test refined3 ≈ [0.0, 1/3, 2/3, 1.0, 4/3, 5/3, 2.0]

            # Test with α = 4
            refined4 = SparseIR.refine_grid(grid, Val(4))
            @test length(refined4) == 9  # α * (n-1) + 1 = 4 * (3-1) + 1 = 9
            @test refined4 ≈ [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        end

        #=
        Currently the SparseIR.refine_grid function is used only for the
        SparseIR.find_all function, which should accept only Float64 grids, but
        just in case it is a good idea to test that it works for other types.
        =#
        @testset "Type stability" begin
            # Integer grid
            int_grid = [0, 1, 2]
            refined_int = SparseIR.refine_grid(int_grid, Val(2))
            @test eltype(refined_int) === Float64
            @test refined_int == [0.0, 0.5, 1.0, 1.5, 2.0]

            # Float32 grid
            f32_grid = Float32[0, 1, 2]
            refined_f32 = SparseIR.refine_grid(f32_grid, Val(2))
            @test eltype(refined_f32) === Float32
            @test refined_f32 ≈ Float32[0, 0.5, 1, 1.5, 2]
        end

        @testset "Edge cases" begin
            # Single interval
            single_interval = [0.0, 1.0]
            refined_single = SparseIR.refine_grid(single_interval, Val(4))
            @test length(refined_single) == 5  # α * (2-1) + 1 = 4 * 1 + 1 = 5
            @test refined_single ≈ [0.0, 0.25, 0.5, 0.75, 1.0]

            # Empty grid
            empty_grid = Float64[]
            @test isempty(SparseIR.refine_grid(empty_grid, Val(2)))

            # Single point
            single_point = [1.0]
            @test SparseIR.refine_grid(single_point, Val(2)) == [1.0]
        end

        @testset "Uneven spacing" begin
            # Test with unevenly spaced grid
            uneven = [0.0, 1.0, 10.0]
            refined_uneven = SparseIR.refine_grid(uneven, Val(2))
            @test length(refined_uneven) == 5
            @test refined_uneven[1:3] ≈ [0.0, 0.5, 1.0]  # First interval
            @test refined_uneven[3:5] ≈ [1.0, 5.5, 10.0] # Second interval
        end

        @testset "Preservation of endpoints" begin
            grid = [-1.0, 0.0, 1.0]
            for α in [2, 3, 4]
                refined = SparseIR.refine_grid(grid, Val(α))
                @test first(refined) == first(grid)
                @test last(refined) == last(grid)
            end
        end
    end

    @testset "discrete_extrema" begin
        nonnegative = collect(0:8)
        symmetric = collect(-8:8)
        @test SparseIR.discrete_extrema(x -> x, nonnegative) == [8]
        @test SparseIR.discrete_extrema(x -> x - eps(), nonnegative) == [0, 8]
        @test SparseIR.discrete_extrema(x -> x^2, symmetric) == [-8, 0, 8]
        @test SparseIR.discrete_extrema(x -> 1, symmetric) == []
    end

    @testset "midpoint" begin
        @test SparseIR.midpoint(typemax(Int), typemax(Int)) === typemax(Int)
        @test SparseIR.midpoint(typemin(Int), typemax(Int)) === -1
        @test SparseIR.midpoint(typemin(Int), typemin(Int)) === typemin(Int)
        @test SparseIR.midpoint(Int16(1000), Int32(2000)) === Int32(1500)
        @test SparseIR.midpoint(floatmax(Float64), floatmax(Float64)) === floatmax(Float64)
        @test SparseIR.midpoint(Float16(0), floatmax(Float32)) === floatmax(Float32) / 2
        @test SparseIR.midpoint(Float16(0), floatmax(BigFloat)) == floatmax(BigFloat) / 2
        @test SparseIR.midpoint(Int16(0), big"99999999999999999999") ==
              big"99999999999999999999" ÷ 2
        @test SparseIR.midpoint(-10.0, 1) === -4.5
    end
end
