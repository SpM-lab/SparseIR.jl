using Test
using SparseIR
using SparseIR.LinearAlgebra
using StableRNGs: StableRNG

isdefined(Main, :sve_logistic) || include("_conftest.jl")

@testset "poly.jl" begin
    @testset "StableRNG" begin
        # Useful for porting poly.jl to C++
        # https://github.com/SpM-lab/SparseIR.jl/issues/51
        rng = StableRNG(2024)
        data = rand(rng, 3, 3)
        knots = rand(rng, size(data, 2) + 1) |> sort
        @test data == [
            0.8177021060277301 0.7085670484724618 0.5033588232863977; 
            0.3804323567786363 0.7911959541742282 0.8268504271915096; 
            0.5425813266814807 0.38397463704084633 0.21626598379927042
        ]
        @test knots == [
            0.507134318967235, 0.5766150365607372, 0.7126662232433161, 0.7357313003784003
        ]
        @assert issorted(knots)

        drng = StableRNG(999)
        randsymm = rand(drng, 1:10)
        randsymm == 9
        ddata = rand(drng, 3, 3)
        ddata == [
            0.5328437345518631 0.8443074122979211 0.6722336389122814; 
            0.1799506228788046 0.6805545318460489 0.17641780726469292; 
            0.13124858727993338 0.2193663343416914 0.7756615110113394
        ]
    end

    @testset "PiecewiseLegendrePoly(data::Matrix, knots::Vector, l::Int)" begin
        rng = StableRNG(2024)
        data = rand(rng, 3, 3)
        knots = rand(rng, size(data, 2) + 1) |> sort
        l = 3

        pwlp = SparseIR.PiecewiseLegendrePoly(data, knots, l)
        @test pwlp.data == data
        @test pwlp.xmin == first(knots)
        @test pwlp.xmax == last(knots)
        @test pwlp.knots == knots
        @test pwlp.polyorder == size(data, 1)
        @test pwlp.symm == 0
    end

    @testset "PiecewiseLegendrePoly(data, p::PiecewiseLegendrePoly; symm=symm(p))" begin
        rng = StableRNG(2024)
        data = rand(rng, 3, 3)
        knots = rand(rng, size(data, 2) + 1) |> sort
        l = 3
        
        pwlp = SparseIR.PiecewiseLegendrePoly(data, knots, l)

        drng = StableRNG(999)
        randsymm = rand(drng, Int)
        ddata = rand(drng, 3, 3)
        ddata_pwlp = SparseIR.PiecewiseLegendrePoly(ddata, pwlp; symm=randsymm)
        
        @test ddata_pwlp.data == ddata
        @test ddata_pwlp.symm == randsymm
        for n in fieldnames(SparseIR.PiecewiseLegendrePoly)
            n === :data && continue
            n === :symm && continue
            @test getfield(pwlp, n) == getfield(ddata_pwlp, n)
        end
    end

    @testset "deriv" begin
        # independent from sve.jl
        # https://github.com/SpM-lab/SparseIR.jl/issues/51
        rng = StableRNG(2024)
        
        data = rand(rng, 3, 3)
        knots = rand(rng, size(data, 2) + 1) |> sort
        l = 3
        pwlp = SparseIR.PiecewiseLegendrePoly(data, knots, l)

        n = 1
        ddata = SparseIR.legder(pwlp.data, n)
        ddata .*= pwlp.inv_xs'
        
        deriv_pwlp = SparseIR.deriv(pwlp)

        @test deriv_pwlp.data == ddata
        @test deriv_pwlp.symm == 0

        for n in fieldnames(SparseIR.PiecewiseLegendrePoly)
            n === :data && continue
            n === :symm && continue
            @test getfield(pwlp, n) == getfield(deriv_pwlp, n)
        end
    end

    @testset "shape" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        l = length(s)
        @test size(u) == (l,)

        @test size(u[4]) == ()
        @test size(u[3:5]) == (3,)
    end

    @testset "knots" begin
        u, s, v = sve_logistic[42]
        @test first(u[1].knots) == -1.0
        @test last(u[1].knots) == 1.0
    end

    @testset "slice" begin
        sve_result = sve_logistic[42]

        basis = FiniteTempBasis{Fermionic}(4.2, 10.0; sve_result)
        @test length(basis[1:4]) == 4
    end

    @testset "eval" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        l = length(s)

        # Evaluate
        @test u(0.4) == [u[i](0.4) for i in 1:l]
        @test u.([0.4, -0.2]) == [[u[i](x) for i in 1:l] for x in (0.4, -0.2)]
    end

    @testset "matrix_hat" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        uhat = SparseIR.PiecewiseLegendreFTVector(u, Fermionic())

        n = MatsubaraFreq.([1, 3, 5, -1, -3, 5])
        result1 = uhat[1](n)
        result = uhat(reshape(n, (3, 2)))
        result_iter = reshape(uhat(n), (:, 3, 2))
        @test size(result1) == (length(n),)
        @test size(result) == size(result_iter)
        @test result == result_iter
    end

    @testset "overlap with Λ = $Λ" for Λ in (10, 42, 10_000)
        atol = 1e-13
        u, s, v = SparseIR.part(sve_logistic[Λ])

        # Keep only even number of polynomials
        u, s, v = u[1:(end - end % 2)], s[1:(end - end % 2)], v[1:(end - end % 2)]

        @test overlap(u[1], u[1])≈1 rtol=0 atol=atol
        @test overlap(u[1], u[2])≈0 rtol=0 atol=atol
        @test overlap(u[1], u[1]; points=[0.1, 0.2])≈1 rtol=0 atol=atol

        ref = float.(eachindex(s) .== 1)
        @test all(isapprox.(overlap(u[1], u), ref; rtol=0, atol))
    end

    @testset "overlap(poly::PiecewiseLegendrePoly, f::F)" begin
        # independent from sve.jl
        # https://github.com/SpM-lab/SparseIR.jl/issues/51
        rng = StableRNG(2024)
        
        data = rand(rng, 3, 3)
        knots = rand(rng, size(data, 2) + 1) |> sort
        l = 3
        pwlp = SparseIR.PiecewiseLegendrePoly(data, knots, l)

        ∫pwlp, ∫pwlp_err = (0.4934184996836404, 8.326672684688674e-17)
        
        @test overlap(pwlp, identity) ≈ ∫pwlp
        @test overlap(pwlp, identity, return_error=true) .≈ (∫pwlp, ∫pwlp_err)
    end

    @testset "roots(poly::PiecewiseLegendrePoly; tol=1e-10, alpha=Val(2))" begin
        # https://github.com/SpM-lab/SparseIR.jl/issues/51
        #=
        The following data and knots are generated by
        julia> using SparseIR
        julia> using Test
        julia> Λ = 1.0
        julia> sve_result = SparseIR.SVEResult(SparseIR.LogisticKernel(Λ))
        julia> basis = SparseIR.FiniteTempBasis{SparseIR.Fermionic}(1, Λ; sve_result)
        =#
        
        data = reshape([
            0.16774734206553019
            0.49223680914312595
            -0.8276728567928646
            0.16912891046582143
            -0.0016231275318572044
            0.00018381683946452256
            -9.699355027805034e-7
            7.60144228530804e-8
            -2.8518324490258146e-10
            1.7090590205708293e-11
            -5.0081401126025e-14
            2.1244236198427895e-15
            2.0478095258000225e-16
            -2.676573801530628e-16
            2.338165820094204e-16
            -1.2050663212312096e-16
            -0.16774734206553019
            0.49223680914312595
            0.8276728567928646
            0.16912891046582143
            0.0016231275318572044
            0.00018381683946452256
            9.699355027805034e-7
            7.60144228530804e-8
            2.8518324490258146e-10
            1.7090590205708293e-11
            5.0081401126025e-14
            2.1244236198427895e-15
            -2.0478095258000225e-16
            -2.676573801530628e-16
            -2.338165820094204e-16
            -1.2050663212312096e-16
        ], 16, 2)

        knots = [0.0, 0.5, 1.0]
        l = 3

        # expected behavior
        #=
        julia> @test basis.u[4].data == data
        julia> @test basis.u[4].knots == knots
        julia> @test basis.u[4].l == l
        =#
        
        pwlp = SparseIR.PiecewiseLegendrePoly(data, knots, l)
        @test SparseIR.roots(pwlp) == [
            0.1118633448586015
            0.4999999999999998
            0.8881366551413985
        ]
    end

    @testset "eval unique" begin
        u, s, v = SparseIR.part(sve_logistic[42])
        û = SparseIR.PiecewiseLegendreFTVector(u, Fermionic())

        # evaluate
        res1 = û([1, 3, 3, 1])
        idx = [1, 2, 2, 1]
        res2 = û([1, 3])[:, idx]
        @test res1 == res2
    end

    @testset "unit tests" begin
        u, s, v = SparseIR.part(sve_logistic[42])

        @test size(u[1](rand(30))) == (30,)

        @test_throws DomainError u(SparseIR.xmax(u) + 123)
        @test_throws DomainError u(SparseIR.xmin(u) - 123)

        int_result, int_error = SparseIR.overlap(u[1], u[1]; return_error=true)

        @test int_error < eps()

        u_linearcombination = 2u[1] - 3u[2]
        @test SparseIR.overlap(u_linearcombination, u[2]) ≈ -3

        @test size(u(rand(2, 3, 4))) == (length(u), 2, 3, 4)

        @test (SparseIR.xmin(u), SparseIR.xmax(u)) === (-1.0, 1.0)
        @test SparseIR.knots(u) == u[end].knots
        @test SparseIR.Δx(u) == u[1].Δx
        @test SparseIR.symm(u)[2] == SparseIR.symm(u[2])
        @test all(<(eps()), SparseIR.overlap(u, sin)[1:2:end])
        @test all(<(eps()), SparseIR.overlap(u, cos)[2:2:end])

        û = SparseIR.PiecewiseLegendreFTVector(u, Fermionic())

        @test length(SparseIR.moments(û)[1]) == length(û)
        û_weird = SparseIR.PiecewiseLegendreFT(-9u[2] + u[3], Fermionic())
        @test_throws ErrorException SparseIR.func_for_part(û_weird)
        @test SparseIR.phase_stable(u, 5) ≈ SparseIR.phase_stable(u, 5.0)
    end
end
