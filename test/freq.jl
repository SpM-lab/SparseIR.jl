using Test
using SparseIR

@testset "freq.jl" begin
    @testset "freq" begin
        @test SparseIR.zeta(MatsubaraFreq(2)) == 0
        @test SparseIR.zeta(MatsubaraFreq(-5)) == 1

        @test Int(FermionicFreq(3)) == 3
        @test Int(BosonicFreq(-2)) == -2

        @test Int(MatsubaraFreq(Int32(4))) == 4

        @test_throws ArgumentError FermionicFreq(4)
        @test_throws ArgumentError BosonicFreq(-7)

        @test FermionicFreq(5) < BosonicFreq(6)
        @test BosonicFreq(6) >= BosonicFreq(6)

        @test SparseIR.value(pioverbeta, 3) == π / 3
        @test SparseIR.valueim(2 * pioverbeta, 3) == 2im * π / 3
    end

    @testset "freqadd" begin
        @test +pioverbeta == pioverbeta
        @test iszero(pioverbeta - pioverbeta)

        @test pioverbeta + oneunit(pioverbeta) == 2 * pioverbeta
        @test Int(4 * pioverbeta) == 4
        @test Int(pioverbeta - 2 * pioverbeta) == -1
        @test iszero(zero(2 * pioverbeta))
    end

    @testset "freqrange" begin
        @test length(FermionicFreq(1):FermionicFreq(-3)) == 0
        @test length(FermionicFreq(3):FermionicFreq(200_000_000_001)) ==
              100_000_000_000

        @test collect(BosonicFreq(2):BosonicFreq(-2)) == []
        @test collect(BosonicFreq(0):BosonicFreq(4)) == (0:2:4) .* pioverbeta
        @test collect(FermionicFreq(-3):FermionicFreq(1)) == (-3:2:1) .* pioverbeta
        
        @test length(FermionicFreq(37):BosonicFreq(10):FermionicFreq(87)) == 6
        @test collect(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(58)) == (-10:4:58) .* pioverbeta
        @test collect(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(60)) == (-10:4:58) .* pioverbeta
        @test length(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(60)) == 18
        @test length(FermionicFreq(1):BosonicFreq(100):FermionicFreq(3)) == 1
        @test length(FermionicFreq(1):BosonicFreq(100):FermionicFreq(-1001)) == 0
    end

    @testset "freqinvalid" begin
        @test_throws ArgumentError BosonicFreq(2) == 2
        @test_throws ArgumentError FermionicFreq(1) - 1
    end

    @testset "unit tests" begin
        @test_throws DomainError SparseIR.Statistics(2)
        @test SparseIR.Statistics(0) + SparseIR.Statistics(1) == Fermionic()
        @test Integer(FermionicFreq(19)) == 19
        @test -BosonicFreq(-24) == BosonicFreq(24)
        @test sign(BosonicFreq(24)) == 1
        @test sign(BosonicFreq(0)) == 0
        @test sign(BosonicFreq(-94)) == -1
        @test promote_type(BosonicFreq, FermionicFreq) == MatsubaraFreq

        io = IOBuffer()
        show(io, FermionicFreq(-3))
        @test String(take!(io)) == "-3π/β"
        show(io, FermionicFreq(1))
        @test String(take!(io)) == "π/β"
        show(io, BosonicFreq(0))
        @test String(take!(io)) == "0"
    end
end
