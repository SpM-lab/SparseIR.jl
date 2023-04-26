using Test
using SparseIR

@testset "freq.jl" begin
    @testset "freq" begin
        @test SparseIR.zeta(MatsubaraFreq(2)) == 0
        @test SparseIR.zeta(MatsubaraFreq(-5)) == 1

        @test Int(FermionicFreq(3)) == 3
        @test Int(BosonicFreq(-2)) == -2

        @test Int(MatsubaraFreq(Int32(4))) == 4

        @test_throws ErrorException FermionicFreq(4)
        @test_throws ErrorException BosonicFreq(-7)

        @test FermionicFreq(5) < BosonicFreq(6)
        @test BosonicFreq(6) >= BosonicFreq(6)

        @test SparseIR.value(pioverbeta, 3) == π / 3
        @test SparseIR.valueim(2 * pioverbeta, 3) == 2im * π / 3

        @test !iszero(MatsubaraFreq(-3))
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
        @test length(FermionicFreq(1):FermionicFreq(-3)) == length(1:2:-3)
        @test length(FermionicFreq(3):FermionicFreq(200_000_000_001)) == length(3:2:200_000_000_001)

        @test collect(BosonicFreq(2):BosonicFreq(-2)) == []
        @test collect(BosonicFreq(0):BosonicFreq(4)) == (0:2:4) .* pioverbeta
        @test collect(FermionicFreq(-3):FermionicFreq(1)) == (-3:2:1) .* pioverbeta
        
        @test length(FermionicFreq(37):BosonicFreq(10):FermionicFreq(87)) == length(37:10:87)
        @test collect(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(58)) == (-10:4:58) .* pioverbeta
        @test collect(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(60)) == (-10:4:58) .* pioverbeta
        @test length(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(60)) == length(-10:4:60)
        @test length(FermionicFreq(1):BosonicFreq(100):FermionicFreq(3)) == length(1:100:3)
        @test length(FermionicFreq(1):BosonicFreq(100):FermionicFreq(-1001)) == length(1:100:-1001)

        @test_throws ErrorException FermionicFreq(5):BosonicFreq(100)
        @test_throws ErrorException BosonicFreq(6):FermionicFreq(3):BosonicFreq(100)
        @test_throws ErrorException FermionicFreq(7):FermionicFreq(3):FermionicFreq(101)
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
        @test BosonicFreq(24) % FermionicFreq(-7) == FermionicFreq(3)
        @test FermionicFreq(123) % FermionicFreq(9) == BosonicFreq(6)
        @test promote_type(BosonicFreq, FermionicFreq) == MatsubaraFreq
    end

    @testset "broadcasting over frequency range" begin
        freqrange = FermionicFreq(-11):FermionicFreq(13)
        @test freqrange .+ BosonicFreq(10) == FermionicFreq(-1):FermionicFreq(23)
        @test freqrange .+ BosonicFreq(20) isa StepRange

        freqrange = FermionicFreq(-123):BosonicFreq(10):FermionicFreq(1237)
        @test freqrange .- FermionicFreq(11) == BosonicFreq(-134):BosonicFreq(10):BosonicFreq(1226)
    end
end
