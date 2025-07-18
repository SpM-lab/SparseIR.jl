@testitem "freq" begin
    @test SparseIR.zeta(MatsubaraFreq(2)) == 0
    @test SparseIR.zeta(MatsubaraFreq(-5)) == 1

    @test Int(FermionicFreq(3)) === 3
    @test Int(BosonicFreq(-2)) === -2

    @test Int(MatsubaraFreq(Int32(4))) === 4

    @test_throws DomainError FermionicFreq(4)
    @test_throws DomainError BosonicFreq(-7)

    @test FermionicFreq(5) < BosonicFreq(6)
    @test BosonicFreq(6) >= BosonicFreq(6)

    @test SparseIR.value(pioverbeta, 3) == π / 3
    @test SparseIR.valueim(2 * pioverbeta, 3) == 2im * π / 3

    @test !iszero(MatsubaraFreq(-3))
end

@testitem "freqadd" begin
    @test +pioverbeta == pioverbeta
    @test iszero(pioverbeta - pioverbeta)

    @test pioverbeta + oneunit(pioverbeta) == 2 * pioverbeta
    @test Int(4 * pioverbeta) == 4
    @test Int(pioverbeta - 2 * pioverbeta) == -1
    @test iszero(zero(2 * pioverbeta))
end

@testitem "freqrange" begin
    @test length(FermionicFreq(1):FermionicFreq(-3)) == 0
    @test length(FermionicFreq(3):FermionicFreq(200_000_000_001)) ==
          100_000_000_000

    @test collect(BosonicFreq(2):BosonicFreq(-2)) == []
    @test collect(BosonicFreq(0):BosonicFreq(4)) == (0:2:4) .* pioverbeta
    @test collect(FermionicFreq(-3):FermionicFreq(1)) == (-3:2:1) .* pioverbeta

    @test length(FermionicFreq(37):BosonicFreq(10):FermionicFreq(87)) == 6
    @test collect(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(58)) ==
          (-10:4:58) .* pioverbeta
    @test collect(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(60)) ==
          (-10:4:58) .* pioverbeta
    @test length(BosonicFreq(-10):BosonicFreq(4):BosonicFreq(60)) == 18
    @test length(FermionicFreq(1):BosonicFreq(100):FermionicFreq(3)) == 1
    @test length(FermionicFreq(1):BosonicFreq(100):FermionicFreq(-1001)) == 0

    @test_throws MethodError FermionicFreq(5):BosonicFreq(100)
    @test_throws MethodError BosonicFreq(6):FermionicFreq(3):BosonicFreq(100)
    @test_throws MethodError FermionicFreq(7):FermionicFreq(3):FermionicFreq(101)
end

@testitem "freqinvalid" begin
    @test_throws ArgumentError BosonicFreq(2)==2
    @test_throws ArgumentError FermionicFreq(1)-1
end

@testitem "unit tests" begin
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

@testitem "findfirst (needs dimensionless div)" begin
    @test BosonicFreq(100) ÷ BosonicFreq(50) === 2
    @test FermionicFreq(101) ÷ BosonicFreq(50) === 2
    haystack = FermionicFreq(-31):BosonicFreq(6):FermionicFreq(111)
    needle = FermionicFreq(23)
    @test findfirst(==(needle), haystack) === 10 # 6 * (10-1) + (-31) == 23
    @test haystack[findfirst(==(needle), haystack)] === needle
    @test isnothing(findfirst(==(needle - oneunit(needle)), haystack))
    @test findfirst(==(first(haystack)), haystack) === 1
    @test findfirst(==(last(haystack)), haystack) === length(haystack)
end
