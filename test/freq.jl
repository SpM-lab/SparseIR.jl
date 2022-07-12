using Test
using SparseIR

@testset "freq" begin

@testset "freq" begin
    @test SparseIR.zeta(MatsubaraFreq(2)) == 0
    @test SparseIR.zeta(MatsubaraFreq(-5)) == 1

    @test Integer(FermionicFreq(3)) == 3
    @test Integer(BosonicFreq(-2)) == -2

    @test Integer(MatsubaraFreq(Int32(4))) == 4

    @test_throws ArgumentError FermionicFreq(4)
    @test_throws ArgumentError BosonicFreq(-7)

    @test FermionicFreq(5) < BosonicFreq(6)
    @test BosonicFreq(6) >= BosonicFreq(6)
end

@testset "freqadd" begin
    @test +pioverbeta == pioverbeta
    @test iszero(pioverbeta - pioverbeta)

    @test pioverbeta + oneunit(pioverbeta) == 2 * pioverbeta
    @test Integer(4 * pioverbeta) == 4
    @test Integer(pioverbeta - 2 * pioverbeta) == -1
    @test iszero(zero(2 * pioverbeta))
end

@testset "freqrange" begin
    @test length(FermionicFreq(1):FermionicFreq(-3)) == 0
    @test length(FermionicFreq(3):FermionicFreq(200_000_000_001)) ==
            100_000_000_000

    @test collect(BosonicFreq(2):BosonicFreq(-2)) == []
    @test collect(BosonicFreq(0):BosonicFreq(4)) == (0:2:4) .* pioverbeta
    @test collect(FermionicFreq(-3):FermionicFreq(1)) == (-3:2:1) .* pioverbeta
end

end
