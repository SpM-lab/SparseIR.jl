@testitem "poly.jl" tags = [:julia, :lib] begin
    using Test
    using SparseIR
    import SparseIR as SparseIR

    β = 10
    ωmax = 4
    ε = 1e-6
    basis = FiniteTempBasis{Fermionic}(β, ωmax, ε)

    ρ₀(ω) = 2 / π * √(1 - clamp(ω, -1, +1)^2)

    @testset "u" begin
        @test SparseIR.xmin(basis.u) == 0.0
        @test SparseIR.xmax(basis.u) == β
    end

    @testset "v" begin
        @test SparseIR.xmin(basis.v) == -ωmax
        @test SparseIR.xmax(basis.v) == ωmax
    end

    @testset "uhat" begin
        @test SparseIR.xmin(basis.uhat) == -1.0
        @test SparseIR.xmax(basis.uhat) == 1.0
    end

    @testset "overlap" begin
        ref_u = [0.27517437799713756, -0.33320547877174056, 0.20676709933869278,
            -0.07612175747193344, -0.04128171229889062,
            0.09679420081388303, -0.0989389721622601, 0.06747728769454617,
            -0.023901271045533704, -0.014714625415717688,
            0.03830797967004657, -0.043884983551122116, 0.034203081565519565,
            -0.015618570179708689, -0.004366240169617414, 0.01933538369730581]
        @test overlap(basis.u, ρ₀) ≈ ref_u
        ref_v = [0.6352548229644916, -2.4069288229178198e-17, -0.22782525665797093,
            -1.0842021724855044e-17, -0.1783957405542744,
            -8.023096076392733e-18, 0.18907994546506804, 8.348356728138384e-18,
            -0.07277201662947068, 2.3852447794681098e-18,
            -0.019708084751718376, -1.919037845299343e-17, 0.05824932416884472,
            -4.9873299934333204e-18, -0.05617142181206885, 1.734723475976807e-18]
        @test overlap(basis.v, ρ₀) ≈ ref_v
    end
end
