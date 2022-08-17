using Test
using SparseIR

include("_conftest.jl")

function check_smooth(u, s, uscale, fudge_factor)
    ε = eps(eltype(s))
    x = u.knots[(begin + 1):(end - 1)]

    jump = abs.(u(x .+ ε) - u(x .- ε))
    compare_below = abs.(u(x .- ε) - u(x .- 3ε))
    compare_above = abs.(u(x .+ 3ε) - u(x .+ ε))
    compare = min.(compare_below, compare_above)
    compare = max.(compare, uscale * ε)

    # loss of precision
    compare .*= fudge_factor * (first(s) ./ s)
    @test all(jump .< compare)
end

leaq(a, b; kwargs...) = (a <= b) || isapprox(a, b; kwargs...)
a ⪅ b = leaq(a, b)

@testset "sve.jl" begin
    @testset "smooth with Λ = $Λ" for Λ in (10, 42, 10_000)
        basis = FiniteTempBasis(Fermionic(), 1, Λ; sve_result=sve_logistic[Λ])
        check_smooth(basis.u, basis.s, 2 * maximum(basis.u(1)), 24)
        check_smooth(basis.v, basis.s, 50, 20)
    end

    @testset "num roots u with Λ = $Λ" for Λ in (10, 42, 10_000)
        basis = FiniteTempBasis(Fermionic(), 1, Λ; sve_result=sve_logistic[Λ])
        for ui in basis.u
            ui_roots = SparseIR.roots(ui)
            @test length(ui_roots) == ui.l
        end
    end

    @testset "num roots û with stat = $stat, Λ = $Λ" for stat in (Fermionic(), Bosonic()),
                                                          Λ in (10, 42, 10_000)

        basis = FiniteTempBasis(stat, 1, Λ; sve_result=sve_logistic[Λ])
        for i in [1, 2, 8, 11]
            x₀ = SparseIR.find_extrema(basis.uhat[i])
            @test i ≤ length(x₀) ≤ i + 1
        end
    end

    @testset "accuracy with stat = $stat, Λ = $Λ" for stat in (Fermionic(), Bosonic()),
                                                      Λ in (10, 42, 10_000)

        basis = FiniteTempBasis(stat, 4, Λ; sve_result=sve_logistic[Λ])
        @test 0 < SparseIR.accuracy(basis) ⪅ last(SparseIR.significance(basis))
        @test isone(first(SparseIR.significance(basis)))
        @test SparseIR.accuracy(basis) ⪅ last(basis.s) / first(basis.s)
    end
end
