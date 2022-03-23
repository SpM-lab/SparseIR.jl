using Test

using SparseIR

const BASES = [(:F, 10), (:F, 42, :F, 10_000)]

function check_smooth(u, s, uscale, fudge_factor)
    eps = eps(eltype(s))
    x = u.knots[(begin + 1):(end - 1)]

    jump = @. abs(u(x + eps) - u(x - eps))
    compare = @. abs(u(x + 3eps) - u(x + eps))
    compare = max.(compare, uscale * eps)

    # loss of precision
    compare .*= fudge_factor * (first(s) ./ s)
    @test all(jump .< compare)
end

@testset "sve.jl" begin
    @testset "smooth" begin
        for Λ in [10, 42, 10_000]
            basis = IRBasis(:F, Λ; sve_result=sve_logistic[Λ])
            check_smooth(basis.u, basis.s, maximum(2 * basis.u(1)), 24)
            check_smooth(basis.v, basis.s, 50, 20)
        end
    end

    @testset "num roots u" begin
        for Λ in [10, 42, 10_000]
            basis = IRBasis(:F, Λ; sve_result=sve_logistic[Λ])
            for i in 1:length(basis.u)
                ui_roots = roots(basis.u[i])
                @test length(ui_roots) == i
            end
        end
    end

    @testset "num roots û" begin
        for stat in [:F, :B], Λ in [10, 42, 10_000]
            basis = IRBasis(stat, Λ; sve_result=sve_logistic[Λ])
            for i in [1, 2, 8, 11]
                x0 = extrema(basis.uhat[i])
                @test i + 1 <= length(x0) <= i + 2
            end
        end
    end
end