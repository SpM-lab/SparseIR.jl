function check_smooth(u, s, uscale, fudge_factor)
    ε = eps(eltype(s))
    x = u.knots[(begin + 1):(end - 1)]

    jump = abs.(u(x .+ ε) - u(x .- ε))
    compare = abs.(u(x .+ 3ε) - u(x .+ ε))
    compare = max.(compare, uscale * ε)

    # loss of precision
    compare .*= fudge_factor * (first(s) ./ s)
    @test all(jump .< compare)
end

@testset "sve.jl" begin
    @testset "smooth" begin
        for Λ in [10, 42, 10_000]
            basis = IRBasis(fermion, Λ; sve_result=sve_logistic[Λ])
            check_smooth(basis.u, basis.s, 2 * maximum(basis.u(1)), 24)
            check_smooth(basis.v, basis.s, 50, 20)
        end
    end

    @testset "num roots u" begin
        for Λ in [10, 42]#, 10_000] 
            # TODO for Λ = 10_000 and i = 45 two roots aren't found :(
            basis = IRBasis(fermion, Λ; sve_result=sve_logistic[Λ])
            for i in 1:length(basis.u)
                ui_roots = roots(basis.u[i])
                @test length(ui_roots) == i - 1
            end
        end
    end

    @testset "num roots û" begin
        for stat in [fermion, boson], Λ in [10, 42, 10_000]
            basis = IRBasis(stat, Λ; sve_result=sve_logistic[Λ])
            for i in [1, 2, 8, 11]
                x₀ = extrema(basis.uhat[i])
                @test i ≤ length(x₀) ≤ i + 1
            end
        end
    end
end