using Test
using SparseIR
using SparseIR: Float64x2
using Logging: with_logger, NullLogger

isdefined(Main, :sve_logistic) || include("_conftest.jl")

function check_smooth(u, s, uscale, fudge_factor)
    ε = eps(eltype(s))
    x = SparseIR.knots(u)[(begin + 1):(end - 1)]

    jump = abs.(u(x .+ ε) - u(x .- ε))
    compare_below = abs.(u(x .- ε) - u(x .- 3ε))
    compare_above = abs.(u(x .+ 3ε) - u(x .+ ε))
    compare = min.(compare_below, compare_above)
    compare = max.(compare, uscale * ε)

    # loss of precision
    compare .*= fudge_factor .* (first(s) ./ s)
    @test all(jump .< compare)
end

leaq(a, b; kwargs...) = (a <= b) || isapprox(a, b; kwargs...)
a ⪅ b = leaq(a, b)

@testset "sve.jl" begin
    @testset "kernel with Λ = $Λ" for Λ in (10, 42, 10_000)
        kernel = LogisticKernel(Λ)
        sve_result = sve_logistic[Λ]
        ε = nothing
        u, s, v = SparseIR.part(sve_result; ε, max_size=typemax(Int))

        N = length(s)
        @test length(s) == N
        @test length(u) == length(s)
        @test length(v) == length(s)

        for i in 1:N
            @test overlap(u[i], u[i])≈1.0 atol=1e-10
        end

        if N < 40 # expensive
            for i in 1:N, j in 1:N
                ref = (i == j ? 1.0 : 0.0)
                @test overlap(u[i], u[j])≈ref atol=1e-10
            end
        else
            for i in 1:10:N, j in 1:N
                ref = (i == j ? 1.0 : 0.0)
                @test overlap(u[i], u[j])≈ref atol=1e-10
            end
        end

        x = [0.0, 0.5, 1.0]
        for i in 1:2:N
            @test u[i](x)≈u[i](-x) atol=abs(u[i](1)) * 1e-12
        end
        for i in 2:2:N
            @test u[i](x)≈-u[i](-x) atol=abs(u[i](1)) * 1e-12
        end

        # check k(x, y) \approx sum_l s_l u_l(x) v_l(y)
        for y in [0.0, 0.1, 0.5, 0.9, 1.0], x in [-1.0, -0.5, 0.0, 0.5, 1.0]
            diff = kernel(x, y) - sum(s[l] * u[l](x) * v[l](y) for l in 1:N)
            @test abs(diff) < 1e-12
        end
    end

    @testset "smooth with Λ = $Λ" for Λ in (10, 42, 10_000)
        basis = FiniteTempBasis{Fermionic}(1, Λ; sve_result=sve_logistic[Λ])
        check_smooth(basis.u, basis.s, 2 * maximum(basis.u(1)), 24)
        check_smooth(basis.v, basis.s, 50, 200)
    end

    @testset "num roots u with Λ = $Λ" for Λ in (10, 42, 10_000)
        basis = FiniteTempBasis{Fermionic}(1, Λ; sve_result=sve_logistic[Λ])
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

    @testset "choose_accuracy" begin
        with_logger(NullLogger()) do # suppress output of warnings
            @test SparseIR.choose_accuracy(nothing, nothing) ==
                  (2.2204460492503131e-16, Float64x2, :default)
            @test SparseIR.choose_accuracy(nothing, Float64) ==
                  (1.4901161193847656e-8, Float64, :default)
            @test SparseIR.choose_accuracy(nothing, Float64x2) ==
                  (2.2204460492503131e-16, Float64x2, :default)
            @test SparseIR.choose_accuracy(1e-6, nothing) == (1.0e-6, Float64, :default)
            @test SparseIR.choose_accuracy(1e-8, nothing) == (1.0e-8, Float64x2, :default)

            @test SparseIR.choose_accuracy(1e-20, nothing) == (1.0e-20, Float64x2, :default)
            @test_logs (:warn,
                """Basis cutoff is 1.0e-20, which is below √ε with ε = 4.9303806576313238e-32.
Expect singular values and basis functions for large l to have lower precision
than the cutoff.""") SparseIR.choose_accuracy(1e-20, nothing)

            @test SparseIR.choose_accuracy(1e-10, Float64) == (1.0e-10, Float64, :accurate)
            @test_logs (:warn,
                """Basis cutoff is 1.0e-10, which is below √ε with ε = 2.220446049250313e-16.
Expect singular values and basis functions for large l to have lower precision
than the cutoff.""") SparseIR.choose_accuracy(1e-10, Float64)

            @test SparseIR.choose_accuracy(1e-6, Float64) == (1.0e-6, Float64, :default)

            @test SparseIR.choose_accuracy(1e-6, Float64, :auto) ==
                  (1.0e-6, Float64, :default)
            @test SparseIR.choose_accuracy(1e-6, Float64, :accurate) ==
                  (1.0e-6, Float64, :accurate)
        end
    end

    @testset "truncate" begin
        sve = SparseIR.CentrosymmSVE(LogisticKernel(5), 1e-6, Float64)

        svds = SparseIR.compute_svd.(SparseIR.matrices(sve))
        u_, s_, v_ = zip(svds...)

        for lmax in 3:20
            u, s, v = SparseIR.truncate(u_, s_, v_; lmax)
            u, s, v = SparseIR.postprocess(sve, u, s, v)
            @test length(u) == length(s) == length(v)
            @test length(s) ≤ lmax - 1
        end
    end
end
