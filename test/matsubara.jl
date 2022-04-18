const ε = (@isdefined Double64) ? nothing : 1e-15

@testset "matsubara.jl" begin
    @testset "single pole" begin
        for stat in (fermion, boson), Λ in (1e1, 1e4)
            wmax = 1.0
            pole = 0.1 * wmax
            β = Λ / wmax
            basis = FiniteTempBasis(stat, β, wmax, ε; sve_result=sve_logistic[Λ])

            stat_shift = (stat == fermion) ? 1 : 0
            weight = (stat == fermion) ? 1 : 1 / tanh(0.5 * Λ * pole / wmax)
            gl = -basis.s .* basis.v(pole) * weight

            func_G(n) = 1 / (im * (2n + stat_shift) * π / β - pole)

            # Compute G(iwn) using unl
            matsu_test = Int[-1, 0, 1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
            prj_w = transpose(basis.uhat(2matsu_test .+ stat_shift))
            Giwn_t = prj_w * gl

            # Compute G(iwn) from analytic expression
            Giwn_ref = func_G.(matsu_test)

            magnitude = maximum(abs, Giwn_ref)
            diff = abs.(Giwn_t - Giwn_ref)
            tol = max(10 * last(basis.s) / first(basis.s), 1e-10)

            # Absolute error
            @test maximum(diff ./ magnitude) < tol

            # TODO this depends on the outer "power model" evaluation which doesn't work yet
            # Relative error
            @test maximum(abs, diff ./ Giwn_ref) < tol
        end
    end
end
