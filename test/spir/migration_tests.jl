# The v2 replacements of v1 code given in the migration guide (README.md and
# llms.txt). The points returned by v1's `default_sampling_points(sve.u, L)` are
# the roots of the SVE function u_L on (-1, 1); here the same roots are checked
# with a larger basis, whose function U_L vanishes at them.

@testitem "migration: v1 SVE sampling points from a v2 basis" tags=[:julia, :spir] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    β, ωmax=10.0, 4.0
    for stat in (Fermionic(), Bosonic())
        basis = get_basis(stat, β, ωmax, 1e-8)
        large = get_basis(stat, β, ωmax, 1e-12)
        for L in (5, 12, length(basis))
            b = basis[1:L]
            x = 2 .* SparseIR.default_tau_sampling_points(b) ./ SparseIR.β(b) .- 1
            @test length(x) == L
            @test issorted(x)
            @test all(-1 .< x .< 1)
            @test x ≈ -reverse(x) atol=1e-12
            τ=SparseIR.β(b)/2 .* (x .+ 1)
            @test maximum(abs, large.u[L + 1](τ)) <
                  1e-8 * maximum(abs, large.u[L + 1](range(0, β; length=201)))

            y=default_omega_sampling_points(b) ./ SparseIR.ωmax(b)
            @test length(y) == L
            @test all(-1 .< y .< 1)
            @test maximum(abs, large.v[L + 1](SparseIR.ωmax(b) .* y)) <
                  1e-8 * maximum(abs, large.v[L + 1](range(-ωmax, ωmax; length=201)))
        end
    end
end
