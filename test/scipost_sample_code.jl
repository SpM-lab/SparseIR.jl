using Test
using SparseIR
using FFTW

@testset "scipost_sample_code.jl" begin
    @testset "sample 2" begin
        Λ = 1000
        β = 100
        ωmax = Λ / β
        ϵ = 1e-8
        b = FiniteTempBasis(fermion, β, ωmax, ϵ)

        x = y = 0.1
        τ = 0.5β * (x + 1)
        ω = ωmax * y

        # All singular values
        @show b.s
        @show b.u[1](τ)
        @show b.v[1](ω)

        # n-th derivative of U_l(τ) and V_l(ω)
        for n in 1:2
            u_n = SparseIR.deriv.(b.u, n)
            v_n = SparseIR.deriv.(b.v, n)
            @show n, u_n[1](τ)
            @show n, v_n[1](ω)
        end

        # Compute u_{ln} as a matrix for the first
        # 10 non-nagative fermionic Matsubara frequencies
        # Fermionic/bosonic frequencies are denoted by odd/even integers.
        hatF_t = b.uhat(1:2:19)
        @show size(hatF_t)
    end

    @testset "sample 3" begin
        β = 1e3
        Λ = 1e5

        ωmax = Λ / β
        ϵ = 1e-15
        @show ωmax

        b = FiniteTempBasis(fermion, β, ωmax, ϵ)
        @show length(b)

        # Sparse sampling in τ
        smpl_τ = TauSampling(b)

        # Sparse sampling in Matsubara frequencies
        smpl_matsu = MatsubaraSampling(b)

        # Parameters
        nk_lin = 64
        U, kps = 2.0, [nk_lin, nk_lin]
        nω = length(SparseIR.sampling_points(smpl_matsu))
        nτ = length(SparseIR.sampling_points(smpl_τ))

        # Generate k mesh and non-interacting band energies
        nk = prod(kps)
        k1 = k2 = 2π * (0:(nk_lin - 1)) / nk_lin
        ek = @. -2 * (cos(k1) + cos(k2'))
        iω = SparseIR.valueim.(SparseIR.sampling_points(smpl_matsu), β)

        # G(iω, k): (nω, nk)
        gkf = 1.0 ./ (iω .- vec(ek)')

        # G(l, k): (L, nk)
        gkl = fit(smpl_matsu, gkf)

        # G(τ, k): (nτ, nk)
        gkt = evaluate(smpl_τ, gkl)

        # G(τ, r): (nτ, nk)
        grt = reshape(fft(reshape(gkt, (nτ, kps...)), (2, 3)), (nτ, nk))

        # Sigma(τ, r): (nτ, nk)
        srt = U .* U .* grt .* grt .* reverse(grt; dims=1)

        # Sigma(l, r): (L, nk)
        srl = fit(smpl_τ, srt)

        # Sigma(iω, r): (nω, nk)
        srf = evaluate(smpl_matsu, srl)

        # Sigma(l, r): (L, nk)
        srl = fit(smpl_τ, srt)

        # Sigma(iω, r): (nω, nk)
        srf = evaluate(smpl_matsu, srl)

        # Sigma(iω, k): (nω, kps[0], kps[1])
        srf = reshape(srf, (nω, kps...))
        skf = ifft(srf, (2, 3)) / nk^2
    end
end
