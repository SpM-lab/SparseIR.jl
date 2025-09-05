@testitem "sampling" tags=[:julia, :spir] begin
    using Test
    using Random
    using SparseIR
    using LinearAlgebra: norm
    import SparseIR as SparseIR

    function getperm(N, src, dst)
        perm = collect(1:N)
        deleteat!(perm, src)
        insert!(perm, dst, src)
        return perm
    end

    """
        movedim(arr::AbstractArray, src => dst)

    Move `arr`'s dimension at `src` to `dst` while keeping the order of the remaining
    dimensions unchanged.
    """
    function movedim(arr::AbstractArray{T,N}, src, dst) where {T,N}
        src == dst && return arr
        return permutedims(arr, getperm(N, src, dst))
    end

    @testset "fit from tau with stat = $stat, Λ = $Λ" for stat in (Bosonic(), Fermionic()),
        Λ in (10, 42)

        sve_logistic = SparseIR.SVEResult(LogisticKernel(Λ), 1e-10)
        basis = FiniteTempBasis(stat, 1, Λ, 1e-10; sve_result=sve_logistic)
        smpl = TauSampling(basis)
        @test issorted(smpl.sampling_points)
        Random.seed!(5318008)

        shape = (2, 3, 4)
        rhol = randn(ComplexF64, (length(basis), shape...))
        originalgl = -basis.s .* rhol
        for dim in 1:ndims(rhol)
            gl = movedim(originalgl, 1, dim)
            gtau = evaluate(smpl, gl; dim)
            @test size(gtau) == (size(gl)[1:(dim - 1)]...,
                length(smpl.sampling_points),
                size(gl)[(dim + 1):end]...)

            gl_from_tau = fit(smpl, gtau; dim)
            @test gl_from_tau ≈ gl

            gl_from_tau2 = similar(gl_from_tau)
            fit!(gl_from_tau2, smpl, gtau; dim)
            @test gl_from_tau2 ≈ gl
        end
    end

    @testset "τ noise with stat = $stat, Λ = $Λ" for stat in (Bosonic(), Fermionic()),
        Λ in (10, 42)

        sve_logistic = SparseIR.SVEResult(LogisticKernel(Λ), 1e-10)
        basis = FiniteTempBasis(stat, 1, Λ, 1e-10; sve_result=sve_logistic)
        smpl = TauSampling(basis)
        @test basis === SparseIR.basis(smpl)
        @test issorted(smpl.sampling_points)
        Random.seed!(5318008)

        ρℓ = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
        Gℓ = basis.s .* ρℓ
        Gℓ_magn = norm(Gℓ)
        @inferred evaluate(smpl, Gℓ)
        @inferred evaluate(smpl, Gℓ, dim=1)
        Gτ = evaluate(smpl, Gℓ)

        Gτ_inplace = similar(Gτ)
        evaluate!(Gτ_inplace, smpl, Gℓ)
        @test Gτ == Gτ_inplace

        noise = 1e-5
        Gτ_n = Gτ + noise * norm(Gτ) * randn(size(Gτ)...)
        @inferred fit(smpl, Gτ_n)
        @inferred fit(smpl, Gτ_n, dim=1)
        Gℓ_n = fit(smpl, Gτ_n)

        Gℓ_n_inplace = similar(Gℓ_n)
        fit!(Gℓ_n_inplace, smpl, Gτ_n)
        @test Gℓ_n == Gℓ_n_inplace

        @test isapprox(Gℓ, Gℓ_n, atol=12 * noise * Gℓ_magn, rtol=0)
    end

    @testset "iω noise with stat = $stat, Λ = $Λ" for stat in (Bosonic(), Fermionic()),
        Λ in (10, 42),
        positive_only in (false, true)
        sve_logistic = SparseIR.SVEResult(LogisticKernel(Λ), 1e-10)
        basis = FiniteTempBasis(stat, 1, Λ, 1e-10; sve_result=sve_logistic)
        smpl = MatsubaraSampling(basis; positive_only)
        @test basis === SparseIR.basis(smpl)
        #=if !positive_only
            @test smpl isa
                (stat == Fermionic() ? MatsubaraSampling64F : MatsubaraSampling64B)
        end
        =#
        @test issorted(smpl.sampling_points)
        Random.seed!(1312 + 161)

        ρℓ = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
        Gℓ = basis.s .* ρℓ
        Gℓ_magn = norm(Gℓ)
        @inferred evaluate(smpl, Gℓ)
        @inferred evaluate(smpl, Gℓ, dim=1)
        Giw = evaluate(smpl, Gℓ)

        Giw_inplace = similar(Giw)
        evaluate!(Giw_inplace, smpl, Gℓ)
        @test Giw == Giw_inplace

        noise = 1e-5
        Giwn_n = Giw + noise * norm(Giw) * randn(size(Giw)...)
        @inferred fit(smpl, Giwn_n)
        @inferred fit(smpl, Giwn_n, dim=1)
        Gℓ_n = fit(smpl, Giwn_n)
        @test isapprox(Gℓ, Gℓ_n, atol=40 * sqrt(1 + positive_only) * noise * Gℓ_magn,
            rtol=0)

        Gℓ_n_inplace = similar(Gℓ_n)
        fit!(Gℓ_n_inplace, smpl, Giwn_n)
        @test Gℓ_n == Gℓ_n_inplace
    end

    @testset "errors with stat = $stat, $sampling" for stat in (Bosonic(), Fermionic()),
        sampling in (TauSampling,
            MatsubaraSampling)

        basis = FiniteTempBasis(stat, 3, 3, 1e-6)
        smpl = sampling(basis)
        @test_throws DimensionMismatch evaluate(smpl, rand(100))
        @test_throws DimensionMismatch evaluate!(rand(100), smpl, rand(100))
        @test_throws Exception fit(smpl, rand(100))
        @test_throws DimensionMismatch fit!(rand(100), smpl, rand(100))
    end

    @testset "frequency range" begin
        basis = FiniteTempBasis{Fermionic}(3, 3, 1e-6)
        freqrange = SparseIR.frequency_range(12)
        smpl = MatsubaraSampling(basis; sampling_points=freqrange)
        @test sampling_points(smpl) !== freqrange
        @test sampling_points(smpl) == freqrange
    end

    @testset "default_matsubara_sampling_points" begin
        β = 10.0
        ωmax = 1.0
        ε = 1e-10
        kernel = LogisticKernel(β * ωmax)
        basis = FiniteTempBasis(Fermionic(), β, ωmax, ε; kernel)
        points = SparseIR.default_matsubara_sampling_points(basis)
        @test length(points) > 0
    end
end
