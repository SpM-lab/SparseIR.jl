@testitem "alias" setup=[CommonTestData] begin
    β = 1
    ωmax = 10
    basis = FiniteTempBasis{Fermionic}(
        β, ωmax; sve_result=CommonTestData.sve_logistic[β * ωmax])
    @test TauSampling(basis) isa TauSampling64
end

@testitem "decomp" begin
    using Random
    using LinearAlgebra

    Random.seed!(420)
    A = randn(49, 39)

    Ad = svd(A)
    norm_A = first(Ad.S) / last(Ad.S)
    @test all(isapprox.(A, Matrix(Ad), atol=1e-15 * norm_A, rtol=0))

    x = randn(39)
    @test A * x≈Ad.U * Diagonal(Ad.S) * Ad.Vt * x atol=1e-14 * norm_A rtol=0

    x = randn(39, 3)
    @test A * x≈Ad.U * Diagonal(Ad.S) * Ad.Vt * x atol=2e-14 * norm_A rtol=0

    y = randn(49)
    @test A \ y≈Ad \ y atol=1e-14 * norm_A rtol=0

    y = randn(49, 2)
    @test A \ y≈Ad \ y atol=1e-14 * norm_A rtol=0
end

@testitem "don't factorize" setup=[CommonTestData] begin
    stat = Bosonic()
    Λ = 10
    basis = FiniteTempBasis(stat, 1, Λ; sve_result=CommonTestData.sve_logistic[Λ])
    τ_smpl = TauSampling(basis; factorize=false)
    ω_smpl = MatsubaraSampling(basis; factorize=false)
    @test isnothing(τ_smpl.matrix_svd)
    @test isnothing(ω_smpl.matrix_svd)
end

@testitem "fit from tau" setup=[CommonTestData] begin
    using Random

    for stat in (Bosonic(), Fermionic()), Λ in (10, 42)
        basis = FiniteTempBasis(stat, 1, Λ; sve_result=CommonTestData.sve_logistic[Λ])
        smpl = TauSampling(basis)
        @test issorted(smpl.sampling_points)
        Random.seed!(5318008)

        shape = (2, 3, 4)
        rhol = randn(ComplexF64, (length(basis), shape...))
        originalgl = -basis.s .* rhol
        for dim in 1:ndims(rhol)
            gl = SparseIR.movedim(originalgl, 1 => dim)
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
end

@testitem "τ noise" setup=[CommonTestData] begin
    using Random
    using LinearAlgebra

    for stat in (Bosonic(), Fermionic()), Λ in (10, 42)
        basis = FiniteTempBasis(stat, 1, Λ; sve_result=CommonTestData.sve_logistic[Λ])
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
end

@testitem "iω noise" setup=[CommonTestData] begin
    using Random
    using LinearAlgebra

    for stat in (Bosonic(), Fermionic()), Λ in (10, 42), positive_only in (false, true)
        basis = FiniteTempBasis(stat, 1, Λ; sve_result=CommonTestData.sve_logistic[Λ])
        smpl = MatsubaraSampling(basis; positive_only)
        @test basis === SparseIR.basis(smpl)
        if !positive_only
            @test smpl isa
                  (stat == Fermionic() ? MatsubaraSampling64F : MatsubaraSampling64B)
        end
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
end

@testitem "conditioning" begin
    using LinearAlgebra

    basis = FiniteTempBasis{Fermionic}(3, 3, 1e-6)
    @test cond(TauSampling(basis)) < 3
    @test cond(MatsubaraSampling(basis)) < 5
    @test_logs (:warn,
        r"Sampling matrix is poorly conditioned \(cond = \d\.\d+e\d+\)\.") TauSampling(
        basis;
        sampling_points=[
            1.0,
            1.0
        ])
    @test_logs (:warn,
        r"Sampling matrix is poorly conditioned \(cond = \d\.\d+e\d+\)\.") MatsubaraSampling(
        basis;
        sampling_points=[
            FermionicFreq(1),
            FermionicFreq(1)
        ])

    basis = FiniteTempBasis{Fermionic}(3, 3, 1e-2)
    @test cond(TauSampling(basis)) < 2
    @test cond(MatsubaraSampling(basis)) < 3
end

@testitem "errors" begin
    for stat in (Bosonic(), Fermionic()), sampling in (TauSampling, MatsubaraSampling)
        basis = FiniteTempBasis(stat, 3, 3, 1e-6)
        smpl = sampling(basis)
        @test_throws DimensionMismatch evaluate(smpl, rand(100))
        @test_throws DimensionMismatch evaluate!(rand(100), smpl, rand(100))
        @test_throws DimensionMismatch fit(smpl, rand(100))
        @test_throws DimensionMismatch fit!(rand(100), smpl, rand(100))
        @test_throws DomainError SparseIR.matop!(rand(2, 3, 4), rand(5, 6), rand(7, 8, 9),
            *, 2)
    end
end

@testitem "noalloc divs" begin
    using LinearAlgebra

    A = rand(ComplexF64, 3, 4)
    B = rand(ComplexF64, 4, 5)
    B_SVD = svd(B)
    Y = Matrix{ComplexF64}(undef, 3, 5)
    workarr = Vector{ComplexF64}(undef, 4 * 5)

    SparseIR.rdiv_noalloc!(Y, A, B_SVD, workarr)

    @test Y * transpose(B) ≈ A
end

@testitem "frequency range" begin
    basis = FiniteTempBasis{Fermionic}(3, 3, 1e-6)
    freqrange = SparseIR.frequency_range(12)
    smpl = MatsubaraSampling(basis; sampling_points=freqrange)
    @test sampling_points(smpl) !== freqrange
    @test sampling_points(smpl) == freqrange
end
