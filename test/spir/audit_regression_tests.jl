@testitem "audit: DLR pole element types and validation" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using LinearAlgebra

    β = 10.0
    ωmax = 4.0
    ε = 1e-8
    basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)

    reference = SparseIR.get_poles(DiscreteLehmannRepresentation(basis, Float64[1.0, -1.0]))

    # Finding A: any real element type must be converted to Float64 before the
    # pointer is taken, and must give bit-identical poles -- not silent zeros
    # (Int64) and not a reinterpreted Float32 buffer.
    @testset "element type $T" for T in (Float64, Float32, Float16, Int64, Int32, Rational{Int})
        dlr = DiscreteLehmannRepresentation(basis, T[1, -1])
        @test SparseIR.get_poles(dlr) == reference
        @test SparseIR.npoles(dlr) == 2
        @test norm(SparseIR.get_poles(dlr)) > 0
    end

    @test_throws ArgumentError DiscreteLehmannRepresentation(basis, ComplexF64[1.0, -1.0])
    @test_throws ArgumentError DiscreteLehmannRepresentation(basis, Float64[])
    @test_throws ArgumentError DiscreteLehmannRepresentation(basis, [1.0, 1.0])
    @test_throws ArgumentError DiscreteLehmannRepresentation(basis, [NaN, 1.0])
    @test_throws ArgumentError DiscreteLehmannRepresentation(basis, [Inf, 1.0])
end

@testitem "audit: from_IR/to_IR element types" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using LinearAlgebra

    basis = FiniteTempBasis(Fermionic(), 10.0, 4.0, 1e-8)
    dlr = DiscreteLehmannRepresentation(basis)

    gl = zeros(Float64, length(basis))
    gl[1] = 1.0

    g_dlr = from_IR(dlr, gl)
    @test eltype(g_dlr) === Float64
    @test norm(g_dlr) > 0
    @test norm(to_IR(dlr, g_dlr)) > 0

    g_dlr_c = from_IR(dlr, ComplexF64.(gl))
    @test eltype(g_dlr_c) === ComplexF64
    @test norm(g_dlr_c) > 0
    @test g_dlr_c ≈ ComplexF64.(g_dlr)

    # Narrower element types used to be reinterpreted as Float64/ComplexF64 at
    # the C boundary (heap corruption / segfault); they must now be rejected.
    @testset "rejected element type $T" for T in (Float32, Float16, ComplexF32, Int64)
        @test_throws ArgumentError from_IR(dlr, ones(T, length(basis)))
        @test_throws ArgumentError to_IR(dlr, ones(T, length(dlr)))
    end

    @test_throws ArgumentError from_IR(dlr, gl, 0)
    @test_throws ArgumentError from_IR(dlr, gl, 2)
    @test_throws ArgumentError to_IR(dlr, g_dlr, 0)
end

@testitem "audit: sampling point validation" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using LinearAlgebra

    β = 10.0
    basis = FiniteTempBasis(Fermionic(), β, 4.0, 1e-8)

    # Finding B: element types are validated/converted before any ccall.
    @testset "tau element type $T" for T in (Float64, Float32, Float16, Int64, Int32)
        smpl = TauSampling(basis; sampling_points=T[1, 2, 3])
        @test sampling_points(smpl) == [1.0, 2.0, 3.0]
        @test eltype(sampling_points(smpl)) === Float64
    end
    @test_throws ArgumentError TauSampling(basis; sampling_points=ComplexF64[1.0, 2.0])

    # Finding H: duplicated points make the sampling matrix rank deficient.
    @test_throws ArgumentError TauSampling(basis; sampling_points=[1.0, 2.0, 1.0])
    @test_throws ArgumentError MatsubaraSampling(basis; sampling_points=[1, 3, 1])

    @test_throws ArgumentError TauSampling(basis; sampling_points=Float64[])
    @test_throws ArgumentError MatsubaraSampling(basis; sampling_points=Int[])

    # Finding F: non-finite input must raise instead of reaching the backend.
    @test_throws ArgumentError TauSampling(basis; sampling_points=[NaN, 1.0])
    @test_throws ArgumentError TauSampling(basis; sampling_points=[Inf, 1.0])

    @test_throws DomainError TauSampling(basis; sampling_points=[2β])
    @test_throws DomainError TauSampling(basis; sampling_points=[-2β])

    # Sanity: the validated path still round-trips and is not all zeros.
    smpl = TauSampling(basis)
    gl = zeros(length(basis))
    gl[1] = 1.0
    gτ = evaluate(smpl, gl)
    @test norm(gτ) > 0
    @test isapprox(fit(smpl, gτ), gl; atol=1e-9)
end

@testitem "audit: status codes are checked exhaustively" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR

    basis = FiniteTempBasis(Fermionic(), 10.0, 4.0, 1e-8)
    smpl = TauSampling(basis)

    # Finding J: every documented status code has a name, and unknown codes are
    # reported verbatim rather than swallowed.
    for code in (SparseIR.C_API.SPIR_GET_IMPL_FAILED,
        SparseIR.C_API.SPIR_NOT_SUPPORTED,
        SparseIR.C_API.SPIR_INVALID_ARGUMENT,
        SparseIR.C_API.SPIR_INTERNAL_ERROR)
        err = try
            SparseIR._check_status(code, "test_op")
            nothing
        catch e
            e
        end
        @test err isa SparseIR.SparseIRError
        @test err.status == Int32(code)
        msg = sprint(showerror, err)
        @test occursin("test_op", msg)
        @test occursin(string(Int32(code)), msg)
    end
    @test SparseIR._check_status(SparseIR.C_API.SPIR_COMPUTATION_SUCCESS, "ok") === nothing
    @test_throws DimensionMismatch SparseIR._check_status(
        SparseIR.C_API.SPIR_INPUT_DIMENSION_MISMATCH, "test_op")
    # An unrecognized code must still raise.
    unknown = try
        SparseIR._check_status(Int32(-999), "test_op")
        nothing
    catch e
        e
    end
    @test unknown isa SparseIR.SparseIRError
    @test occursin("unrecognized", sprint(showerror, unknown))

    # Finding I: the unsupported-type branches throw instead of falling through
    # to an UndefVarError on the status variable.
    out = Vector{Float32}(undef, length(sampling_points(smpl)))
    @test_throws ArgumentError evaluate!(out, smpl, zeros(length(basis)))
    @test_throws ArgumentError evaluate(smpl, zeros(Float32, length(basis)))
    @test_throws ArgumentError fit(smpl, zeros(Float32, length(sampling_points(smpl))))
end

@testitem "audit: rescale" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR

    β = 10.0
    ωmax = 4.0
    basis = FiniteTempBasis(Fermionic(), β, ωmax, 1e-8)

    # Finding C: `rescale` used to throw MethodError unconditionally.
    rescaled = SparseIR.rescale(basis, 2β)
    @test rescaled isa FiniteTempBasis
    @test SparseIR.β(rescaled) == 2β
    @test SparseIR.Λ(rescaled) ≈ SparseIR.Λ(basis)
    @test SparseIR.ωmax(rescaled) ≈ ωmax / 2
    @test length(rescaled) == length(basis)
    @test SparseIR.statistics(rescaled) == SparseIR.statistics(basis)

    @test_throws DomainError SparseIR.rescale(basis, 0.0)
    @test_throws DomainError SparseIR.rescale(basis, -1.0)
end

@testitem "audit: FiniteTempBasisSet.sve_result" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using LinearAlgebra

    bset = FiniteTempBasisSet(10.0, 4.0, 1e-8)

    # Finding D: this used to raise UndefVarError.
    sve = bset.sve_result
    @test sve isa SparseIR.SVEResult
    @test sve === bset.basis_f.sve_result
    @test sve === bset.basis_b.sve_result
    @test norm(sve.s) > 0
    @test norm(bset.tau) > 0
    @test !isempty(bset.wn_f)
end

@testitem "audit: deriv, xmin/xmax, overlap(return_error=true)" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using LinearAlgebra

    β = 10.0
    ωmax = 4.0
    basis = FiniteTempBasis(Fermionic(), β, ωmax, 1e-8)

    # Finding E: `deriv` had no method for the polynomial types at all.
    du = SparseIR.deriv(basis.u)
    @test du isa SparseIR.PiecewiseLegendrePolyVector
    @test length(du(1.0)) == length(basis)
    @test norm(du(1.0)) > 0
    @test SparseIR.deriv(basis.u, 0)(1.0) ≈ basis.u(1.0)
    @test SparseIR.deriv(basis.u, Val(1))(1.0) ≈ du(1.0)
    @test SparseIR.deriv(basis.u, 2)(1.0) isa Vector{Float64}

    d1 = SparseIR.deriv(basis.u[1])
    @test d1 isa SparseIR.PiecewiseLegendrePoly
    @test d1(1.0) ≈ du(1.0)[1]

    @test_throws DomainError SparseIR.deriv(basis.u, -1)

    # A finite-difference cross-check of the first derivative.
    h = 1e-5
    fd = (basis.u(1.0 + h) .- basis.u(1.0 - h)) ./ (2h)
    @test isapprox(du(1.0), fd; rtol=1e-4)

    # `xmin`/`xmax` exist for the polynomial types...
    @test SparseIR.xmin(basis.u) ≈ -β
    @test SparseIR.xmax(basis.u) ≈ β
    @test SparseIR.xmin(basis.u[1]) ≈ -β
    @test SparseIR.xmax(basis.v) ≈ ωmax
    # ...and must not exist for the Matsubara-axis type, which has no such field.
    @test !hasmethod(SparseIR.xmin, Tuple{SparseIR.PiecewiseLegendreFTVector})
    @test !hasmethod(SparseIR.xmax, Tuple{SparseIR.PiecewiseLegendreFTVector})

    # `overlap(..., return_error=true)` on a vector of polys used to hit an
    # undefined `size(::Tuple)`.
    vals = overlap(basis.u, x -> 1.0)
    @test length(vals) == length(basis)
    @test norm(vals) > 0
    vals2, errs = overlap(basis.u, x -> 1.0; return_error=true)
    @test vals2 ≈ vals
    @test length(errs) == length(basis)
    @test all(≥(0), errs)

    v1, e1 = overlap(basis.u[1], x -> 1.0; return_error=true)
    @test v1 ≈ vals[1]
    @test e1 ≈ errs[1]

    # A vector-valued integrand must keep its matrix shape in both modes.
    N = 4
    gram = overlap(basis.u[1:N], basis.u[1:N], 0.0, β)
    @test size(gram) == (N, N)
    gram2, gram_err = overlap(basis.u[1:N], basis.u[1:N], 0.0, β; return_error=true)
    @test size(gram2) == (N, N)
    # quadgk reports one scalar error estimate per integral.
    @test size(gram_err) == (N,)
    @test gram2 ≈ gram
    @test isapprox(gram, Matrix{Float64}(LinearAlgebra.I, N, N); atol=1e-12)
end

@testitem "audit: augmented basis deriv and sampling boundary" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using LinearAlgebra

    β = 1000.0
    ωmax = 2.0
    basis = FiniteTempBasis{Bosonic}(β, ωmax, 1e-6)
    aug = AugmentedBasis(basis, TauConst, TauLinear)

    # Finding E: `deriv` broadcast over a non-iterable poly vector.
    d = SparseIR.deriv(aug.u)
    @test length(d(1.0)) == length(aug)
    @test norm(d(1.0)) > 0
    @test SparseIR.xmin(aug.u) ≈ -β
    @test SparseIR.xmax(aug.u) ≈ β

    # Findings F/H on the `*_new_with_matrix` boundary.
    @test_throws ArgumentError TauSampling(aug; sampling_points=[1.0, 1.0])
    @test_throws ArgumentError TauSampling(aug; sampling_points=[NaN, 1.0])
    @test_throws ArgumentError TauSampling(aug; sampling_points=Float64[])
    @test_throws ArgumentError MatsubaraSampling(aug; sampling_points=[2, 2])
    @test_throws ArgumentError MatsubaraSampling(aug; sampling_points=Int[])

    # The validated path still works and is not all zeros.
    τ_smpl = TauSampling(aug)
    @test eltype(sampling_points(τ_smpl)) === Float64
    gl = zeros(length(aug))
    gl[3] = 1.0
    gτ = evaluate(τ_smpl, gl)
    @test norm(gτ) > 0
    @test isapprox(fit(τ_smpl, gτ), gl; atol=1e-8)

    m_smpl = MatsubaraSampling(AugmentedBasis(basis, MatsubaraConst))
    @test !isempty(sampling_points(m_smpl))
end

@testitem "audit: positive_only contract and real output" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using LinearAlgebra
    using StableRNGs

    rng = StableRNG(1234)
    basis = FiniteTempBasis(Fermionic(), 10.0, 4.0, 1e-8)

    # Finding G: `positive_only = true` is an unchecked contract. For data that
    # honors it the round trip is exact and the coefficients are exactly real.
    smpl_pos = MatsubaraSampling(basis; positive_only=true)
    gl = randn(rng, length(basis))
    giν = evaluate(smpl_pos, gl)
    fitted = fit(smpl_pos, giν)
    @test isapprox(fitted, gl; atol=1e-8)
    @test all(iszero, imag(fitted))

    # For data that violates it the real system is exactly determined, so the
    # residual vanishes and the wrong answer cannot be detected here. This test
    # pins that documented behavior so a future checked implementation has to
    # update the docstring together with the code.
    violating = giν .+ 0.5im .* (1:length(giν))
    bad = fit(smpl_pos, violating)
    @test all(iszero, imag(bad))
    @test norm(evaluate(smpl_pos, bad) .- violating) < 1e-8 * norm(violating)

    # Requesting a real output array for genuinely complex coefficients must
    # raise rather than silently drop the imaginary part.
    smpl = MatsubaraSampling(basis)
    gl_c = randn(rng, length(basis)) .+ 1im .* randn(rng, length(basis))
    giν_c = evaluate(smpl, gl_c)
    out = Vector{Float64}(undef, length(basis))
    @test_throws ArgumentError fit!(out, smpl, giν_c)

    # ...while a real quantity still fits into a real array.
    out2 = Vector{Float64}(undef, length(basis))
    fit!(out2, smpl, ComplexF64.(evaluate(smpl, gl)))
    @test isapprox(out2, gl; atol=1e-8)
end

@testitem "audit: no exported symbol fails with MethodError/UndefVarError" tags=[
    :julia, :sparseir] begin
    using Test
    using SparseIR

    # Every exported name must resolve to a defined object.
    for name in names(SparseIR)
        name === :SparseIR && continue
        @test isdefined(SparseIR, name)
        @test getproperty(SparseIR, name) !== nothing
    end
end
