using Test
using LinearAlgebra
using SparseIR
using SparseIR._LinAlg
using MultiFloats

@testset "_linalg.jl" begin
    @testset "jacobi with T = $T" for T in (Float64, Float64x2)
        A = T.(randn(20, 10))
        U, S, V = svd_jacobi(A)
        @test U * Diagonal(S) * V' ≈ A
    end

    @testset "rrqr with T = $T" for T in (Float64, Float64x2)
        A = T.(randn(40, 30))
        A_eps = norm(A) * eps(eltype(A))
        A_qr, A_rank = rrqr(A)
        A_rec = A_qr.Q * A_qr.R * A_qr.P'
        @test isapprox(A_rec, A; rtol=0, atol=4 * A_eps)
        @test A_rank == 30
    end

    @testset "rrqr_trunc with T = $T" for T in (Float64, Float64x2)
        # Vandermonde matrix
        A = Vector{T}(-1:0.02:1) .^ Vector(0:20)'
        m, n = size(A)
        A_qr, k = SparseIR._LinAlg.rrqr(A; rtol=1e-5)
        @test k < min(m, n)

        Q, R = SparseIR._LinAlg.truncate_qr_result(A_qr, k)
        A_rec = Q * R * A_qr.P'
        @test isapprox(A, A_rec, rtol=0, atol=1e-5 * norm(A))
    end

    @testset "tsvd with T = $T" for T in (Float64, Float64x2), tol in (1e-14, 1e-13)
        A = Vector{T}(-1:0.01:1) .^ Vector(0:50)'
        U, S, V = SparseIR._LinAlg.tsvd(A; rtol=tol)
        k = length(S)

        @test U * Diagonal(S) * V'≈A rtol=0 atol=tol * norm(A)
        @test U'U ≈ I
        @test V'V ≈ I
        @test issorted(S; rev=true)
        @test k < minimum(size(A))

        A_svd = svd(Float64.(A))
        @test S ≈ A_svd.S[1:k]
    end

    @testset "svd of VERY triangular 2x2 with T = $T" for T in (Float64, Float64x2)
        (cu, su), (smax, smin), (cv, sv) = SparseIR._LinAlg.svd2x2(T(1), T(1e100), T(1))
        @test cu ≈ 1.0
        @test su ≈ 1e-100
        @test smax ≈ 1e100
        @test smin ≈ 1e-100
        @test cv ≈ 1e-100
        @test sv ≈ 1.0
        U  = [cu -su
              su  cu]
        S  = [smax  0
              0    smin]
        Vt = [ cv  sv
              -sv  cv]
        A  = [T(1)  T(1e100)
              T(0)    T(1)]
        @test U * S * Vt ≈ A

        (cu, su), (smax, smin), (cv, sv) = SparseIR._LinAlg.svd2x2(T(1), T(1e100), T(1e100))
        @test cu ≈ 1/√2
        @test su ≈ 1/√2
        @test smax ≈ √2 * 1e100
        @test smin ≈ 1/√2
        @test cv ≈ 5e-101
        @test sv ≈ 1.0
        U  = [cu -su
              su  cu]
        S  = [smax  0
              0    smin]
        Vt = [ cv  sv
              -sv  cv]
        A  = [T(1)  T(1e100)
              T(0)  T(1e100)]
        @test U * S * Vt ≈ A

        (cu, su), (smax, smin), (cv, sv) = SparseIR._LinAlg.svd2x2(T(1e100), T(1e200), T(2))
        @test cu ≈ 1.0
        @test su ≈ 2e-200
        @test smax ≈ 1e200
        @test smin ≈ 2e-100
        @test cv ≈ 1e-100
        @test sv ≈ 1.0
        U  = [cu -su
              su  cu]
        S  = [smax  0
              0    smin]
        Vt = [ cv  sv
              -sv  cv]
        A  = [T(1e100)  T(1e200)
              T(0)  T(2)]
        @test U * S * Vt ≈ A

        (cu, su), (smax, smin), (cv, sv) = SparseIR._LinAlg.svd2x2(T(1e-100), T(1), T(1e-100))
        @test cu ≈ 1.0
        @test su ≈ 1e-100
        @test smax ≈ 1.0
        @test smin ≈ 1e-200
        @test cv ≈ 1e-100
        @test sv ≈ 1.0
        U  = [cu -su
              su  cu]
        S  = [smax  0
              0    smin]
        Vt = [ cv  sv
              -sv  cv]
        A  = [T(1e-100)    T(1)
              T(0)       T(1e-100)]
        @test U * S * Vt ≈ A
    end

    @testset "svd of 'more lower' 2x2 with T = $T" for T in (Float64, Float64x2)
        (cu, su), (smax, smin), (cv, sv) = SparseIR._LinAlg.svd2x2(T(1), T(1e-100), T(1e100), T(1))
        @test cu ≈ 1e-100
        @test su ≈ 1.0
        @test smax ≈ 1e100
        @test abs(smin) < 1e-100 # should be ≈ 0.0, but x ≈ 0 is equivalent to x == 0
        @test cv ≈ 1.0
        @test sv ≈ 1e-100
        U  = [cu -su
              su  cu]
        S  = [smax  0
              0    smin]
        Vt = [ cv  sv
              -sv  cv]
        A  = [T(1)      T(1e-100)
              T(1e100)    T(1)]
        @test U * S * Vt ≈ A
    end

    @testset "givens rotation of 2d vector - special cases with T = $T" for T in (Float64, Float64x2)
        for v in ([42, 0], [-42, 0], [0, 42], [0, -42], [0, 0])
            v = T.(v)
            (c, s), r = SparseIR._LinAlg.givens_params(v...)
            R  = [ c  s
                  -s  c]
            Rv = [  r
                  T(0)]
            @test R * v == Rv
        end
    end
end
