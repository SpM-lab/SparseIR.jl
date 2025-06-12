@testitem "matsubara" tags=[:above_julia1_11] begin
    using JET

    beta = 2.0
    wmax = 3.0
    rtol = 1e-8
    basis = FiniteTempBasis{Fermionic}(beta, wmax, rtol)

    sampling = MatsubaraSampling(basis)
    wsample = sampling_points(sampling)
    G_matsubara = rand(ComplexF64, length(wsample))
    @test_opt fit(sampling, G_matsubara)

    G_l = fit(sampling, G_matsubara)
    @test_opt fit!(G_l, sampling, G_matsubara)

    workarr = Vector{eltype(G_l)}(
        undef, SparseIR.workarrlength(sampling, G_matsubara))
    @test_opt fit!(G_l, sampling, G_matsubara; workarr)

    @test_opt evaluate(sampling, G_l)

    @test_opt evaluate!(G_matsubara, sampling, G_l)
end

@testitem "imaginary time" tags=[:above_julia1_11] begin
    using JET

    beta = 2.0
    wmax = 3.0
    rtol = 1e-8
    basis = FiniteTempBasis{Fermionic}(beta, wmax, rtol)

    sampling = TauSampling(basis)
    wsample = sampling_points(sampling)
    G_tau = rand(ComplexF64, length(wsample))
    @test_opt fit(sampling, G_tau)

    G_l = fit(sampling, G_tau)
    @test_opt fit!(G_l, sampling, G_tau)

    workarr = Vector{eltype(G_l)}(undef, SparseIR.workarrlength(sampling, G_tau))
    @test_opt fit!(G_l, sampling, G_tau; workarr)

    @test_opt evaluate(sampling, G_l)
    @test_opt evaluate!(G_tau, sampling, G_l)
end
