@testitem "Integration Test" tags=[:julia, :spir] begin
    using SparseIR
    using LinearAlgebra
    using Random

    # Helper function to get dimensions array
    function _get_dims(target_dim_size::Int, extra_dims::Vector{Int}, target_dim::Int)
        ndim = length(extra_dims) + 1
        dims = zeros(Int, ndim)
        dims[target_dim + 1] = target_dim_size  # Julia is 1-indexed
        pos = 1
        for i in 1:ndim
            if i == target_dim + 1
                continue
            end
            dims[i] = extra_dims[pos]
            pos += 1
        end
        return dims
    end


    # Helper function to compare tensors with relative error
    function compare_tensors_with_relative_error(a::Array{T,N}, b::Array{T,N}, tol) where {T,N}
        diff = abs.(a .- b)
        ref = abs.(a)
        max_diff = maximum(diff)
        max_ref = maximum(ref)

        if max_diff > tol * max_ref
            @info "max_diff: $max_diff"
            @info "max_ref: $max_ref"
            @info "tol: $tol"
            return false
        end
        return true
    end

    # Generate random coefficient based on type
    function generate_random_coeff(::Type{Float64}, random_value_real, random_value_imag, pole)
        return (2.0 * random_value_real - 1.0) * sqrt(abs(pole))
    end

    function generate_random_coeff(::Type{ComplexF64}, random_value_real, random_value_imag, pole)
        return ComplexF64(
            (2.0 * random_value_real - 1.0) * sqrt(abs(pole)),
            (2.0 * random_value_imag - 1.0) * sqrt(abs(pole))
        )
    end

    # Main integration test function
    function integration_test(::Type{T}, ::Type{S}, ::Type{K}, ndim::Int,
                            beta, wmax, epsilon, extra_dims, target_dim, tol, positive_only) where {T,S,K}
        # positive_only is not supported for complex numbers
        @assert !(T <: Complex && positive_only)

        @assert ndim == 1 + length(extra_dims)

        # IR basis
        kernel = K(beta * wmax)
        basis = FiniteTempBasis(S(), beta, wmax, epsilon; kernel)
        basis_size = length(basis)

        # Tau Sampling
        @info "Tau sampling"
        tau_points = SparseIR.default_tau_sampling_points(basis)
        num_tau_points = length(tau_points)
        tau_sampling = TauSampling(basis; sampling_points=tau_points)

        @assert num_tau_points >= basis_size
        @assert tau_sampling.sampling_points ≈ tau_points

        # Matsubara Sampling
        @info "Matsubara sampling"
        matsubara_points = SparseIR.default_matsubara_sampling_points(basis; positive_only=positive_only)
        num_matsubara_points = length(matsubara_points)
        matsubara_sampling = MatsubaraSampling(basis; positive_only=positive_only, sampling_points=matsubara_points)
        if positive_only
            @assert num_matsubara_points >= basis_size ÷ 2
        else
            @assert num_matsubara_points >= basis_size
        end
        @assert Int.(matsubara_sampling.sampling_points) == matsubara_points

        # DLR
        @info "DLR"
        dlr = DiscreteLehmannRepresentation(basis)
        npoles = SparseIR.npoles(dlr)
        poles = SparseIR.get_poles(dlr)
        @assert npoles >= basis_size
        @assert length(poles) == npoles

        # Calculate total size of extra dimensions
        extra_size = prod(extra_dims)

        # Generate random DLR coefficients
        Random.seed!(982743)
        coeffs_targetdim0_dims = _get_dims(npoles, extra_dims, 0)
        coeffs_targetdim0 = Array{T}(undef, coeffs_targetdim0_dims...)

        # Fill with random values
        coeffs_2d = reshape(coeffs_targetdim0, npoles, extra_size)
        for i in 1:npoles
            for j in 1:extra_size
                coeffs_2d[i, j] = generate_random_coeff(T, rand(), rand(), poles[i])
            end
        end

        # DLR sampling objects
        tau_sampling_dlr = TauSampling(dlr; sampling_points=tau_points)
        matsubara_sampling_dlr = MatsubaraSampling(dlr; positive_only=positive_only, sampling_points=matsubara_points)

        # Move the axis for the poles from the first to the target dimension
        perm = collect(1:ndim)
        perm[1], perm[target_dim + 1] = perm[target_dim + 1], perm[1]
        coeffs = permutedims(coeffs_targetdim0, perm)

        # Convert DLR coefficients to IR coefficients
        g_IR = SparseIR.to_IR(dlr, coeffs, target_dim + 1)  # Julia is 1-indexed

        # Convert IR coefficients back to DLR coefficients
        g_DLR_reconst = SparseIR.from_IR(dlr, g_IR, target_dim + 1)

        # Compare the Greens function at all tau points between IR and DLR
        # Instead of using basis functions directly, we'll use the sampling objects
        # to evaluate at tau points

        # Evaluate g_IR at tau points using tau_sampling
        gtau_from_IR_dims = collect(size(g_IR))
        gtau_from_IR_dims[target_dim + 1] = num_tau_points
        gtau_from_IR = similar(g_IR, T, gtau_from_IR_dims...)
        evaluate!(gtau_from_IR, tau_sampling, g_IR; dim=target_dim + 1)

        # Evaluate DLR coefficients at tau points using tau_sampling_dlr
        gtau_from_DLR_dims = collect(size(coeffs))
        gtau_from_DLR_dims[target_dim + 1] = num_tau_points
        gtau_from_DLR = similar(coeffs, T, gtau_from_DLR_dims...)
        evaluate!(gtau_from_DLR, tau_sampling_dlr, coeffs; dim=target_dim + 1)

        # Evaluate reconstructed DLR at tau points
        gtau_from_DLR_reconst_dims = collect(size(g_DLR_reconst))
        gtau_from_DLR_reconst_dims[target_dim + 1] = num_tau_points
        gtau_from_DLR_reconst = similar(g_DLR_reconst, T, gtau_from_DLR_reconst_dims...)
        evaluate!(gtau_from_DLR_reconst, tau_sampling_dlr, g_DLR_reconst; dim=target_dim + 1)

        @test compare_tensors_with_relative_error(gtau_from_IR, gtau_from_DLR, tol)
        @test compare_tensors_with_relative_error(gtau_from_IR, gtau_from_DLR_reconst, tol)

        # Use sampling to evaluate the Greens function at all tau points between IR and DLR
        gtau_from_DLR_sampling = similar(gtau_from_DLR)
        evaluate!(gtau_from_DLR_sampling, tau_sampling_dlr, coeffs; dim=target_dim + 1)
        @test compare_tensors_with_relative_error(gtau_from_IR, gtau_from_DLR_sampling, tol)

        # Compare the Greens function at all Matsubara frequencies between IR and DLR
        # Use sampling objects to evaluate at Matsubara frequencies

        # Evaluate g_IR at Matsubara frequencies using matsubara_sampling
        giw_from_IR_dims = collect(size(g_IR))
        giw_from_IR_dims[target_dim + 1] = num_matsubara_points
        giw_from_IR = similar(g_IR, ComplexF64, giw_from_IR_dims...)
        evaluate!(giw_from_IR, matsubara_sampling, g_IR; dim=target_dim + 1)

        # Evaluate DLR coefficients at Matsubara frequencies using matsubara_sampling_dlr
        giw_from_DLR_dims = collect(size(coeffs))
        giw_from_DLR_dims[target_dim + 1] = num_matsubara_points
        giw_from_DLR = similar(coeffs, ComplexF64, giw_from_DLR_dims...)
        evaluate!(giw_from_DLR, matsubara_sampling_dlr, coeffs; dim=target_dim + 1)

        @test compare_tensors_with_relative_error(giw_from_IR, giw_from_DLR, tol)

        # Use sampling to evaluate the Greens function at all Matsubara frequencies
        giw_from_DLR_sampling = similar(giw_from_DLR, ComplexF64)
        evaluate!(giw_from_DLR_sampling, matsubara_sampling_dlr, coeffs; dim=target_dim + 1)
        @test compare_tensors_with_relative_error(giw_from_IR, giw_from_DLR_sampling, tol)

        # Prepare arrays for transformations
        # Use the actual dimensions from g_IR to ensure consistency
        gIR_dims = collect(size(g_IR))
        gIR = Array{T}(undef, gIR_dims...)
        gIR2 = Array{T}(undef, gIR_dims...)

        # For gtau, use tau_points along target dimension
        gtau_dims = collect(size(g_IR))
        gtau_dims[target_dim + 1] = num_tau_points
        gtau = Array{T}(undef, gtau_dims...)

        # For giw_reconst, use matsubara_points along target dimension
        giw_reconst_dims = collect(size(g_IR))
        giw_reconst_dims[target_dim + 1] = num_matsubara_points
        giw_reconst = Array{ComplexF64}(undef, giw_reconst_dims...)

        # Matsubara -> IR
        if T <: Real
            gIR_work = Array{ComplexF64}(undef, gIR_dims...)
            fit!(gIR_work, matsubara_sampling, giw_from_DLR; dim=target_dim + 1)
            gIR .= real.(gIR_work)
        else
            fit!(gIR, matsubara_sampling, giw_from_DLR; dim=target_dim + 1)
        end

        # IR -> tau
        evaluate!(gtau, tau_sampling, gIR; dim=target_dim + 1)

        # tau -> IR
        fit!(gIR2, tau_sampling, gtau; dim=target_dim + 1)

        # IR -> Matsubara
        evaluate!(giw_reconst, matsubara_sampling, gIR2; dim=target_dim + 1)

        giw_from_IR_reconst = similar(giw_reconst)
        evaluate!(giw_from_IR_reconst, matsubara_sampling, gIR2; dim=target_dim + 1)
        @test compare_tensors_with_relative_error(giw_from_DLR, giw_from_IR_reconst, tol)

        # Note: Julia uses automatic garbage collection with finalizers for C resource cleanup.
        # Unlike the C_API version, we don't need explicit release calls.
    end

    # Test _get_dims helper function
    @testset "_get_dims" begin
        extra_dims = [2, 3, 4]

        # Test target_dim = 0
        dims = _get_dims(100, extra_dims, 0)
        @test dims == [100, 2, 3, 4]

        # Test target_dim = 1
        dims = _get_dims(100, extra_dims, 1)
        @test dims == [2, 100, 3, 4]

        # Test target_dim = 2
        dims = _get_dims(100, extra_dims, 2)
        @test dims == [2, 3, 100, 4]

        # Test target_dim = 3
        dims = _get_dims(100, extra_dims, 3)
        @test dims == [2, 3, 4, 100]
    end

    # Run integration tests
    beta = 1e+4
    wmax = 2.0
    epsilon = 1e-10
    tol = 10 * epsilon

    @testset "Integration Tests" begin
        for positive_only in [false, true]
            @info "positive_only = $positive_only"

            # 1D tests
            extra_dims = Int[]
            @info "Integration test for bosonic LogisticKernel"
            integration_test(Float64, SparseIR.Bosonic, SparseIR.LogisticKernel, 1,
                           beta, wmax, epsilon, extra_dims, 0, tol, positive_only)

            @info "Integration test for fermionic LogisticKernel"
            integration_test(Float64, SparseIR.Fermionic, SparseIR.LogisticKernel, 1,
                           beta, wmax, epsilon, extra_dims, 0, tol, positive_only)

            if !positive_only
                integration_test(ComplexF64, SparseIR.Bosonic, SparseIR.LogisticKernel, 1,
                               beta, wmax, epsilon, extra_dims, 0, tol, positive_only)

                integration_test(ComplexF64, SparseIR.Fermionic, SparseIR.LogisticKernel, 1,
                               beta, wmax, epsilon, extra_dims, 0, tol, positive_only)
            end

            # 4D tests with extra_dims = [2, 3, 4]
            for target_dim in 0:3
                extra_dims = [2, 3, 4]
                @info "Integration test for bosonic LogisticKernel, target_dim = $target_dim"
                integration_test(Float64, SparseIR.Bosonic, SparseIR.LogisticKernel, 4,
                                beta, wmax, epsilon, extra_dims, target_dim, tol, positive_only)

                # Also test complex for multi-dimensional arrays when positive_only=false
                if !positive_only && target_dim == 0
                    integration_test(ComplexF64, SparseIR.Bosonic, SparseIR.LogisticKernel, 4,
                                    beta, wmax, epsilon, extra_dims, target_dim, tol, positive_only)
                end
            end
        end
    end
end