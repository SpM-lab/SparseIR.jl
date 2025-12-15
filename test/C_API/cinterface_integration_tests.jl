@testitem "cinterface_integration_tests.jl" tags=[:cinterface] begin
    # Test file corresponding to test/cpp/cinterface_integration.cxx

    using SparseIR
    using Test
    using Random

    # Helper function corresponding to _get_dims in cinterface_integration.cxx
    function _get_dims(target_dim_size::Integer, extra_dims::Vector{<:Integer},
            target_dim::Integer, ndim::Integer)
        dims = Vector{Int32}(undef, ndim)
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

    function generate_random_coeffs(
            ::Type{<:Real}, random_value_real, random_value_imag, pole)
        (2 * random_value_real - 1.0) * sqrt(abs(pole))
    end

    function generate_random_coeffs(
            ::Type{<:Complex}, random_value_real, random_value_imag, pole)
        real_part = (2 * random_value_real - 1.0) * sqrt(abs(pole))
        imag_part = (2 * random_value_imag - 1.0) * sqrt(abs(pole))
        return complex(real_part, imag_part)
    end

    # Helper function corresponding to _spir_basis_new in cinterface_integration.cxx
    function _spir_basis_new(statistics::Integer, beta::Float64, omega_max::Float64,
            epsilon::Float64, status::Ref{Cint})
        # Create logistic kernel
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * omega_max, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test kernel != C_NULL

        # Create SVE result
        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, NaN, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test sve != C_NULL

        # Create basis
        basis = SparseIR.spir_basis_new(statistics, beta, omega_max, epsilon, kernel, sve, -1, status)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Clean up intermediate objects (like C++ version)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)

        return basis
    end

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

    function dlr_to_IR(dlr, order, ndim, dims, target_dim,
            coeffs::AbstractArray{<:Real}, g_IR::AbstractArray{<:Real})
        backend = C_NULL
        SparseIR.spir_dlr2ir_dd(dlr, backend, order, ndim, dims, target_dim, coeffs, g_IR)
    end

    function dlr_to_IR(dlr, order, ndim, dims, target_dim,
            coeffs::AbstractArray{<:Complex}, g_IR::AbstractArray{<:Complex})
        backend = C_NULL
        SparseIR.spir_dlr2ir_zz(dlr, backend, order, ndim, dims, target_dim, coeffs, g_IR)
    end

    function dlr_from_IR(dlr, order, ndim, dims, target_dim, g_IR::AbstractArray{<:Real},
            g_DLR_reconst::AbstractArray{<:Real})
        SparseIR.spir_ir2dlr_dd(dlr, order, ndim, dims, target_dim, g_IR, g_DLR_reconst)
    end

    function dlr_from_IR(
            dlr, order, ndim, dims, target_dim, g_IR::AbstractArray{<:Complex},
            g_DLR_reconst::AbstractArray{<:Complex})
        SparseIR.spir_ir2dlr_zz(dlr, order, ndim, dims, target_dim, g_IR, g_DLR_reconst)
    end

    function compare_tensors_with_relative_error(a, b, tol)
        diff = abs.(a - b)
        ref = abs.(a)
        max_diff = maximum(diff)
        max_ref = maximum(ref)

        # Debug output like C++ version
        if max_diff > tol * max_ref
            println("max_diff: ", max_diff)
            println("max_ref: ", max_ref)
            println("tol: ", tol)
        end

        return max_diff <= tol * max_ref
    end

    function _transform_coefficients(
            coeffs::AbstractArray{T,N}, basis_eval::AbstractMatrix{U},
            target_dim::Integer) where {T,U,N}
        # Move the target dimension to the first position
        coeffs_targetdim0 = movedim(coeffs, 1 + target_dim, 1)

        # Calculate the size of the extra dimensions
        extra_size = prod(size(coeffs_targetdim0)[2:end])

        # Create result tensor with correct dimensions
        dims = collect(size(coeffs_targetdim0))
        dims[1] = size(basis_eval, 1)

        # Initialize the result
        PromotedType = promote_type(T, U)
        result = Array{PromotedType,N}(undef, dims...)

        # Map tensors to matrices for multiplication
        coeffs_mat = reshape(coeffs_targetdim0, size(coeffs_targetdim0, 1), extra_size)
        result_mat = reshape(result, size(basis_eval, 1), extra_size)

        # Perform the matrix multiplication
        result_mat .= basis_eval * coeffs_mat

        # Move dimension back to original order
        return movedim(result, 1, 1 + target_dim)
    end

    function _evaluate_basis_functions(::Type{T}, u, x_values) where {T}
        status = Ref{Cint}(-100)
        funcs_size_ref = Ref{Cint}(0)
        status[] = SparseIR.spir_funcs_get_size(u, funcs_size_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        funcs_size = funcs_size_ref[]

        u_eval_mat = Matrix{T}(undef, length(x_values), funcs_size)
        for i in eachindex(x_values)
            u_eval = Vector{T}(undef, funcs_size)
            status[] = SparseIR.spir_funcs_eval(u, x_values[i], u_eval)
            @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
            u_eval_mat[i, :] .= u_eval
        end
        return u_eval_mat
    end

    function _evaluate_gtau(coeffs::AbstractArray{T,N}, u, target_dim, x_values) where {T,N}
        u_eval_mat = _evaluate_basis_functions(Float64, u, x_values)
        return _transform_coefficients(coeffs, u_eval_mat, target_dim)
    end

    function _evaluate_matsubara_basis_functions(uhat, matsubara_indices)
        status = Ref{Cint}(-100)
        funcs_size_ref = Ref{Cint}(0)
        status[] = SparseIR.spir_funcs_get_size(uhat, funcs_size_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        funcs_size = funcs_size_ref[]

        uhat_eval_mat = Matrix{ComplexF64}(undef, length(matsubara_indices), funcs_size)
        freq_indices = Int64.(matsubara_indices)
        order = SparseIR.SPIR_ORDER_COLUMN_MAJOR
        status[] = SparseIR.spir_funcs_batch_eval_matsu(
            uhat, order, length(matsubara_indices), freq_indices, uhat_eval_mat)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        return uhat_eval_mat
    end

    function _evaluate_giw(coeffs, uhat, target_dim, matsubara_indices)
        uhat_eval_mat = _evaluate_matsubara_basis_functions(uhat, matsubara_indices)
        result = _transform_coefficients(coeffs, uhat_eval_mat, target_dim)
        return result
    end

    function _tau_sampling_evaluate(sampling, order, ndim, dims, target_dim,
            gIR::AbstractArray{<:Real}, gtau::AbstractArray{<:Real})
        status = SparseIR.spir_sampling_eval_dd(
            sampling, order, ndim, dims, target_dim, gIR, gtau)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        return status
    end

    function _tau_sampling_evaluate(sampling, order, ndim, dims, target_dim,
            gIR::AbstractArray{<:Complex}, gtau::AbstractArray{<:Complex})
        status = SparseIR.spir_sampling_eval_zz(
            sampling, order, ndim, dims, target_dim, gIR, gtau)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        return status
    end

    function _tau_sampling_fit(sampling, order, ndim, dims, target_dim,
            gtau::AbstractArray{<:Real}, gIR::AbstractArray{<:Real})
        backend = C_NULL
        status = SparseIR.spir_sampling_fit_dd(
            sampling, backend, order, ndim, dims, target_dim, gtau, gIR)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        return status
    end

    function _tau_sampling_fit(sampling, order, ndim, dims, target_dim,
            gtau::AbstractArray{<:Complex}, gIR::AbstractArray{<:Complex})
        status = SparseIR.spir_sampling_fit_zz(
            sampling, order, ndim, dims, target_dim, gtau, gIR)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        return status
    end

    function _matsubara_sampling_evaluate(sampling, order, ndim, dims, target_dim,
            gIR::AbstractArray{<:Real}, giw::AbstractArray{<:Complex})
        status = SparseIR.spir_sampling_eval_dz(
            sampling, order, ndim, dims, target_dim, gIR, giw)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        return status
    end

    function _matsubara_sampling_evaluate(sampling, order, ndim, dims, target_dim,
            gIR::AbstractArray{<:Complex}, giw::AbstractArray{<:Complex})
        status = SparseIR.spir_sampling_eval_zz(
            sampling, order, ndim, dims, target_dim, gIR, giw)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        return status
    end

    # Main integration test function
    function integration_test(::Type{T}, beta, wmax, epsilon, extra_dims,
            target_dim, order, tol, positive_only) where {T}
        # positive_only is not supported for complex numbers
        @test !(T <: Complex && positive_only)

        ndim = 1 + length(extra_dims)
        @test ndim == 1 + length(extra_dims)

        # Verify that the order parameter is consistent
        if order == SparseIR.SPIR_ORDER_COLUMN_MAJOR
            # Column major order
        else
            # Row major order
        end

        stat = SparseIR.SPIR_STATISTICS_BOSONIC
        status = Ref{Cint}(-100)

        # IR basis
        basis = _spir_basis_new(stat, beta, wmax, epsilon, status)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        basis_size_ref = Ref{Cint}(-100)
        status[] = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        # Tau Sampling
        println("Tau sampling")
        num_tau_points_ref = Ref{Cint}(-100)
        status[] = SparseIR.spir_basis_get_n_default_taus(basis, num_tau_points_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        num_tau_points = num_tau_points_ref[]
        @test num_tau_points > 0

        tau_points_org = Vector{Float64}(undef, num_tau_points)
        status[] = SparseIR.spir_basis_get_default_taus(basis, tau_points_org)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        tau_sampling_status = Ref{Cint}(-100)
        tau_sampling = SparseIR.spir_tau_sampling_new(
            basis, num_tau_points, tau_points_org, tau_sampling_status)
        @test tau_sampling_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test tau_sampling != C_NULL

        status[] = SparseIR.spir_sampling_get_npoints(tau_sampling, num_tau_points_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        num_tau_points = num_tau_points_ref[]
        tau_points = Vector{Float64}(undef, num_tau_points)
        status[] = SparseIR.spir_sampling_get_taus(tau_sampling, tau_points)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test num_tau_points >= basis_size

        # Compare tau_points and tau_points_org
        for i in 1:num_tau_points
            @test tau_points[i] ≈ tau_points_org[i]
        end

        # Matsubara Sampling
        println("Matsubara sampling")
        num_matsubara_points_org_ref = Ref{Cint}(0)
        status[] = SparseIR.spir_basis_get_n_default_matsus(
            basis, positive_only, num_matsubara_points_org_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        num_matsubara_points_org = num_matsubara_points_org_ref[]
        @test num_matsubara_points_org > 0

        matsubara_points_org = Vector{Int64}(undef, num_matsubara_points_org)
        status[] = SparseIR.spir_basis_get_default_matsus(
            basis, positive_only, matsubara_points_org)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        matsubara_sampling_status = Ref{Cint}(-100)
        matsubara_sampling = SparseIR.spir_matsu_sampling_new(
            basis, positive_only, num_matsubara_points_org,
            matsubara_points_org, matsubara_sampling_status)
        @test matsubara_sampling_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test matsubara_sampling != C_NULL

        if positive_only
            @test num_matsubara_points_org >= basis_size ÷ 2
        else
            @test num_matsubara_points_org >= basis_size
        end

        num_matsubara_points_ref = Ref{Cint}(0)
        status[] = SparseIR.spir_sampling_get_npoints(
            matsubara_sampling, num_matsubara_points_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        num_matsubara_points = num_matsubara_points_ref[]
        matsubara_points = Vector{Int64}(undef, num_matsubara_points)
        status[] = SparseIR.spir_sampling_get_matsus(matsubara_sampling, matsubara_points)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Compare matsubara_points and matsubara_points_org
        for i in 1:num_matsubara_points
            @test matsubara_points[i] == matsubara_points_org[i]
        end

        # DLR
        println("DLR")
        dlr_status = Ref{Cint}(-100)
        dlr = SparseIR.spir_dlr_new(basis, dlr_status)
        @test dlr_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test dlr != C_NULL

        npoles_ref = Ref{Cint}(-100)
        status[] = SparseIR.spir_dlr_get_npoles(dlr, npoles_ref)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        npoles = npoles_ref[]
        @test npoles >= basis_size

        poles = Vector{Float64}(undef, npoles)
        status[] = SparseIR.spir_dlr_get_poles(dlr, poles)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Calculate total size of extra dimensions
        extra_size = prod(extra_dims)

        # Generate random DLR coefficients
        coeffs_targetdim0 = Array{T,ndim}(undef, npoles, extra_dims...)

        coeffs_2d = reshape(coeffs_targetdim0, Int64(npoles), Int64(extra_size))
        Random.seed!(982743)  # Same seed as C++ version
        for i in 1:npoles
            for j in 1:extra_size
                coeffs_2d[i, j] = generate_random_coeffs(T, rand(), rand(), poles[i])
            end
        end
        #coeffs_targetdim0 .= 0.0
        #coeffs_targetdim0[npoles ÷ 2] = 1.0
        #coeffs_targetdim0[npoles ÷ 2 + 1] = 1.0

        # DLR sampling objects (MISSING in original Julia code)
        tau_sampling_dlr_status = Ref{Cint}(-100)
        tau_sampling_dlr = SparseIR.spir_tau_sampling_new(
            dlr, num_tau_points, tau_points_org, tau_sampling_dlr_status)
        @test tau_sampling_dlr_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test tau_sampling_dlr != C_NULL

        dlr_uhat2_status = Ref{Cint}(-100)
        dlr_uhat2 = SparseIR.spir_basis_get_uhat(dlr, dlr_uhat2_status)
        @test dlr_uhat2_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test dlr_uhat2 != C_NULL
        @test SparseIR.spir_funcs_is_assigned(dlr_uhat2) == 1

        matsubara_sampling_dlr_status = Ref{Cint}(-100)
        matsubara_sampling_dlr = SparseIR.spir_matsu_sampling_new(
            dlr, positive_only, num_matsubara_points_org,
            matsubara_points_org, matsubara_sampling_dlr_status)
        @test matsubara_sampling_dlr_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test matsubara_sampling_dlr != C_NULL

        # Move the axis for the poles from the first to the target dimension
        coeffs = movedim(coeffs_targetdim0, 1, 1 + target_dim)

        # Convert DLR coefficients to IR coefficients
        g_IR = Array{T,ndim}(undef, _get_dims(basis_size, extra_dims, target_dim, ndim)...)
        status[] = dlr_to_IR(
            dlr, order, ndim, _get_dims(npoles, extra_dims, target_dim, ndim),
            target_dim, coeffs, g_IR)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Convert IR coefficients back to DLR coefficients (this should reconstruct the original coeffs)
        # Note: C++ version has a bug here - it creates g_DLR_reconst with basis_size but calls with npoles dims
        # We follow the C++ version exactly to match behavior
        g_DLR_reconst = Array{T,ndim}(
            undef, _get_dims(basis_size, extra_dims, target_dim, ndim)...)
        status[] = dlr_from_IR(
            dlr, order, ndim, _get_dims(npoles, extra_dims, target_dim, ndim),
            target_dim, g_IR, g_DLR_reconst)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # From IR to DLR
        g_dlr = Array{T,ndim}(undef, _get_dims(basis_size, extra_dims, target_dim, ndim)...)
        status[] = dlr_from_IR(
            dlr, order, ndim, _get_dims(basis_size, extra_dims, target_dim, ndim),
            target_dim, g_IR, g_dlr)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # DLR basis functions
        dlr_u_status = Ref{Cint}(-100)
        dlr_u = SparseIR.spir_basis_get_u(dlr, dlr_u_status)
        @test dlr_u_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test dlr_u != C_NULL

        dlr_uhat_status = Ref{Cint}(-100)
        dlr_uhat = SparseIR.spir_basis_get_uhat(dlr, dlr_uhat_status)
        @test dlr_uhat_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test dlr_uhat != C_NULL

        # IR basis functions
        ir_u_status = Ref{Cint}(-100)
        ir_u = SparseIR.spir_basis_get_u(basis, ir_u_status)
        @test ir_u_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test ir_u != C_NULL

        ir_uhat_status = Ref{Cint}(-100)
        ir_uhat = SparseIR.spir_basis_get_uhat(basis, ir_uhat_status)
        @test ir_uhat_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test ir_uhat != C_NULL

        # Compare the Greens function at all tau points between IR and DLR
        gtau_from_IR = _evaluate_gtau(g_IR, ir_u, target_dim, tau_points)
        gtau_from_DLR = _evaluate_gtau(coeffs, dlr_u, target_dim, tau_points)
        gtau_from_DLR_reconst = _evaluate_gtau(g_DLR_reconst, dlr_u, target_dim, tau_points)
        @test compare_tensors_with_relative_error(gtau_from_IR, gtau_from_DLR, tol)
        @test compare_tensors_with_relative_error(gtau_from_IR, gtau_from_DLR_reconst, tol)

        # Use sampling to evaluate the Greens function at all tau points between IR and DLR (MISSING in original Julia code)
        gtau_from_DLR_sampling = similar(gtau_from_DLR)
        if T <: Real
            status[] = SparseIR.spir_sampling_eval_dd(
                tau_sampling_dlr, order, ndim,
                _get_dims(npoles, extra_dims, target_dim, ndim),
                target_dim, coeffs, gtau_from_DLR_sampling)
        elseif T <: Complex
            status[] = SparseIR.spir_sampling_eval_zz(
                tau_sampling_dlr, order, ndim,
                _get_dims(npoles, extra_dims, target_dim, ndim),
                target_dim, coeffs, gtau_from_DLR_sampling)
        end
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test compare_tensors_with_relative_error(gtau_from_IR, gtau_from_DLR_sampling, tol)

        # Compare the Greens function at all Matsubara frequencies between IR and DLR
        giw_from_IR = _evaluate_giw(g_IR, ir_uhat, target_dim, matsubara_points)
        giw_from_DLR = _evaluate_giw(coeffs, dlr_uhat, target_dim, matsubara_points)
        @test compare_tensors_with_relative_error(giw_from_IR, giw_from_DLR, tol)

        # Use sampling to evaluate the Greens function at all Matsubara frequencies between IR and DLR (MISSING in original Julia code)
        giw_from_DLR_sampling = similar(giw_from_DLR)
        if T <: Real
            status[] = SparseIR.spir_sampling_eval_dz(
                matsubara_sampling_dlr, order, ndim,
                _get_dims(npoles, extra_dims, target_dim, ndim),
                target_dim, coeffs, giw_from_DLR_sampling)
        elseif T <: Complex
            status[] = SparseIR.spir_sampling_eval_zz(
                matsubara_sampling_dlr, order, ndim,
                _get_dims(npoles, extra_dims, target_dim, ndim),
                target_dim, coeffs, giw_from_DLR_sampling)
        end
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test compare_tensors_with_relative_error(giw_from_IR, giw_from_DLR_sampling, tol)

        dims_matsubara = _get_dims(num_matsubara_points, extra_dims, target_dim, ndim)
        dims_IR = _get_dims(basis_size, extra_dims, target_dim, ndim)
        dims_tau = _get_dims(num_tau_points, extra_dims, target_dim, ndim)

        gIR = Array{T,ndim}(undef, _get_dims(basis_size, extra_dims, target_dim, ndim)...)
        gIR2 = Array{T,ndim}(undef, _get_dims(basis_size, extra_dims, target_dim, ndim)...)
        gtau = Array{T,ndim}(
            undef, _get_dims(num_tau_points, extra_dims, target_dim, ndim)...)
        giw_reconst = Array{ComplexF64,ndim}(
            undef, _get_dims(num_matsubara_points, extra_dims, target_dim, ndim)...)

        # Matsubara -> IR
        begin
            gIR_work = Array{ComplexF64,ndim}(
                undef, _get_dims(basis_size, extra_dims, target_dim, ndim)...)
            status[] = SparseIR.spir_sampling_fit_zz(
                matsubara_sampling, order, ndim, dims_matsubara, target_dim, giw_from_DLR, gIR_work
            )
            @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
            if T <: Real
                gIR .= real(gIR_work)
            else
                gIR .= gIR_work
            end
        end

        # IR -> tau
        status[] = _tau_sampling_evaluate(
            tau_sampling, order, ndim, dims_IR, target_dim, gIR, gtau)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # tau -> IR
        status[] = _tau_sampling_fit(
            tau_sampling, order, ndim, dims_tau, target_dim, gtau, gIR2)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # IR -> Matsubara
        status[] = _matsubara_sampling_evaluate(
            matsubara_sampling, order, ndim, dims_IR, target_dim, gIR2, giw_reconst)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Final comparison test (MISSING in original Julia code)
        giw_from_IR_reconst = _evaluate_giw(gIR2, ir_uhat, target_dim, matsubara_points)
        @test compare_tensors_with_relative_error(giw_from_DLR, giw_from_IR_reconst, tol)

        # Memory cleanup (MISSING in original Julia code)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_basis_release(dlr)
        SparseIR.spir_funcs_release(dlr_u)
        SparseIR.spir_funcs_release(ir_u)
        SparseIR.spir_sampling_release(tau_sampling)
        SparseIR.spir_sampling_release(tau_sampling_dlr)
        SparseIR.spir_sampling_release(matsubara_sampling)
        SparseIR.spir_sampling_release(matsubara_sampling_dlr)
    end

    # Test parameters
    beta = 1e+4
    wmax = 2.0
    epsilon = 1e-10
    tol = 10 * epsilon

    # Run tests for different configurations like C++ version
    for positive_only in [false, true]
        println("positive_only = ", positive_only)

        # Test 1: Simple 1D case
        begin
            extra_dims = Int[]
            println("Integration test for bosonic LogisticKernel")
            integration_test(Float64, beta, wmax, epsilon, extra_dims, 0,
                SparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)

            if !positive_only
                integration_test(ComplexF64, beta, wmax, epsilon, extra_dims, 0,
                    SparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)
            end
        end

        # Test 2: ColMajor, target_dim = 0
        begin
            target_dim = 0
            extra_dims = Int[]
            println("Integration test for bosonic LogisticKernel, ColMajor, target_dim = ",
                target_dim)
            integration_test(Float64, beta, wmax, epsilon, extra_dims, target_dim,
                SparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)
            if !positive_only
                integration_test(ComplexF64, beta, wmax, epsilon, extra_dims, target_dim,
                    SparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)
            end
        end

        # Test 3: RowMajor, target_dim = 0
        begin
            target_dim = 0
            extra_dims = Int[]
            println("Integration test for bosonic LogisticKernel, RowMajor, target_dim = ",
                target_dim)
            integration_test(Float64, beta, wmax, epsilon, extra_dims, target_dim,
                SparseIR.SPIR_ORDER_ROW_MAJOR, tol, positive_only)
            if !positive_only
                integration_test(ComplexF64, beta, wmax, epsilon, extra_dims, target_dim,
                    SparseIR.SPIR_ORDER_ROW_MAJOR, tol, positive_only)
            end
        end

        # Test 4: Multi-dimensional cases with extra dims = [2,3,4]
        for target_dim in 0:3
            extra_dims = [2, 3, 4]
            println("Integration test for bosonic LogisticKernel, ColMajor, target_dim = ",
                target_dim)
            integration_test(Float64, beta, wmax, epsilon, extra_dims, target_dim,
                SparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)
        end
    end
end
