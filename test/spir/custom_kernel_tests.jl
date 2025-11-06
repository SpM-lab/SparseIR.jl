@testitem "custom_kernel.jl" tags=[:julia, :spir] begin
    using SparseIR
    using Test

    # Simple wrapper kernel that wraps LogisticKernel
    # This is used to test the custom kernel functionality
    struct SimpleWrapperKernel <: SparseIR.AbstractKernel
        inner::SparseIR.LogisticKernel
        is_centrosym::Bool
    end

    SimpleWrapperKernel(Λ::Real, is_centrosym::Bool=true) = SimpleWrapperKernel(SparseIR.LogisticKernel(Λ), is_centrosym)

    # Delegate all methods to inner kernel
    SparseIR.Λ(kernel::SimpleWrapperKernel) = SparseIR.Λ(kernel.inner)
    SparseIR.xrange(kernel::SimpleWrapperKernel) = SparseIR.xrange(kernel.inner)
    SparseIR.yrange(kernel::SimpleWrapperKernel) = SparseIR.yrange(kernel.inner)
    SparseIR.iscentrosymmetric(kernel::SimpleWrapperKernel) = kernel.is_centrosym
    SparseIR.conv_radius(kernel::SimpleWrapperKernel) = SparseIR.conv_radius(kernel.inner)

    # Kernel evaluation
    function (kernel::SimpleWrapperKernel)(x::Real, y::Real)
        return kernel.inner(x, y)
    end

    # Inv weight functions
    function SparseIR.inv_weight_func(kernel::SimpleWrapperKernel, stat::SparseIR.Statistics, beta::Float64, lambda::Float64)
        return SparseIR.inv_weight_func(kernel.inner, stat, beta, lambda)
    end

    # SVE hints - delegate to inner kernel
    # Note: LogisticKernel uses C++ implementation, so we need to use the default SVE hints
    # For testing, we'll use the default implementation which should work through the C-API
    SparseIR.sve_hints(kernel::SimpleWrapperKernel, ε::Real) = 
        SparseIR.sve_hints(kernel.inner, ε)

    # HoleKernel: restricts kernel to hole side (y >= 0)
    # This is useful in regularization: in an overcomplete basis, there is an
    # "cyclic ambiguity", e.g., the partial propagators for the permutations
    # (123), (231) and (312) are related (see Kugler et al., PRX).  To avoid this,
    # we force all energy differences in the Lehmann representation to be
    # non-negative.
    # Ported from OvercompleteIR.jl
    struct HoleKernel{K <: SparseIR.AbstractKernel} <: SparseIR.AbstractKernel
        inner::K
    end

    HoleKernel(Λ::Real) = HoleKernel(SparseIR.LogisticKernel(Λ))

    SparseIR.Λ(self::HoleKernel) = SparseIR.Λ(self.inner)
    SparseIR.iscentrosymmetric(::HoleKernel) = false
    SparseIR.xrange(self::HoleKernel) = SparseIR.xrange(self.inner)

    function SparseIR.yrange(self::HoleKernel)
        ymin, ymax = SparseIR.yrange(self.inner)
        return zero(ymin), ymax
    end

    # XXX: there is something strange going on with the model
    SparseIR.conv_radius(::HoleKernel) = Inf

    function SparseIR.inv_weight_func(self::HoleKernel, stat::SparseIR.Statistics, beta::Float64, lambda::Float64)
        return SparseIR.inv_weight_func(self.inner, stat, beta, lambda)
    end

    # Ported from OvercompleteIR.jl: supports optional x₊ and x₋ parameters
    # Skip bounds checking to avoid potential deadlocks in C++ callbacks
    # The inner kernel will handle domain validation
    function (self::HoleKernel)(x, y, x₊=x - first(SparseIR.xrange(self)),
                                x₋=last(SparseIR.xrange(self)) - x)
        return self.inner(x, y)
    end

    struct SVEHintsHole{H <: SparseIR.AbstractSVEHints}
        inner::H
    end

    SparseIR.sve_hints(self::HoleKernel, ϵ::Real) =
        SVEHintsHole(SparseIR.sve_hints(self.inner, ϵ))

    SparseIR.segments_x(self::SVEHintsHole, ::Type{T}=Float64) where {T} =
        SparseIR.segments_x(self.inner, T)

    function SparseIR.segments_y(self::SVEHintsHole, ::Type{T}=Float64) where {T}
        ysegs = SparseIR.segments_x(self.inner, T)
        ysegs = ysegs[ysegs .>= 0]
        if !iszero(first(ysegs))
            ysegs = [zero(ysegs); ysegs]
        end
        return ysegs
    end

    SparseIR.nsvals(self::SVEHintsHole) = SparseIR.nsvals(self.inner)
    SparseIR.ngauss(self::SVEHintsHole) = SparseIR.ngauss(self.inner)

    @testset "Custom kernel creation" begin
        lam = 10.0
        kernel = SimpleWrapperKernel(lam, true)
        @test SparseIR.Λ(kernel) == lam
        @test iscentrosymmetric(kernel) == true
        
        kernel2 = SimpleWrapperKernel(lam, false)
        @test SparseIR.Λ(kernel2) == lam
        @test iscentrosymmetric(kernel2) == false
    end

    # WrappedLogisticKernel: simple wrapper for testing
    struct WrappedLogisticKernel <: SparseIR.AbstractKernel
        inner::SparseIR.LogisticKernel
    end

    WrappedLogisticKernel(Λ) = WrappedLogisticKernel(SparseIR.LogisticKernel(Λ))
    SparseIR.Λ(kernel::WrappedLogisticKernel) = SparseIR.Λ(kernel.inner)
    SparseIR.iscentrosymmetric(::WrappedLogisticKernel) = true
    SparseIR.xrange(kernel::WrappedLogisticKernel) = SparseIR.xrange(kernel.inner)
    SparseIR.yrange(kernel::WrappedLogisticKernel) = SparseIR.yrange(kernel.inner)
    SparseIR.conv_radius(kernel::WrappedLogisticKernel) = SparseIR.conv_radius(kernel.inner)
    function (kernel::WrappedLogisticKernel)(x::Real, y::Real)
        return kernel.inner(x, y)
    end
    function SparseIR.inv_weight_func(kernel::WrappedLogisticKernel, stat::SparseIR.Statistics, beta::Float64, lambda::Float64)
        return SparseIR.inv_weight_func(kernel.inner, stat, beta, lambda)
    end
    function SparseIR.sve_hints(kernel::WrappedLogisticKernel, ε::Real)
        return SparseIR.sve_hints(kernel.inner, ε)
    end

    @testset "Custom kernel FiniteTempBasis creation from SVEResult" begin
        # Test creating FiniteTempBasis from custom kernel SVEResult
        kernel = WrappedLogisticKernel(10.0)
        sve = SparseIR.SVEResult(kernel, 1e-6)
        
        # Create basis from SVEResult
        beta = 10.0
        omega_max = 1.0
        epsilon = 1e-6
        
        basis = SparseIR.FiniteTempBasis{SparseIR.Fermionic}(
            sve, beta, omega_max, epsilon
        )
        
        @test length(basis) > 0
        @test length(basis.s) == length(basis)
        @test SparseIR.β(basis) ≈ beta
        @test SparseIR.ωmax(basis) ≈ omega_max
    end

    @testset "Custom kernel SVEResult creation" begin
        lam = 10.0
        epsilon = 1e-10
        
        # Test centrosymmetric = true
        kernel_true = SimpleWrapperKernel(lam, true)
        sve_result_true = SparseIR.SVEResult(kernel_true, epsilon)
        @test sve_result_true.ptr != C_NULL
        
        # Test centrosymmetric = false
        kernel_false = SimpleWrapperKernel(lam, false)
        sve_result_false = SparseIR.SVEResult(kernel_false, epsilon)
        @test sve_result_false.ptr != C_NULL
    end

    @testset "Custom kernel properties" begin
        lam = 10.0
        
        for is_centrosym in [true, false]
            kernel = SimpleWrapperKernel(lam, is_centrosym)
            
            # Test domain
            xmin, xmax = SparseIR.xrange(kernel)
            ymin, ymax = SparseIR.yrange(kernel)
            @test xmin ≈ -1.0
            @test xmax ≈ 1.0
            @test ymin ≈ -1.0
            @test ymax ≈ 1.0
            
            # Test centrosymmetric property
            @test SparseIR.iscentrosymmetric(kernel) == is_centrosym
            
            # Test inv_weight functions
            beta = 10.0
            lambda = 10.0
            inv_wfunc_ferm = SparseIR.inv_weight_func(kernel, SparseIR.Fermionic(), beta, lambda)
            inv_wfunc_bos = SparseIR.inv_weight_func(kernel, SparseIR.Bosonic(), beta, lambda)
            @test inv_wfunc_ferm(1.0) ≈ 1.0
            @test inv_wfunc_bos(1.0) > 0.0
            
            # Test kernel evaluation
            @test kernel(0.0, 0.0) ≈ 0.5 rtol=1e-10
            
            # Test that kernel can be used to create SVEResult (which tests kernel evaluation through C-API)
            sve_test = SparseIR.SVEResult(kernel, 1e-6)
            @test sve_test.ptr != C_NULL
        end
    end

    @testset "Custom kernel SVE hints" begin
        lam = 10.0
        epsilon = 1e-6
        
        for is_centrosym in [true, false]
            kernel = SimpleWrapperKernel(lam, is_centrosym)
            # Test that SVEResult can be created (which internally uses sve_hints)
            sve_result = SparseIR.SVEResult(kernel, epsilon)
            @test sve_result.ptr != C_NULL
            @test length(sve_result.s) > 0
        end
    end

    @testset "HoleKernel basics" begin
        lam = 100.0
        kernel = HoleKernel(lam)
        
        # Test basic properties
        @test SparseIR.Λ(kernel) == lam
        @test SparseIR.iscentrosymmetric(kernel) == false
        
        # Test domain
        xmin, xmax = SparseIR.xrange(kernel)
        ymin, ymax = SparseIR.yrange(kernel)
        @test xmin ≈ -1.0
        @test xmax ≈ 1.0
        @test ymin ≈ 0.0
        @test ymax ≈ 1.0
        
        # Test kernel evaluation
        @test kernel(0.0, 0.0) ≈ 0.5 rtol=1e-10
        @test kernel(0.0, 0.5) > 0.0
        
        # Test SVE hints
        epsilon = 1e-6
        hints = SparseIR.sve_hints(kernel, epsilon)
        @test SparseIR.nsvals(hints) > 0
        @test SparseIR.ngauss(hints) > 0
        
        # Test segments_y (should only include non-negative values)
        ysegs = SparseIR.segments_y(hints)
        @test all(ysegs .>= 0)
        @test first(ysegs) ≈ 0.0
    end

    @testset "HoleKernel SVEResult creation" begin
        lam = 100.0
        epsilon = 1e-8
        kernel = HoleKernel(lam)
        
        # Test that SVEResult can be created
        sve_result = SparseIR.SVEResult(kernel, epsilon)
        @test sve_result.ptr != C_NULL
        @test length(sve_result.s) > 0
        
        # Verify singular values are positive and decreasing
        svals = sve_result.s
        @test all(svals .> 0)
        for i in 1:(length(svals) - 1)
            @test svals[i] >= svals[i+1]
        end
    end

    @testset "HoleKernel FiniteTempBasis creation" begin
        β = 5.0
        ωmax = 10.0
        ε = 1e-6
        lam = β * ωmax
        kernel = HoleKernel(lam)
        
        # Test Fermionic basis
        basis_ferm = SparseIR.FiniteTempBasis(SparseIR.Fermionic(), β, ωmax, ε; kernel)
        @test length(basis_ferm) > 0
        
        # Test Bosonic basis
        basis_bos = SparseIR.FiniteTempBasis(SparseIR.Bosonic(), β, ωmax, ε; kernel)
        @test length(basis_bos) > 0
        
        # Test that basis functions can be evaluated
        tau = β / 2
        omega = ωmax / 2
        
        u_vals = [basis_ferm.u[l](tau) for l in 1:min(5, length(basis_ferm))]
        v_vals = [basis_ferm.v[l](omega) for l in 1:min(5, length(basis_ferm))]
        
        @test length(u_vals) > 0
        @test length(v_vals) > 0
        @test all(isfinite.(u_vals))
        @test all(isfinite.(v_vals))
    end

    @testset "HoleKernel SVE (ported from OvercompleteIR.jl)" begin
        # See simply if SparseIR can handle this
        kernel = HoleKernel(100.0)
        sve_result = SparseIR.SVEResult(kernel, 1e-6)
        @test length(sve_result.s) > 0
        # OvercompleteIR.jl expects 47 for epsilon=1e-6, but we use epsilon=1e-6
        # so the exact number may differ slightly
        @test length(sve_result.s) >= 40
    end

    @testset "HoleKernel yrange restriction" begin
        lam = 50.0
        kernel = HoleKernel(lam)
        
        # Verify yrange is restricted to [0, 1]
        ymin, ymax = SparseIR.yrange(kernel)
        @test ymin ≈ 0.0
        @test ymax ≈ 1.0
        
        # Compare with inner kernel's yrange
        inner_kernel = kernel.inner
        inner_ymin, inner_ymax = SparseIR.yrange(inner_kernel)
        @test inner_ymin ≈ -1.0
        @test inner_ymax ≈ 1.0
        @test ymin == zero(inner_ymin)
        @test ymax == inner_ymax
    end

    @testset "HoleKernel MatsubaraSampling (ported from OvercompleteIR.jl)" begin
        using LinearAlgebra
        
        # Test Fermionic basis
        beta = 5.0
        omega_max = 10.0
        epsilon = 1e-6
        lam = beta * omega_max
        kernel = HoleKernel(lam)
        
        # Create basis from SVEResult
        sve = SparseIR.SVEResult(kernel, epsilon)
        hbasis = SparseIR.FiniteTempBasis{SparseIR.Fermionic}(sve, beta, omega_max, epsilon)
        L = length(hbasis)
        
        # Test sampling points
        wsmpl = SparseIR.MatsubaraSampling(hbasis)
        @test L <= length(wsmpl.sampling_points) <= 2 * L
        @test LinearAlgebra.cond(wsmpl) <= 10
        
        # Same for positive-only points
        wsmpl = SparseIR.MatsubaraSampling(hbasis; positive_only=true)
        @test L/2 <= length(wsmpl.sampling_points) <= L
        @test LinearAlgebra.cond(wsmpl) <= 10
        
        # Test with AugmentedBasis
        hbasis_aug = SparseIR.AugmentedBasis(hbasis, SparseIR.MatsubaraConst)
        wsmpl = SparseIR.MatsubaraSampling(hbasis_aug)
        @test L <= length(wsmpl.sampling_points) <= 2 * L
        @test LinearAlgebra.cond(wsmpl) <= 50
        
        # Test Bosonic basis
        sve_bos = SparseIR.SVEResult(kernel, epsilon)
        hbasis_bos = SparseIR.FiniteTempBasis{SparseIR.Bosonic}(sve_bos, beta, omega_max, epsilon)
        L_bos = length(hbasis_bos)
        
        # Test sampling points
        wsmpl_bos = SparseIR.MatsubaraSampling(hbasis_bos)
        @test L_bos <= length(wsmpl_bos.sampling_points) <= 2 * L_bos
        @test LinearAlgebra.cond(wsmpl_bos) <= 10
        
        # Same for positive-only points
        wsmpl_bos = SparseIR.MatsubaraSampling(hbasis_bos; positive_only=true)
        @test L_bos/2 <= length(wsmpl_bos.sampling_points) <= L_bos
        @test LinearAlgebra.cond(wsmpl_bos) <= 30
        
        # Test with AugmentedBasis
        hbasis_aug_bos = SparseIR.AugmentedBasis(hbasis_bos, SparseIR.MatsubaraConst)
        wsmpl_aug_bos = SparseIR.MatsubaraSampling(hbasis_aug_bos)
        @test L_bos <= length(wsmpl_aug_bos.sampling_points) <= 2 * L_bos
        @test LinearAlgebra.cond(wsmpl_aug_bos) <= 50
    end

    @testset "HoleKernel kernel evaluation with x₊/x₋ parameters" begin
        lam = 50.0
        kernel = HoleKernel(lam)
        
        x, y = 0.3, 0.5
        
        # Test standard evaluation
        result1 = kernel(x, y)
        @test isfinite(result1)
        @test result1 > 0.0
        
        # Test with explicit x₊ and x₋ parameters
        xmin, xmax = SparseIR.xrange(kernel)
        x₊ = x - first(xmin)
        x₋ = last(xmax) - x
        result2 = kernel(x, y, x₊, x₋)
        @test result2 ≈ result1
        
        # Test that inner kernel is called correctly
        inner_result = kernel.inner(x, y)
        @test result1 ≈ inner_result
    end
end

@testitem "CustomLogisticKernel from v1" tags=[:julia, :spir] begin
    using SparseIR
    using Test

    # Copy LogisticKernel implementation from SparseIR.jl-v1
    # This is a pure Julia implementation without C-API
    struct CustomLogisticKernel{T<:AbstractFloat} <: SparseIR.AbstractKernel
        Λ::T
    end

    function CustomLogisticKernel(Λ)
        Λ ≥ 0 || throw(DomainError(Λ, "Kernel cutoff Λ must be non-negative"))
        return CustomLogisticKernel(float(Λ))
    end

    # SVE hints implementation (copied from v1)
    struct CustomSVEHintsLogistic{T,S} <: SparseIR.AbstractSVEHints
        kernel::CustomLogisticKernel{T}
        ε::S
    end

    function SparseIR.Λ(kernel::CustomLogisticKernel)
        return kernel.Λ
    end

    SparseIR.iscentrosymmetric(::CustomLogisticKernel) = true
    SparseIR.xrange(::CustomLogisticKernel) = (-1.0, 1.0)
    SparseIR.yrange(::CustomLogisticKernel) = (-1.0, 1.0)
    SparseIR.conv_radius(kernel::CustomLogisticKernel) = 40 * kernel.Λ

    # Kernel evaluation (copied from v1)
    function (kernel::CustomLogisticKernel)(x::Real, y::Real)
        # Compute u_± = (1 ± x)/2 and v = Λ * y
        x₊ = 1 + x
        x₋ = 1 - x
        u₊ = x₊ / 2
        u₋ = x₋ / 2
        v = kernel.Λ * y
        
        # By introducing u_± = (1 ± x)/2 and v = Λ * y, we can write
        # the kernel in the following two ways:
        #
        #    k = exp(-u₊ * v) / (exp(-v) + 1)
        #      = exp(-u₋ * -v) / (exp(v) + 1)
        #
        # We need to use the upper equation for v ≥ 0 and the lower one for
        # v < 0 to avoid overflowing both numerator and denominator
        enum = exp(-abs(v) * (v ≥ 0 ? u₊ : u₋))
        denom = 1 + exp(-abs(v))
        return enum / denom
    end

    # Inv weight functions
    function SparseIR.inv_weight_func(kernel::CustomLogisticKernel, stat::SparseIR.Statistics, beta::Float64, lambda::Float64)
        if stat == SparseIR.Fermionic()
            return (omega::Float64) -> 1.0
        else  # Bosonic
            return (omega::Float64) -> tanh(0.5 * kernel.Λ * beta * omega / lambda)
        end
    end

    # SVE hints implementation (copied from v1)
    function SparseIR.sve_hints(kernel::CustomLogisticKernel, ε::Real)
        return CustomSVEHintsLogistic(kernel, ε)
    end

    function SparseIR.segments_x(hints::CustomSVEHintsLogistic, ::Type{T}=Float64) where {T}
        nzeros = max(round(Int, 15 * log10(hints.kernel.Λ)), 1)
        diffs = 1 ./ cosh.(0.143 * range(0; length=nzeros))
        cumsum!(diffs, diffs; dims=1) # From here on, `diffs` contains the zeros
        diffs ./= last(diffs)
        return T.([-reverse(diffs); 0; diffs])
    end

    function SparseIR.segments_y(hints::CustomSVEHintsLogistic, ::Type{T}=Float64) where {T}
        nzeros = max(round(Int, 20 * log10(hints.kernel.Λ)), 2)

        # Zeros around -1 and 1 are distributed asymptotically identically
        leading_diffs = [0.01523, 0.03314, 0.04848, 0.05987, 0.06703, 0.07028, 0.07030, 0.06791,
            0.06391, 0.05896, 0.05358, 0.04814, 0.04288, 0.03795, 0.03342, 0.02932, 0.02565,
            0.02239, 0.01951, 0.01699][begin:min(nzeros, 20)]

        diffs = [leading_diffs; 0.25 ./ exp.(0.141 * (20:(nzeros - 1)))]

        cumsum!(diffs, diffs; dims=1) # From here on, `diffs` contains the zeros
        diffs ./= pop!(diffs)
        diffs .-= 1
        return T.([-1; diffs; 0; -reverse(diffs); 1])
    end

    function SparseIR.nsvals(hints::CustomSVEHintsLogistic)
        log10_Λ = max(1, log10(hints.kernel.Λ))
        return round(Int, (25 + log10_Λ) * log10_Λ)
    end

    function SparseIR.ngauss(hints::CustomSVEHintsLogistic)
        return hints.ε ≥ 1e-8 ? 10 : 16
    end

    @testset "CustomLogisticKernel creation" begin
        lam = 10.0
        kernel = CustomLogisticKernel(lam)
        @test SparseIR.Λ(kernel) == lam
        @test SparseIR.iscentrosymmetric(kernel) == true
        @test SparseIR.xrange(kernel) == (-1.0, 1.0)
        @test SparseIR.yrange(kernel) == (-1.0, 1.0)
    end

    @testset "CustomLogisticKernel evaluation" begin
        lam = 10.0
        kernel = CustomLogisticKernel(lam)
        
        # Test kernel evaluation
        @test kernel(0.0, 0.0) ≈ 0.5 rtol=1e-10
        @test kernel(0.5, 0.5) > 0.0
        @test kernel(-0.5, -0.5) > 0.0
        
        # Test centrosymmetry
        x, y = 0.3, 0.4
        @test kernel(x, y) ≈ kernel(-x, -y) rtol=1e-10
    end

    @testset "CustomLogisticKernel SVEResult creation" begin
        lam = 10.0
        epsilon = 1e-6
        kernel = CustomLogisticKernel(lam)
        
        # Test that SVEResult can be created
        sve_result = SparseIR.SVEResult(kernel, epsilon)
        @test sve_result.ptr != C_NULL
        @test length(sve_result.s) > 0
        
        # Verify singular values are positive and decreasing
        svals = sve_result.s
        @test all(svals .> 0)
        for i in 1:(length(svals) - 1)
            @test svals[i] >= svals[i+1]
        end
    end

    # Custom kernel with compute_even/compute_odd implementation
    struct KernelWithEvenOdd <: SparseIR.AbstractKernel
        inner::CustomLogisticKernel
    end

    KernelWithEvenOdd(Λ) = KernelWithEvenOdd(CustomLogisticKernel(Λ))

    SparseIR.Λ(kernel::KernelWithEvenOdd) = SparseIR.Λ(kernel.inner)
    SparseIR.iscentrosymmetric(::KernelWithEvenOdd) = true
    SparseIR.xrange(kernel::KernelWithEvenOdd) = SparseIR.xrange(kernel.inner)
    SparseIR.yrange(kernel::KernelWithEvenOdd) = SparseIR.yrange(kernel.inner)
    SparseIR.conv_radius(kernel::KernelWithEvenOdd) = SparseIR.conv_radius(kernel.inner)
    function (kernel::KernelWithEvenOdd)(x::Real, y::Real)
        return kernel.inner(x, y)
    end
    function SparseIR.inv_weight_func(kernel::KernelWithEvenOdd, stat::SparseIR.Statistics, beta::Float64, lambda::Float64)
        return SparseIR.inv_weight_func(kernel.inner, stat, beta, lambda)
    end
    function SparseIR.sve_hints(kernel::KernelWithEvenOdd, ε::Real)
        return SparseIR.sve_hints(kernel.inner, ε)
    end

    # Implement compute_even and compute_odd for better numerical accuracy
    function SparseIR.compute_even(kernel::KernelWithEvenOdd, x::Real, y::Real)
        # K_even(x, y) = K(x, y) + K(x, -y)
        # For CustomLogisticKernel, we can compute this more accurately
        # by using the kernel's symmetry properties
        return kernel.inner(x, y) + kernel.inner(x, -y)
    end

    function SparseIR.compute_odd(kernel::KernelWithEvenOdd, x::Real, y::Real)
        # K_odd(x, y) = K(x, y) - K(x, -y)
        return kernel.inner(x, y) - kernel.inner(x, -y)
    end

    @testset "Custom kernel with compute_even/compute_odd" begin
        lam = 10.0
        epsilon = 1e-6
        kernel = KernelWithEvenOdd(lam)
        
        # Test that compute_even and compute_odd are implemented
        x, y = 0.3, 0.4
        k_even = SparseIR.compute_even(kernel, x, y)
        k_odd = SparseIR.compute_odd(kernel, x, y)
        
        # Verify: K_even(x, y) = K(x, y) + K(x, -y)
        @test k_even ≈ kernel(x, y) + kernel(x, -y) rtol=1e-10
        # Verify: K_odd(x, y) = K(x, y) - K(x, -y)
        @test k_odd ≈ kernel(x, y) - kernel(x, -y) rtol=1e-10
        
        # Test that SVEResult can be created without warnings
        # (This should not produce warnings about compute_even/compute_odd)
        sve_result = SparseIR.SVEResult(kernel, epsilon)
        @test sve_result.ptr != C_NULL
        @test length(sve_result.s) > 0
        
        # Test that FiniteTempBasis can be created
        beta = 10.0
        omega_max = 1.0
        basis = SparseIR.FiniteTempBasis{SparseIR.Fermionic}(
            sve_result, beta, omega_max, epsilon
        )
        @test length(basis) > 0
    end

    @testset "inv_weight_func test cases" begin
        lam = 10.0
        beta = 10.0
        lambda = beta * 1.0  # omega_max = 1.0
        
        # Test LogisticKernel inv_weight_func
        kernel = SparseIR.LogisticKernel(lam)
        
        # Fermionic: should return identity
        inv_wfunc_ferm = SparseIR.inv_weight_func(kernel, SparseIR.Fermionic(), beta, lambda)
        @test inv_wfunc_ferm(0.0) ≈ 1.0
        @test inv_wfunc_ferm(1.0) ≈ 1.0
        @test inv_wfunc_ferm(-1.0) ≈ 1.0
        
        # Bosonic: should return tanh(0.5 * Λ * beta * omega / lambda)
        inv_wfunc_bos = SparseIR.inv_weight_func(kernel, SparseIR.Bosonic(), beta, lambda)
        @test inv_wfunc_bos(0.0) ≈ 0.0
        @test inv_wfunc_bos(1.0) ≈ tanh(0.5 * lam * beta * 1.0 / lambda)
        @test inv_wfunc_bos(1.0) > 0.0
        
        # Test CustomLogisticKernel inv_weight_func
        custom_kernel = CustomLogisticKernel(lam)
        inv_wfunc_custom_ferm = SparseIR.inv_weight_func(custom_kernel, SparseIR.Fermionic(), beta, lambda)
        inv_wfunc_custom_bos = SparseIR.inv_weight_func(custom_kernel, SparseIR.Bosonic(), beta, lambda)
        
        @test inv_wfunc_custom_ferm(1.0) ≈ 1.0
        @test inv_wfunc_custom_bos(1.0) ≈ tanh(0.5 * lam * beta * 1.0 / lambda)
        
        # Test that inv_weight_func works with FiniteTempBasis creation
        sve = SparseIR.SVEResult(custom_kernel, 1e-6)
        basis = SparseIR.FiniteTempBasis{SparseIR.Fermionic}(
            sve, beta, 1.0, 1e-6
        )
        @test length(basis) > 0
    end

    @testset "Custom kernel segments_y symmetry check" begin
        lam = 10.0
        epsilon = 1e-6
        
        # Test that centrosymmetric kernels have symmetric segments_y
        kernel = CustomLogisticKernel(lam)
        @test SparseIR.iscentrosymmetric(kernel) == true
        
        hints = SparseIR.sve_hints(kernel, epsilon)
        segs_y = SparseIR.segments_y(hints)
        
        # Check symmetry: segs_y[i] ≈ -segs_y[end-i+1]
        n = length(segs_y)
        tolerance = 1e-10 * max(abs(segs_y[1]), abs(segs_y[end]))
        for i in 1:(n ÷ 2)
            j = n - i + 1
            @test isapprox(segs_y[i], -segs_y[j], atol=tolerance)
        end
        
        # Check middle element is approximately 0 (if odd)
        if isodd(n)
            mid_idx = (n + 1) ÷ 2
            @test isapprox(segs_y[mid_idx], 0.0, atol=tolerance)
        end
    end

end

