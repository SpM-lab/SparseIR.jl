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

    # Weight functions
    SparseIR.weight_func(kernel::SimpleWrapperKernel, stat::SparseIR.Statistics) = 
        SparseIR.weight_func(kernel.inner, stat)

    # SVE hints - delegate to inner kernel
    # Note: LogisticKernel uses C++ implementation, so we need to use the default SVE hints
    # For testing, we'll use the default implementation which should work through the C-API
    SparseIR.sve_hints(kernel::SimpleWrapperKernel, ε::Real) = 
        SparseIR.sve_hints(kernel.inner, ε)

    # HoleKernel: restricts kernel to hole side (y >= 0)
    # This is used to test custom kernels with modified yrange
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

    SparseIR.conv_radius(::HoleKernel) = Inf

    SparseIR.weight_func(self::HoleKernel, stat::SparseIR.Statistics) =
        SparseIR.weight_func(self.inner, stat)

    function (self::HoleKernel)(x, y)
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

    @testset "Custom kernel FiniteTempBasis creation" begin
        β = 2.0
        ωmax = 5.0
        ε = 1e-6
        lam = β * ωmax
        
        for is_centrosym in [true, false]
            kernel = SimpleWrapperKernel(lam, is_centrosym)
            sve_result = SparseIR.SVEResult(kernel, ε)
            
            # Test Fermionic
            basis_ferm = SparseIR.FiniteTempBasis(SparseIR.Fermionic(), β, ωmax, ε; kernel, sve_result)
            @test length(basis_ferm) > 0
            
            # Test Bosonic
            basis_bos = SparseIR.FiniteTempBasis(SparseIR.Bosonic(), β, ωmax, ε; kernel, sve_result)
            @test length(basis_bos) > 0
        end
    end

    @testset "Custom kernel vs LogisticKernel comparison" begin
        β = 2.0
        ωmax = 5.0
        ε = 1e-6
        lam = β * ωmax
        
        # Create LogisticKernel
        log_kernel = SparseIR.LogisticKernel(lam)
        log_sve = SparseIR.SVEResult(log_kernel, ε)
        log_basis_ferm = SparseIR.FiniteTempBasis(SparseIR.Fermionic(), β, ωmax, ε; kernel=log_kernel, sve_result=log_sve)
        
        # Create custom kernels with centrosymmetric = true and false
        for is_centrosym in [true, false]
            custom_kernel = SimpleWrapperKernel(lam, is_centrosym)
            custom_sve = SparseIR.SVEResult(custom_kernel, ε)
            custom_basis_ferm = SparseIR.FiniteTempBasis(SparseIR.Fermionic(), β, ωmax, ε; kernel=custom_kernel, sve_result=custom_sve)
            
            # Compare singular values (should be identical)
            log_svals = log_basis_ferm.s
            custom_svals = custom_basis_ferm.s
            @test length(log_svals) == length(custom_svals)
            @test log_svals ≈ custom_svals rtol=1e-10
            
            # Compare basis functions u at sample points
            tau_points = [0.0, β/4, β/2, 3*β/4, β]
            for tau in tau_points
                x = 2*tau/β - 1.0
                log_u = [log_basis_ferm.u[l](tau) for l in 1:min(10, length(log_basis_ferm))]
                custom_u = [custom_basis_ferm.u[l](tau) for l in 1:min(10, length(custom_basis_ferm))]
                @test length(log_u) == length(custom_u)
                @test log_u ≈ custom_u rtol=1e-10
            end
            
            # Compare basis functions v at sample points
            omega_points = [-ωmax, -ωmax/2, 0.0, ωmax/2, ωmax]
            for omega in omega_points
                log_v = [log_basis_ferm.v[l](omega) for l in 1:min(10, length(log_basis_ferm))]
                custom_v = [custom_basis_ferm.v[l](omega) for l in 1:min(10, length(custom_basis_ferm))]
                @test length(log_v) == length(custom_v)
                @test log_v ≈ custom_v rtol=1e-10
            end
        end
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
            
            # Test weight functions
            wfunc_ferm = SparseIR.weight_func(kernel, SparseIR.Fermionic())
            wfunc_bos = SparseIR.weight_func(kernel, SparseIR.Bosonic())
            @test wfunc_ferm([1.0])[1] ≈ 1.0
            @test wfunc_bos([1.0])[1] > 0.0
            
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
end

