@testitem "build support parsing" tags=[:build, :julia] begin
    using Test
    using TOML
    using Libdl: dlext

    include(joinpath(dirname(@__DIR__), "deps", "build_support.jl"))
    using .BuildSupport

    repo_root = dirname(@__DIR__)
    expected_version = BuildSupport.read_backend_version(joinpath(repo_root, "Project.toml"))

    @test BuildSupport.parse_keep_workdir(Dict("SPARSEIR_BUILD_DEBUG" => "1")) === true
    @test BuildSupport.parse_keep_workdir(Dict{String,String}()) === false
    @test BuildSupport.parse_debuginfo(Dict{String,String}()) == "line-tables-only"
    @test BuildSupport.parse_debuginfo(Dict("SPARSEIR_BUILD_DEBUGINFO" => "line")) == "line-tables-only"
    @test BuildSupport.parse_debuginfo(Dict("SPARSEIR_BUILD_DEBUGINFO" => "limited")) == "limited"

    mktempdir() do root
        mkpath(joinpath(root, "deps"))
        @test BuildSupport.backend_stamp_path(root) == joinpath(root, "deps", "backend.stamp")
        @test BuildSupport.select_build_source(root; dev_dir=joinpath(root, "..", "sparse-ir-rs")).kind == :crates_io
    end

    mktempdir() do root
        project_toml = joinpath(root, "Project.toml")
        write(project_toml, """
        [tool.sparseir]
        rust_backend_version = "$expected_version"
        """)
        @test BuildSupport.read_backend_version(project_toml) == expected_version

        BuildSupport.write_build_state!(
            root;
            phase="building-from-crates-io",
            mode="release",
            workspace="/tmp/work",
            debug=false,
            keep_workdir=false,
        )
        state = TOML.parsefile(BuildSupport.build_state_path(root))
        @test state["phase"] == "building-from-crates-io"
        @test state["mode"] == "release"
        @test state["workspace"] == "/tmp/work"
        @test state["debug"] == false
        @test state["keep_workdir"] == false

        BuildSupport.write_backend_stamp!(
            root;
            source="crates-io",
            workspace="/tmp/work",
            version=expected_version,
        )
        stamp = TOML.parsefile(BuildSupport.backend_stamp_path(root))
        @test stamp["source"] == "crates-io"
        @test stamp["workspace"] == "/tmp/work"
        @test stamp["version"] == expected_version
        @test haskey(stamp, "timestamp")
    end

    @test expected_version == BuildSupport.read_backend_version(joinpath(repo_root, "Project.toml"))

    mktempdir() do root
        write(joinpath(root, "Project.toml"), """
        [tool.sparseir]
        rust_backend_version = "$expected_version"
        """)
        dev_dir = joinpath(root, "sparse-ir-rs")
        mkpath(dev_dir)
        plan = BuildSupport.build_plan(root; env=Dict{String,String}(), dev_dir=dev_dir)
        @test plan.source == :local
        @test plan.workspace == dev_dir
        @test plan.version == expected_version
        @test plan.keep_workdir == false
        @test plan.debuginfo == "line-tables-only"
    end

    mktempdir() do root
        write(joinpath(root, "Project.toml"), """
        [tool.sparseir]
        rust_backend_version = "$expected_version"
        """)
        plan = BuildSupport.build_plan(
            root;
            env=Dict(
                "SPARSEIR_BUILD_DEBUG" => "1",
                "SPARSEIR_BUILD_DEBUGINFO" => "full",
            ),
            dev_dir=joinpath(root, "missing"),
        )
        @test plan.source == :crates_io
        @test plan.workspace === nothing
        @test plan.version == expected_version
        @test plan.keep_workdir == true
        @test plan.debuginfo == "full"
    end

    @test BuildSupport.crates_io_download_url(expected_version) ==
        "https://crates.io/api/v1/crates/sparse-ir-capi/$expected_version/download"

    local_plan = (
        root="/tmp/root",
        source=:local,
        workspace="/tmp/dev",
        version=expected_version,
        keep_workdir=false,
        debug=false,
        debuginfo="line-tables-only",
        mode="release",
    )
    @test BuildSupport.crate_dir(local_plan) == joinpath("/tmp/dev", "sparse-ir-capi")
    @test BuildSupport.rustflags(local_plan) == "-C debuginfo=line-tables-only"
    @test BuildSupport.stamp_source(local_plan) == "local-checkout"

    crates_plan = (
        root="/tmp/root",
        source=:crates_io,
        workspace="/tmp/work",
        version=expected_version,
        keep_workdir=true,
        debug=true,
        debuginfo="full",
        mode="release",
    )
    @test BuildSupport.crate_dir(crates_plan) == joinpath("/tmp/work", "sparse-ir-capi-$expected_version")
    @test BuildSupport.rustflags(crates_plan) == "-C debuginfo=full"
    @test BuildSupport.stamp_source(crates_plan) == "crates-io"
    @test BuildSupport.build_workdir(local_plan) == "/tmp/dev"
    @test BuildSupport.build_workdir(crates_plan) == joinpath("/tmp/work", "sparse-ir-capi-$expected_version")
    @test BuildSupport.installed_library_path("/tmp/root") ==
        joinpath("/tmp/root", "deps", "libsparse_ir_capi.$(dlext)")
    @test BuildSupport.built_library_path(local_plan) ==
        joinpath("/tmp/dev", "target", "release", "libsparse_ir_capi.$(dlext)")
    @test BuildSupport.built_library_path(crates_plan) ==
        joinpath("/tmp/work", "sparse-ir-capi-$expected_version", "target", "release", "libsparse_ir_capi.$(dlext)")

    mktempdir() do root
        workspace = joinpath(root, "work")
        plan = (
            root=root,
            source=:crates_io,
            workspace=workspace,
            version=expected_version,
            keep_workdir=false,
            debug=false,
            debuginfo="none",
            mode="release",
        )
        libpath = BuildSupport.built_library_path(plan)
        mkpath(dirname(libpath))
        write(libpath, "stub")
        copied_path = BuildSupport.copy_backend_library!(plan)
        @test copied_path == BuildSupport.installed_library_path(root)
        @test read(copied_path, String) == "stub"
    end

    mktempdir() do root
        plan = (
            root=root,
            source=:crates_io,
            workspace="/tmp/work",
            version=expected_version,
            keep_workdir=true,
            debug=true,
            debuginfo="full",
            mode="release",
        )
        BuildSupport.write_phase!(plan; phase="success")
        state = TOML.parsefile(BuildSupport.build_state_path(root))
        @test state["phase"] == "success"
        @test state["mode"] == "release"
        @test state["workspace"] == "/tmp/work"
        @test state["debug"] == true
        @test state["keep_workdir"] == true
    end
end
