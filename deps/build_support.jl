module BuildSupport

using Downloads: Downloads
using Libdl: dlext
using Tar: Tar
using TOML: TOML

const LIBSPARSEIR_FILENAME = "libsparse_ir_capi.$(dlext)"

function parse_keep_workdir(env::AbstractDict)
    return get(env, "SPARSEIR_BUILD_DEBUG", "0") == "1"
end

function parse_debuginfo(env::AbstractDict)
    value = get(env, "SPARSEIR_BUILD_DEBUGINFO", "line")
    mapping = Dict(
        "none" => "none",
        "line" => "line-tables-only",
        "line-tables-only" => "line-tables-only",
        "limited" => "limited",
        "full" => "full",
    )
    haskey(mapping, value) || error("Unsupported SPARSEIR_BUILD_DEBUGINFO=$value")
    return mapping[value]
end

backend_stamp_path(root::AbstractString) = joinpath(root, "deps", "backend.stamp")
build_state_path(root::AbstractString) = joinpath(root, "deps", "build-state.toml")
build_log_path(root::AbstractString) = joinpath(root, "deps", "build.log")
installed_library_path(root::AbstractString) = joinpath(root, "deps", LIBSPARSEIR_FILENAME)
generated_c_api_path(root::AbstractString) = joinpath(root, "deps", "C_API.jl")

function read_backend_version(project_toml::AbstractString)
    data = TOML.parsefile(project_toml)
    return data["tool"]["sparseir"]["rust_backend_version"]
end

function write_build_state!(
    root::AbstractString;
    phase,
    mode,
    workspace,
    debug,
    keep_workdir,
)
    mkpath(joinpath(root, "deps"))
    open(build_state_path(root), "w") do io
        TOML.print(io, Dict(
            "phase" => phase,
            "mode" => mode,
            "workspace" => something(workspace, ""),
            "debug" => debug,
            "keep_workdir" => keep_workdir,
        ))
    end
end

function build_state_snapshot(plan; phase)
    return (
        phase=phase,
        mode=plan.mode,
        workspace=plan.workspace,
        debug=plan.debug,
        keep_workdir=plan.keep_workdir,
    )
end

function write_phase!(plan; phase)
    write_build_state!(
        plan.root;
        build_state_snapshot(plan; phase)...,
    )
end

function write_backend_stamp!(
    root::AbstractString;
    source,
    workspace,
    version,
)
    mkpath(joinpath(root, "deps"))
    open(backend_stamp_path(root), "w") do io
        TOML.print(io, Dict(
            "source" => string(source),
            "workspace" => workspace,
            "version" => version,
            "timestamp" => string(time()),
        ))
    end
end

function select_build_source(
    root::AbstractString;
    dev_dir::AbstractString=joinpath(dirname(root), "sparse-ir-rs"),
)
    if isdir(dev_dir)
        return (kind=:local, workspace=dev_dir)
    end
    return (kind=:crates_io, workspace=nothing)
end

function build_plan(
    root::AbstractString;
    env::AbstractDict=ENV,
    dev_dir::AbstractString=joinpath(dirname(root), "sparse-ir-rs"),
)
    source = select_build_source(root; dev_dir)
    keep_workdir = parse_keep_workdir(env)
    return (
        root=root,
        source=source.kind,
        workspace=source.workspace,
        version=read_backend_version(joinpath(root, "Project.toml")),
        keep_workdir=keep_workdir,
        debug=keep_workdir,
        debuginfo=parse_debuginfo(env),
        mode="release",
    )
end

crates_io_download_url(version::AbstractString) =
    "https://crates.io/api/v1/crates/sparse-ir-capi/$version/download"

rustflags(plan) = "-C debuginfo=$(plan.debuginfo)"

function stamp_source(plan)
    if plan.source == :local
        return "local-checkout"
    end
    return "crates-io"
end

function crate_dir(plan)
    if plan.source == :local
        return joinpath(plan.workspace, "sparse-ir-capi")
    end
    return joinpath(plan.workspace, "sparse-ir-capi-$(plan.version)")
end

function build_workdir(plan)
    if plan.source == :local
        return plan.workspace
    end
    return crate_dir(plan)
end

function built_library_path(plan)
    if plan.source == :local
        return joinpath(plan.workspace, "target", plan.mode, LIBSPARSEIR_FILENAME)
    end
    return joinpath(crate_dir(plan), "target", plan.mode, LIBSPARSEIR_FILENAME)
end

function copy_backend_library!(plan)
    source = built_library_path(plan)
    isfile(source) || error("SparseIR backend library not found at $source")

    destination = installed_library_path(plan.root)
    mkpath(dirname(destination))
    cp(source, destination; force=true)
    return destination
end

function cargo_build_cmd(plan, cargo_cmd::Cmd)
    return addenv(
        `$(cargo_cmd) build --manifest-path $(joinpath(crate_dir(plan), "Cargo.toml")) --release --features system-blas`,
        "RUSTFLAGS" => rustflags(plan),
    )
end

function gunzip_cmd(archive::AbstractString)
    gzip = Sys.which("gzip")
    gzip === nothing && error("gzip executable is required to extract crates.io sources")
    return `$gzip -dc $archive`
end

function prepare_workspace!(plan, log_io::IO)
    if plan.source == :local
        return plan
    end

    workspace = mktempdir(; cleanup=false)
    archive, archive_io = mktemp()
    close(archive_io)
    url = crates_io_download_url(plan.version)
    try
        println(log_io, "Downloading crates.io source: $url")
        Downloads.download(url, archive)
        println(log_io, "Extracting crates.io source into $workspace")
        Tar.extract(gunzip_cmd(archive), workspace)
    finally
        rm(archive; force=true)
    end
    return merge(plan, (workspace=workspace,))
end

function regenerate_c_api!(plan, log_io::IO)
    utils_dir = joinpath(plan.root, "utils")
    command = `$(Base.julia_cmd()) --project=. generate_C_API.jl --libsparseir-dir $(crate_dir(plan))`
    println(log_io, "Regenerating Julia C API bindings in $utils_dir")
    cd(utils_dir) do
        run(pipeline(command; stdout=log_io, stderr=log_io))
    end
end

function run_backend_build!(plan, log_io::IO; cargo_cmd::Cmd)
    command = cargo_build_cmd(plan, cargo_cmd)
    println(log_io, "Running build command in $(build_workdir(plan))")
    println(log_io, command)
    cd(build_workdir(plan)) do
        run(pipeline(command; stdout=log_io, stderr=log_io))
    end
    return built_library_path(plan)
end

function cleanup_workspace!(plan; success::Bool)
    if plan.source == :crates_io && plan.workspace !== nothing && success && !plan.keep_workdir
        rm(plan.workspace; recursive=true, force=true)
    end
    return nothing
end

function build_phase(plan)
    if plan.source == :local
        return "building-local-checkout"
    end
    return "building-from-crates-io"
end

function failure_message(root::AbstractString, workspace)
    message = "SparseIR backend build failed. See $(build_log_path(root)) and $(build_state_path(root))."
    if workspace !== nothing
        message *= " Workspace: $workspace."
    end
    return message
end

function main(; root::AbstractString=dirname(@__DIR__), env::AbstractDict=ENV, cargo_cmd::Cmd)
    plan = build_plan(root; env)
    success = false
    mkpath(joinpath(root, "deps"))

    open(build_log_path(root), "w") do log_io
        println(log_io, "SparseIR backend build")
        println(log_io, "source = $(plan.source)")
        println(log_io, "version = $(plan.version)")
        println(log_io, "debug = $(plan.debug)")
        println(log_io, "keep_workdir = $(plan.keep_workdir)")
        println(log_io)
        write_phase!(plan; phase=plan.source == :local ? "using-local-checkout" : "fetching-crates-io")

        try
            plan = prepare_workspace!(plan, log_io)
            write_phase!(plan; phase=build_phase(plan))

            run_backend_build!(plan, log_io; cargo_cmd)
            copy_backend_library!(plan)

            write_phase!(plan; phase="regenerating-c-api")
            regenerate_c_api!(plan, log_io)

            write_backend_stamp!(
                root;
                source=stamp_source(plan),
                workspace=something(plan.workspace, ""),
                version=plan.version,
            )
            write_phase!(plan; phase="success")
            success = true
        catch err
            write_phase!(plan; phase="failed")
            println(log_io)
            showerror(log_io, err, catch_backtrace())
            println(log_io)
            error(failure_message(root, plan.workspace))
        finally
            cleanup_workspace!(plan; success)
        end
    end

    return nothing
end

end
