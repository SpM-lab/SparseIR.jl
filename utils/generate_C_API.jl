using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Clang.Generators
using Clang.LibClang.Clang_jll

# Function to print help message
function print_help()
    println("Usage: julia build.jl [OPTIONS]")
    println()
    println("Options:")
    println("  --libsparseir-dir PATH    Specify the libsparseir directory path")
    println("  --help, -h                Show this help message")
    println()
    println("Examples:")
    println("  julia build.jl --libsparseir-dir /path/to/libsparseir")
    println()
    println("Default: Uses ../../sparse-ir-rs/sparse-ir-capi relative to this script")
end

# Parse command line arguments
libsparseir_dir = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--help" || arg == "-h"
        print_help()
        exit(0)
    elseif arg == "--libsparseir-dir"
        if i + 1 <= length(ARGS)
            global libsparseir_dir
            libsparseir_dir = ARGS[i + 1]
        else
            println("Error: --libsparseir-dir requires a path argument")
            exit(1)
        end
    end
end

# Get libsparseir directory from command line or use default
if libsparseir_dir === nothing
    # Default path
    libsparseir_dir = normpath(joinpath(@__DIR__, "../../sparse-ir-rs/sparse-ir-capi"))
else
    # Convert to absolute path
    libsparseir_dir = normpath(abspath(libsparseir_dir))
end

# Check if the directory exists
if !isdir(libsparseir_dir)
    println("Error: libsparseir directory not found: $libsparseir_dir")
    println("Please specify the correct path using --libsparseir-dir")
    exit(1)
end

include_dir = joinpath(libsparseir_dir, "include")
sparseir_dir = joinpath(include_dir, "sparseir")

# Check if include directory exists
if !isdir(include_dir)
    println("Error: include directory not found: $include_dir")
    println("Please ensure the libsparseir directory contains an 'include' subdirectory")
    exit(1)
end

println("Using libsparseir directory: $libsparseir_dir")
println("Using include directory: $include_dir")

# wrapper generator options
generator_toml = joinpath(@__DIR__, "generator.toml")
if isfile(generator_toml)
    options = load_options(generator_toml)
else
    println("Warning: generator.toml not found, using default options")
    options = Dict{String,Any}()
end

# add compiler flags, e.g. "-DXXXXXXXXX"
args = get_default_args()
push!(args, "-I$include_dir")

headers = [joinpath(sparseir_dir, header)
           for header in readdir(sparseir_dir) if endswith(header, ".h")]
# there is also an experimental `detect_headers` function for auto-detecting top-level headers in the directory
# headers = detect_headers(sparseir_dir, args)

# create context
ctx = create_context(headers, args, options)

# run generator
build!(ctx)

# Replace line 28 with:
file_path = joinpath(@__DIR__, "../src/C_API.jl")
content = read(file_path, String)
content = replace(content, "const c_complex = ComplexF32" => "const c_complex = ComplexF64")
write(file_path, content)
