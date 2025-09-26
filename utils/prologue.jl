using libsparseir_jll

function get_libsparseir()
    # Use debug library if SPARSEIR_LIB_PATH environment variable is set
    if haskey(ENV, "SPARSEIR_LIB_PATH")
        debug_path = ENV["SPARSEIR_LIB_PATH"]
        if isfile(debug_path)
            @info "Using debug library: $debug_path"
            return debug_path
        else
            @warn "Debug library not found at $debug_path, falling back to JLL"
        end
    end
    
    # Production: use JLL package
    return libsparseir_jll.libsparseir
end

const libsparseir = get_libsparseir()
