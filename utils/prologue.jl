
function get_libsparseir()
    # Use debug library if SPARSEIR_LIB_PATH environment variable is set
    if haskey(ENV, "SPARSEIR_LIB_PATH")
        debug_path = ENV["SPARSEIR_LIB_PATH"]
        print("SPARSEIR_LIB_PATH is set to: $debug_path")
        if !isfile(debug_path)
            error("Debug library not found at $debug_path")
        end
        try
            return Libdl.LazyLibrary(debug_path)
        catch e
            error("Failed to load debug library: $e")
        end
    else
        # Production: use JLL package - load dynamically
        @eval using libsparseir_jll
        return libsparseir_jll.libsparseir
    end
end

const libsparseir = get_libsparseir()