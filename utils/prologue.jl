include("backend_loader.jl")
using .BackendLoader

const _backend_stamp = BackendLoader.backend_stamp_path()
if isfile(_backend_stamp)
    Base.include_dependency(_backend_stamp)
end

const libsparseir = BackendLoader.require_backend_library()
