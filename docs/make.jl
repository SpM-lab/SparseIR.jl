using SparseIR
using Documenter

# DocMeta.setdocmeta!(SparseIR, :DocTestSetup, :(using SparseIR); recursive=true)

makedocs(;
    # modules=[SparseIR],
    authors="Samuel Badr <samuel.badr@gmail.com> and contributors",
    repo="https://github.com/SpM-lab/SparseIR.jl/blob/{commit}{path}#{line}",
    sitename="SparseIR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
        mathengine=MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/physics"]),
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    "packages" => ["base", "ams", "autoload", "physics"],
                ),
            ),
        ),
    ),
    pages=["Home" => "guide.md"])

# deploydocs(;
#     repo="github.com/SpM-lab/SparseIR.jl.git"
# )
