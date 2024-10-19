using SparseIR
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(SparseIR, :DocTestSetup, :(using SparseIR))

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

makedocs(; modules=[SparseIR],
    authors="Samuel Badr <samuel.badr@gmail.com> and contributors",
    #  repo="https://github.com/SpM-lab/SparseIR.jl/blob/{commit}{path}#{line}",
    sitename="SparseIR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String["assets/custom.css"],
        mathengine=MathJax3(Dict(:loader => Dict("load" => [
                "[tex]/physics"
            ]),
            :tex => Dict("inlineMath" => [
                    ["\$", "\$"],
                    ["\\(", "\\)"]
                ],
                "tags" => "ams",
                "packages" => [
                    "base",
                    "ams",
                    "autoload",
                    "physics"
                ])))),
    pages=["Home" => "index.md",
        "Guide" => "guide.md",
        "Public" => "public.md",
        "Private" => "private.md"],
    draft=get(ENV, "CI", "false") == "false",
    plugins=[bib])

deploydocs(; repo="github.com/SpM-lab/SparseIR.jl.git")
