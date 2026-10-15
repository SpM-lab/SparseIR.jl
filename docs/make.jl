using SparseIR
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(SparseIR, :DocTestSetup, :(using SparseIR))

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

makedocs(; modules=[SparseIR],
    checkdocs=:none,
    authors="SatoshiTerasaki <terasakisatoshi.math@gmail.com>, Samuel Badr <samuel.badr@gmail.com>, Hiroshi Shinaoka <h.shinaoka@gmail.com>, Markus Wallerberger <markus.wallerberger@tuwien.ac.at>",
    repo="https://github.com/SpM-lab/SparseIR.jl/blob/{commit}{path}#{line}",
    sitename="SparseIR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        # The navigation banner shared by the sparse-ir documentation sites;
        # the links are maintained in SpM-lab/SpM-lab.github.io.
        assets=Any["assets/custom.css",
            asset("https://spm-lab.github.io/assets/ecosystem-banner.js"; class=:js)],
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
