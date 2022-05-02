using SparseIR
using Documenter

DocMeta.setdocmeta!(SparseIR, :DocTestSetup, :(using SparseIR); recursive=true)

makedocs(;
         modules=[SparseIR],
         authors="Samuel Badr <samuel.badr@gmail.com> and contributors",
         repo="https://github.com/Samuel3008/SparseIR.jl/blob/{commit}{path}#{line}",
         sitename="SparseIR.jl",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", "false") == "true",
                                assets=String[]),
         pages=["Home" => "index.md"])
