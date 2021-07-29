using Documenter, HighFrequencyCovariance

makedocs(
    format = Documenter.HTML(),
    sitename = "HighFrequencyCovariance",
    modules = [HighFrequencyCovariance],
    pages = ["index.md", "1_algorithms.md" , "2_data_structures.md","3_WritingCode.md", "9_references.md"]
)

deploydocs(
    repo   = "github.com/s-baumann/HighFrequencyCovariance.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
    devbranch = "main"
)
