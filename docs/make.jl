using Documenter, HighFrequencyCovariance

makedocs(
    format = Documenter.HTML(),
    sitename = "HighFrequencyCovariance",
    modules = [HighFrequencyCovariance],
    pages = ["index.md"]
)

deploydocs(
    repo   = "github.com/s-baumann/HighFrequencyCovariance.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)
