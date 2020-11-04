using HighFrequencyCovariance
using Test

# Run tests

println("TestHighFrequencyCovariance")
@time @test include("tests.jl")
