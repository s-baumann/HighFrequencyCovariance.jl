using HighFrequencyCovariance
using Test

# Run tests

println("TestHighFrequencyCovariance")
@time @test include("main_tests.jl")
