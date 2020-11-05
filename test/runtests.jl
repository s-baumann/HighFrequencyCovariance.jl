using HighFrequencyCovariance
using Test

# Run tests

println("TestHighFrequencyCovariance")
@time @test include("main_tests_syncronous.jl")
@time @test include("main_tests_asyncronous.jl")
@time @test include("test_31_Higham.jl")
