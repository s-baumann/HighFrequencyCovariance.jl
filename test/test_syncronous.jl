using Revise
using UnivariateFunctions
using StochasticIntegrals
using DataFrames
using Distributions
using LinearAlgebra
using Statistics: std, var, mean, cov
using HighFrequencyCovariance
using Random

brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                  0.0 1.0 0.5 0.25;
                                  0.0 0.0 1.0 0.25;
                                  0.0 0.0 0.0 1.0])
brownian_ids = [:BARC, :HSBC, :VODL, :RYAL]

BARC  = ItoIntegral(:BARC, PE_Function(0.02))
HSBC  = ItoIntegral(:HSBC, PE_Function(0.03))
VODL  = ItoIntegral(:VODL, PE_Function(0.04))
RYAL  = ItoIntegral(:RYAL, PE_Function(0.05))
ito_integrals = Dict([:BARC, :HSBC, :VODL, :RYAL] .=> [BARC, HSBC, VODL, RYAL])
ito_set_ = ItoSet(brownian_corr_matrix, brownian_ids, ito_integrals)

covar = ForwardCovariance(ito_set_, 0.0, 1.0)
stock_processes = Dict([:BARC, :HSBC, :VODL, :RYAL] .=>
                           [ItoProcess(0.0, 180.0, PE_Function(0.00), ito_integrals[:BARC]),
                           ItoProcess(0.0, 360.0, PE_Function(0.00), ito_integrals[:HSBC]),
                           ItoProcess(0.0, 720.0, PE_Function(0.00), ito_integrals[:VODL]),
                           ItoProcess(0.0, 500.0, PE_Function(0.0), ito_integrals[:RYAL])])


# The syncronous case.
spacing = 2.345
ts = make_ito_process_syncronous_time_series(stock_processes, covar,spacing,5000; ito_twister = MersenneTwister(3))
ts = SortedDataFrame(ts)
true_microstructure_variance = 0.00001
ts.df[:, :Value] += rand( Normal(0, sqrt(true_microstructure_variance)), nrow(ts.df))

assets = brownian_ids

simple_eigen = syncronised_simple_eigen(ts;
