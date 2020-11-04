using Revise
using UnivariateFunctions
using StochasticIntegrals
using DataFrames
using Dates
using LinearAlgebra
using Distributions: Exponential, MersenneTwister, quantile
using DataStructures: OrderedDict
using Statistics: std, var, mean, cov
using HighFrequencyCovariance
using Sobol
using Random
const tol = 10*eps()

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

# The asyncronous case
update_rates = OrderedDict([:BARC, :HSBC, :VODL, :RYAL] .=> [Exponential(2.0), Exponential(3.0), Exponential(15.0), Exponential(2.5)])
ts = make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, 400000; timing_twister = MersenneTwister(2), ito_twister = MersenneTwister(5))
ts = SortedDataFrame(ts)

at_times = get_all_refresh_times(ts)
dd_compiled = latest_value(ts, at_times)

simp_returns = get_returns(dd_compiled; returns = :simple)
diff_returns = get_returns(dd_compiled; returns = :difference)
log__returns = get_returns(dd_compiled; returns = :log)

naive_covar = rearrange(simple_covariance(diff_returns), brownian_ids)
realised_covar = rearrange(realised_kernal_covariance_estimate(diff_returns), brownian_ids)
