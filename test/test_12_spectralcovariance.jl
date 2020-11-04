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
assets = [:BARC, :HSBC, :VODL, :RYAL]
vols = [0.02,0.03,0.04,0.02]
BARC  = ItoIntegral(:BARC, PE_Function(vols[1]))
HSBC  = ItoIntegral(:HSBC, PE_Function(vols[2]))
VODL  = ItoIntegral(:VODL, PE_Function(vols[3]))
RYAL  = ItoIntegral(:RYAL, PE_Function(vols[4]))
ito_integrals = Dict([:BARC, :HSBC, :VODL, :RYAL] .=> [BARC, HSBC, VODL, RYAL])
ito_set_ = ItoSet(brownian_corr_matrix, assets, ito_integrals)

covar = ForwardCovariance(ito_set_, 0.0, 1.0)
stock_processes = Dict([:BARC, :HSBC, :VODL, :RYAL] .=>
                           [ItoProcess(0.0, 180.0, PE_Function(0.00), ito_integrals[:BARC]),
                           ItoProcess(0.0, 360.0, PE_Function(0.00), ito_integrals[:HSBC]),
                           ItoProcess(0.0, 720.0, PE_Function(0.00), ito_integrals[:VODL]),
                           ItoProcess(0.0, 500.0, PE_Function(0.0), ito_integrals[:RYAL])])


# The syncronous case.
spacing = 12.3
ts = make_ito_process_syncronous_time_series(stock_processes, covar,spacing,50000; ito_twister = MersenneTwister(4))
ts = SortedDataFrame(ts)
true_microstructure_variance = 0.01
ts.df[:, :Value] += rand( Normal(0, sqrt(true_microstructure_variance)), nrow(ts.df))

simple_estimate = simple_covariance(ts, assets)
bnhls_estimate = bnhls_covariance_estimate(ts, assets)
spectral_estimate = spectral_covariance(ts, assets)
preav_estimate = preaveraged_HY_covariance(ts, assets)
two_scales_estimate = two_scales_covariance(ts, assets)

simple_vol = simple_volatility(ts)
two_scales_vol, micro_noise = two_scales_volatility(ts)

simple_estimate_identity = identity_regularisation(simple_estimate, ts)
simple_estimate_nearest  = find_nearest_correlation_matrix(simple_estimate, ts)
simple_estimate_eigen    = eigenvalue_clean(simple_estimate, ts)



bnhls_estimate_identity = identity_regularisation(bnhls_estimate, ts)
bnhls_estimate_nearest  = find_nearest_correlation_matrix(bnhls_estimate, ts)
bnhls_estimate_eigen    = eigenvalue_clean(bnhls_estimate, ts)

spectral_estimate_identity = identity_regularisation(spectral_estimate, ts)
spectral_estimate_nearest  = find_nearest_correlation_matrix(spectral_estimate, ts)
spectral_estimate_eigen    = eigenvalue_clean(spectral_estimate, ts)

preav_estimate_identity = identity_regularisation(preav_estimate, ts)
preav_estimate_nearest  = find_nearest_correlation_matrix(preav_estimate, ts)
preav_estimate_eigen    = eigenvalue_clean(preav_estimate, ts)





#############################
update_rates = Dict(assets .=> [Exponential(1.0), Exponential(2.0), Exponential(4.0), Exponential(2.5)])
ts = make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, 200000; timing_twister = MersenneTwister(4), ito_twister = MersenneTwister(6))
ts = SortedDataFrame(ts)
true_microstructure_variance = 0.01
ts.df[:, :Value] += rand( Normal(0, sqrt(true_microstructure_variance)), nrow(ts.df))


obs_multiple_for_new_block = 0.35
blocking_frame = blockwise_estimation_dataframe(ts, obs_multiple_for_new_block, spectral_covariance, (num_blocks = 13,))
covar = blockwise_estimation(ts, blocking_frame)





simple_estimate = simple_covariance(ts, assets)
bnhls_estimate = bnhls_covariance_estimate(ts, assets)
spectral_estimate = spectral_covariance(ts, assets)
preav_estimate = preaveraged_HY_covariance(ts, assets)
two_scales_estimate = two_scales_covariance(ts, assets)

simple_vol = simple_volatility(ts)
two_scales_vol, micro_noise = two_scales_volatility(ts)

simple_estimate_identity = identity_regularisation(simple_estimate, ts)
simple_estimate_nearest  = find_nearest_correlation_matrix(simple_estimate, ts)
simple_estimate_eigen    = eigenvalue_clean(simple_estimate, ts)



bnhls_estimate_identity = identity_regularisation(bnhls_estimate, ts)
bnhls_estimate_nearest  = find_nearest_correlation_matrix(bnhls_estimate, ts)
bnhls_estimate_eigen    = eigenvalue_clean(bnhls_estimate, ts)

spectral_estimate_identity = identity_regularisation(spectral_estimate, ts)
spectral_estimate_nearest  = find_nearest_correlation_matrix(spectral_estimate, ts)
spectral_estimate_eigen    = eigenvalue_clean(spectral_estimate, ts)

preav_estimate_identity = identity_regularisation(preav_estimate, ts)
preav_estimate_nearest  = find_nearest_correlation_matrix(preav_estimate, ts)
preav_estimate_eigen    = eigenvalue_clean(preav_estimate, ts)
