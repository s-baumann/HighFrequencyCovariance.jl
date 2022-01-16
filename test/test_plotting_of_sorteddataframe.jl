using DataFrames
using Dates
using LinearAlgebra
using Statistics: std, var, mean, cov
using HighFrequencyCovariance
using Random
using Test

brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                  0.0 1.0 0.5 0.25;
                                  0.0 0.0 1.0 0.25;
                                  0.0 0.0 0.0 1.0])
assets = [:BARC, :HSBC, :VODL, :RYAL]
twister = MersenneTwister(3)
time_period_per_unit = Dates.Hour(1)

ts2, true_covar, micro_noise, update_rates = generate_random_path(4, 50000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], twister = deepcopy(twister), syncronous = true, time_period_per_unit = time_period_per_unit)

plot(ts2)
