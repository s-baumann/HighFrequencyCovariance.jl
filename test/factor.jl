using DataFrames
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
twister = MersenneTwister(1)
ts, true_covar, micro_noise, update_rates = generate_random_path(4, 200000;
          brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05],
          twister = deepcopy(twister), syncronous = true)

preav_estimate1 = preaveraged_covariance(ts, assets)
covar = covariance(preav_estimate1,60)

spacing = 5
time_grid = collect(minimum(ts.df[:,ts.time]):spacing:maximum(ts.df[:,ts.time]))

eigenvalues, eigenvectors = eigen(covar)

dd = latest_value(ts, time_grid; assets = assets)

function make_factor(dd, cols, weights)
    return Matrix(dd[:,cols]) * weights
end
dd[!,:fac1] = make_factor(dd, assets, eigenvectors[:,1])
dd[!,:fac2] = make_factor(dd, assets, eigenvectors[:,2])
dd[!,:fac3] = make_factor(dd, assets, eigenvectors[:,3])
dd[!,:fac4] = make_factor(dd, assets, eigenvectors[:,4])

#using DataFrames
#using CSV
#using Plots
#using Lathe
using GLM
#using Statistics
#using StatsPlots
#using MLBase


fm = @formula(BARC  ghvy8y7cu  ~ Adult_Mortality)
linearRegressor = lm(fm, dd)


eigenvalues, eigenvectors = eigen(covar)
reconstituted = construct_matrix_from_eigen(eigenvalues, eigenvectors)

# factors
