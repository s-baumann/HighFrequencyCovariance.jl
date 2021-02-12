using HighFrequencyCovariance
using Random
using LinearAlgebra

# Making some monte carlo data
dims = 4
ticks = 40000
twister = MersenneTwister(2)
ts_data, true_covar , true_micro_noise , true_update_rates =
      generate_random_path(dims, ticks; twister = twister)

print(first(ts_data.df, 10))
# 10x3 DataFrame
# │ Row │ Time    │ Name    │ Value        │
# │     │ Float64 │ Symbol  │ Float64      │
# ├─────┼─────────┼─────────┼──────────────┤
# │ 1   │ 1.20273 │ asset_4 │ -0.000855704 │
# │ 2   │ 1.26914 │ asset_2 │ 0.0157253    │
# │ 3   │ 1.41356 │ asset_4 │ -0.00427701  │
# │ 4   │ 2.4442  │ asset_1 │ -0.00616605  │
# │ 5   │ 2.5964  │ asset_2 │ 0.0337566    │
# │ 6   │ 2.82763 │ asset_3 │ -0.0187528   │
# │ 7   │ 3.24451 │ asset_3 │ 0.0047703    │
# │ 8   │ 4.11039 │ asset_1 │ -0.00262265  │
# │ 9   │ 4.14443 │ asset_4 │ -0.0792699   │
# │ 10  │ 4.51243 │ asset_1 │ 0.0609101    │

# Making the covariance matrix
assets = unique(ts_data.df[:,ts_data.grouping])
simple_estimate = simple_covariance(ts_data, assets)
bnhls_estimate = bnhls_covariance(ts_data, assets)
spectral_estimate = spectral_covariance(ts_data, assets)
preav_estimate = preaveraged_covariance(ts_data, assets)
two_scales_estimate = two_scales_covariance(ts_data, assets)

# Inspecting the two_scales_estimate CovarianceMatrix.
two_scales_estimate.correlation
# 4×4 Hermitian{Float64,Array{Float64,2}}:
#   1.0        0.997957  -0.989019   0.104429
#   0.997957   1.0       -0.99644    0.167749
#  -0.989019  -0.99644    1.0       -0.250264
#   0.104429   0.167749  -0.250264   1.0
two_scales_estimate.volatility
# 4-element Array{Float64,1}:
#  0.012014556059904117
#  0.008224163665913463
#  0.012810267729395761
#  0.004636468894515986
two_scales_estimate.labels
# 4-element Array{Symbol,1}:
#  :asset_4
#  :asset_2
#  :asset_1
#  :asset_3

two_scales_estimate_rearranged = rearrange(two_scales_estimate,
                                     [:asset_1, :asset_2, :asset_3, :asset_4])

valid_correlation_matrix(bnhls_estimate)
# true

# Now we can eyeball the coorrelation matrix of the other methods and can see that they all deliver valid correlation matrices. We can try to average over all of these methods and use that as our correlation matrix estimate.
# This is easy to achieve by using the combine_covariance_matrices function.
matrices = [spectral_estimate, preav_estimate, two_scales_estimate, bnhls_estimate]
combined_estimate = combine_covariance_matrices(matrices)


# Now as this is a monte carlo we can see how close each of the estimates is to the true correlation matrix.
# We can do this by examining the mean absolute difference between estimated correlations.
calculate_mean_abs_distance(true_covar, combined_estimate)
# (Correlation_error = 0.1740457414286581, Volatility_error = 0.0019898346275229696)
calculate_mean_abs_distance(true_covar, simple_estimate)
# (Correlation_error = 0.2259142303699192, Volatility_error = 0.010991840484883827)
calculate_mean_abs_distance(true_covar, bnhls_estimate)
# (Correlation_error = 0.4074373766699608, Volatility_error = 0.0012992603626814532)
calculate_mean_abs_distance(true_covar, spectral_estimate)
# (Correlation_error = 0.17067853314675663, Volatility_error = 0.0013612065039419734)
calculate_mean_abs_distance(true_covar, preav_estimate)
# (Correlation_error = 0.0888438753129166, Volatility_error = 0.0027198479043355726)
calculate_mean_abs_distance(true_covar, two_scales_estimate)
# (Correlation_error = 0.17531271650289737, Volatility_error = 0.0027198479043355726)

ticks_per_asset(ts_data)
# Dict{Symbol,Int64} with 4 entries:
#   :asset_4 => 13518
#   :asset_3 => 13231
#   :asset_2 => 5277
#   :asset_1 => 7974


# Blocking and Regularisation.
new_block_threshold = 1.2
blocking_frame = put_assets_into_blocks_by_trading_frequency(
                      ts_data, new_block_threshold, bnhls_covariance)
print(blocking_frame[:,1:2])
# 3x2 DataFrame
# │ Row │ assets                                        │ f                │
# │     │ Set{Symbol}                                   │ Function         │
# ├─────┼───────────────────────────────────────────────┼──────────────────┤
# │ 1   │ Set([:asset_3, :asset_4, :asset_2, :asset_1]) │ bnhls_covariance │
# │ 2   │ Set([:asset_2, :asset_1])                     │ bnhls_covariance │
# │ 3   │ Set([:asset_3, :asset_4])                     │ bnhls_covariance │
print(blocking_frame[:,3:6])
# 3x4 DataFrame
# │ Row │ optional_parameters │ number_of_assets │ mean_number_of_ticks │ mean_time_per_tick │
# │     │ NamedTuple…         │ Int64            │ Float64              │ Float64            │
# ├─────┼─────────────────────┼──────────────────┼──────────────────────┼────────────────────┤
# │ 1   │ NamedTuple()        │ 4                │ 10000.0              │ 0.477042           │
# │ 2   │ NamedTuple()        │ 2                │ 6625.5               │ 0.316064           │
# │ 3   │ NamedTuple()        │ 2                │ 13374.5              │ 0.63802            │

one_asset_rows = findall(blocking_frame[:,:number_of_assets] .== 4)
blocking_frame[one_asset_rows, :f] = spectral_covariance
block_estimate = blockwise_estimation(ts_data, blocking_frame)
reg_block_estimate = nearest_correlation_matrix(block_estimate, ts_data)
covariance_interval = 1000
covar = covariance(combined_estimate, covariance_interval)

# financial applications

struct Portfolio{R<:Real}
    weights::Vector{R}
    labels::Vector{Symbol}
end
function portfolio_variance(port::Portfolio, cov::CovarianceMatrix, duration::Real)
    cov2 = rearrange(cov, port.labels)
    return transpose(port.weights) * covariance(cov2, duration) * port.weights
end
our_portfolio = Portfolio([0.25, 0.3, 0.2, 0.25], assets)
duration = 1.0
portfolio_variance(our_portfolio, preav_estimate, duration)
# 1.0031572462484779e-5

duration = 1.0
cov = covariance(preav_estimate, duration)
sigma_1_234 = cov[[1],[2,3,4]]
sigma_234_234 = cov[[2,3,4],[2,3,4]]
weights = sigma_1_234 * inv(sigma_234_234)
hedging_portfolio = -1 * weights
# 1x3 Array{Float64,2}:
# 0.64921
# 0.704808
# 0.547914



# For Monte Carlo tests on accuracy.jl

# For ensemble estimatator code see ensemble.R

function ensemble_covariance(ts, assets, regularisation = eigenvalue_clean)
    functions = [simple_covariance, bnhls_covariance, spectral_covariance,
                preaveraged_covariance, two_scales_covariance]
    cor_weights = [0.01, 0.1, 0.09, 0.6, 0.2]
    vol_weights = [0.0, 0.1, 0.5, 0.19, 0.21]
    estimates = map(f -> f(ts,assets), functions)
    combined = combine_covariance_matrices(estimates, cor_weights, vol_weights)
    return regularisation(combined)
end

# For computational complexity code see accuracy.jl
