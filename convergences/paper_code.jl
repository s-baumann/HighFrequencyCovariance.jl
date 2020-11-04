
#using Revise

using HighFrequencyCovariance
using Random
using LinearAlgebra


dims = 4
ticks = 40000
twister = MersenneTwister(3)
ts, true_covar , true_micro_noise , true_update_rates =
      generate_random_path(dims, ticks; twister = twister )

assets = unique(ts.df[:,ts.grouping])
simple_estimate = simple_covariance(ts, assets)
bnhls_estimate = bnhls_covariance(ts, assets)
spectral_estimate = spectral_covariance(ts, assets)
preav_estimate = preaveraged_covariance(ts, assets)
two_scales_estimate = two_scales_covariance(ts, assets)

# Now we may be particularly interested in the bnhls estimate. We can first see if the correlation matrix is valid:
inv(bnhls_estimate.correlation)
all(abs.(diag(bnhls_estimate.correlation) .- 1) .< eps())
all(abs.(bnhls_estimate.correlation) .<= 1)
# Unfortunately it is not. We have ended up with some correlations greater than one.

# We can try to map this estimate however to the nearest correlation matrix.
bnhls_nearest_corr = nearest_correlation_matrix(bnhls_estimate , ts)
# We could also try regularsing using with the identity mixing and the eigenvalue cleaning method.
bnhls_estimate_identity = identity_regularisation(bnhls_estimate , ts)
bnhls_estimate_eigen = eigenvalue_clean(bnhls_estimate , ts)
# While these techniques are good at ensuring a correlation matrix is PSD however neither is effective in fixing a correlation matrix with estimated correlations
# greater than 1 in absolute value. So in this case these other techniques are not highly useful.

# Now we can eyeball the coorrelation matrix of the other methods and can see that they all deliver valid correlation matrices. We can try to average over all of these methods and use that as our correlation matrix estimate.
# This is easy to achieve by using the combine_covariance_matrices function.
matrices = [spectral_estimate , preav_estimate , two_scales_estimate, simple_estimate]
combined_estimate = combine_covariance_matrices(matrices)


# Now as this is a monte carlo we can see how close each of the estimates is to the true correlation matrix.
# We can do this by examining the mean absolute difference between estimated correlations.
calculate_mean_abs_distance(true_covar, combined_estimate)
# (Correlation_error = 0.4350180260215464, Volatility_error = 0.003072184143861097)
calculate_mean_abs_distance(true_covar, simple_estimate)
# (Correlation_error = 0.7111531694097382, Volatility_error = 0.0031455466613445936)
calculate_mean_abs_distance(true_covar, bnhls_nearest_corr)
# (Correlation_error = 0.30717874101638376, Volatility_error = 0.0038342012733956284)
calculate_mean_abs_distance(true_covar, spectral_estimate)
# (Correlation_error = 0.8211960904277994, Volatility_error = 0.003931912616700584)
calculate_mean_abs_distance(true_covar, preav_estimate)
# (Correlation_error = 0.051458554354227894, Volatility_error = 0.0026056386486996066)
calculate_mean_abs_distance(true_covar, two_scales_estimate)
# (Correlation_error = 0.1792152720895492, Volatility_error = 0.0026056386486996066)

# We can see that in this particular case the correlation matrix calculated with preaveraging performed the best and then the two scales covariance method.
# Despite not (without regularisation) producing a valid correlation matrix the bnhls's regularised correlation matrix was the third best.

# Now examining the data we can see that we have one asset that trades much more frequently than the others.
assets = unique(ts.df[:,ts.grouping])
ticks_per_asset = map(a -> length(ts.groupingrows[a]), assets)
# While we have 19472 price updates for asset3 we only have 5947 for the second stock.

# We thus might try a blocking and regularisation technique in this case. We can start this by first making a dataframe detailing what assets should be in what block.
# We will generate a new block if the minimum number of ticks of a new block has 20% more ticks than the minimum of the previous:
obs_multiple_for_new_block = 1.2
blocking_frame = put_assets_into_blocks_by_trading_frequency(ts, obs_multiple_for_new_block, bnhls_covariance)
# This blocking_frame is a regular dataframe wtih six columns where each row represents a different estimation. The first column contains Set{Symbol} which represents the assets
# in each estimation. The second column contains the function that will be used in the estimation of that block. The third, fourth and fifth column contains the number of assets in the block, the mean number of ticks in the block and the mean time per tick.
# These do not do anything in the subsequent blockwise\_estimation function but can be used to alter the dataframe. The final column has the name :optional_parameters and contains named tuples.

# Every covariance estimation has a function signature with only two arguments before the semicolon. These are for the SortedDataFrame and a vector of symbol which represent what assets to use. There can also be a number of named optional arguments.
# The blockwise\_estimation function then estimates a block with the line
i = 1
blocking_frame[i,:f](ts, collect(blocking_frame[i,:assets]); blocking_frame[i,:optional_parameters]... )
# Thus a user can insert a named tuple containing whatever optional parameters are used by the function.

# Now in the current case we may decide to always estimate blocks with only one asset by using the two_scales_covariance method (as there are no correlations this is equivalent to using two\_scales\_volatility to estimate that asset's volatility).
blocking_frame[findall(blocking_frame[:,:number_of_assets] .== 1), :f] = two_scales_covariance

# We can now estimate the blockwise estimated covariance matrix as:
block_estimate = blockwise_estimation(ts, blocking_frame)
# After a blockwise estimation the result may often not be PSD. So we could regulaise at this point. As it turns out however the result here is PSD and in addition it is a valid correlation matrix.
# We can also see that the resultant correlation matrix is closer to the true correlation matrix than when we estimated all assets together.
calculate_mean_abs_distance(true_covar, block_estimate)
# (Correlation_error = 0.2747655814595664, Volatility_error = 0.00393163299598061)

function ensemble_covariance(ts, assets)
      functions = [simple_covariance, bnhls_covariance, spectral_covariance,
                      preaveraged_covariance, two_scales_covariance]
      cor_weights = [0.01, 0.1, 0.09, 0.6, 0.2]
      vol_weights = [0.0, 0.1, 0.5, 0.19, 0.21]
      return combine_covariance_matrices(estimates, cor_weights, vol_weights)
end
