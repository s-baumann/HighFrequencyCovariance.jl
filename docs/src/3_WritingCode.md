
# Using HighFrequencyCovariance

## Loading in data

We first load our data into a `DataFrame`. As an example we have a `DataFrame` of price updates (like that in `df` in the below code block).
Then we can put our data into a `SortedDataFrame` by putting the `DataFrame` and the names of the time, label and value columns into the constructor:
```
using HighFrequencyCovariance
using DataFrames
# Making example data
df = DataFrame(:stock => [:A,:B,:A,:A,:A,:B,:A,:B,:B], :time => [1,2,3,4,5,5,6,7,8],
               :logprice => [1.01,2.0,1.011,1.02,1.011,2.2,1.0001,2.2,2.3])
# Making a SortedDataFrame
ts = SortedDataFrame(df, :time, :stock, :logprice)
```
In a real setting this is how we would turn our `DataFrame` of data into a `SortedDataFrame`.

For the succeeding sections it is useful to get more realistic time series data. So we will generate some Monte Carlo data here using the `generate_random_path` function which generates a random correlation matrix, volatilities, price update times and microstructure noises and generates a `SortedDataFrame` from a random time series consistent with these.
```
using HighFrequencyCovariance
using Random
dims = 4
ticks = 10000
ts, true_covar, true_micro_noise, true_update_rates = generate_random_path(dims, ticks, twister = MersenneTwister(2))
```

## Estimating Volatility

We can use the `SortedDataFrame` we have generated (in the `ts` variable) to estimate the volatility of each asset:
```
assets         = get_assets(ts)
simple_vol     = estimate_volatility(ts, assets, :simple_volatility)
two_scales_vol = estimate_volatility(ts, assets, :two_scales_volatility)
```

Now the true volatility is contained in `true_covar.volatility`. We can present these true volatilities alongside the two estimated volatilities

```
using DataFrames
true_volatility = Dict{Symbol,Float64}(true_covar.labels .=> true_covar.volatility)
summary_frame = vcat(DataFrame.([true_volatility, simple_vol, two_scales_vol] )... )
summary_frame = hcat(DataFrame(Dict(:estimation => ["True", "Simple", "2 Scales"])), summary_frame)
print(summary_frame)
# │ Row │ estimation │ asset_1   │ asset_2   │ asset_3    │ asset_4   │
# │     │ String     │ Float64   │ Float64   │ Float64    │ Float64   │
# ├─────┼────────────┼───────────┼───────────┼────────────┼───────────┤
# │ 1   │ True       │ 0.0157077 │ 0.0137856 │ 0.00484516 │ 0.0142265 │
# │ 2   │ Simple     │ 0.0178855 │ 0.0284619 │ 0.00502814 │ 0.0129842 │
# │ 3   │ 2 Scales   │ 0.0173682 │ 0.0192129 │ 0.00605092 │ 0.0149015 │
```
We can see that the accuracy of the simple method was particularly bad for `asset_2`.

This is due to microstructure noise which we can estimate as:
```
noise          = estimate_microstructure_noise(ts, assets)
```
And tabling the estimated and true microstructure noise we can see that there was more
microstructure noise for `asset_2` relative to the other assets.
```
using DataFrames
summary_frame = vcat(DataFrame.([true_micro_noise, noise] )... )
summary_frame = hcat(DataFrame(Dict(:estimation => ["True", "2 Scales noise estimate"])), summary_frame)
print(summary_frame)
# 2×5 DataFrame
# │ Row │ estimation              │ asset_1    │ asset_2    │ asset_3     │ asset_4     │
# │     │ String                  │ Float64    │ Float64    │ Float64     │ Float64     │
# ├─────┼─────────────────────────┼────────────┼────────────┼─────────────┼─────────────┤
# │ 1   │ True                    │ 0.00216696 │ 0.0092135  │ 0.000226909 │ 0.000938589 │
# │ 2   │ 2 Scales noise estimate │ 0.0021294  │ 0.00854816 │ 0.000226053 │ 0.000871175 │
```

## Estimating a covariance matrix

As this is a Monte Carlo we already have the true `CovarianceMatrix` in the `true_covar` variable. As we don't have this in applied settings we will disregard this for now and try to estimate it using our generated tick data stored in the `SortedDataFrame` with name `ts`:
```
assets              = get_assets(ts)
simple_estimate     = estimate_covariance(ts, assets, :simple_covariance)
bnhls_estimate      = estimate_covariance(ts, assets, :bnhls_covariance)
spectral_estimate   = estimate_covariance(ts, assets, :spectral_covariance)
preav_estimate      = estimate_covariance(ts, assets, :preaveraged_covariance)
two_scales_estimate = estimate_covariance(ts, assets, :two_scales_covariance)
```

We may alternatively use the functions corresponding to each method directly. This has the same result:
```
bnhls_estimate2     = bnhls_covariance(ts, assets)
spectral_estimate2  = spectral_covariance(ts, assets)
```

Now we may be particularly interested in one of the estimates, for instance the `bnhls_estimate`. We can first see if the correlation matrix it produces is valid (is positive semi-definite, has a unit diagonal and has all other entries below 1):
```
valid_correlation_matrix(bnhls_estimate)
# true
```
and fortunately it is. We could also examine the others similarly and see that they all deliver valid correlation matrices. One thing we might try then is to average over all of the more advanced methods and use the result as our correlation matrix estimate. This is easy to achieve by using the `combine_covariance_matrices` function.
```
matrices = [spectral_estimate, preav_estimate, two_scales_estimate, bnhls_estimate]
combined_estimate = combine_covariance_matrices(matrices)
```

Now we can compare how close each of the estimates is to the true correlation matrix. We can do this by examining the mean absolute difference between estimated correlations.
```
calculate_mean_abs_distance(true_covar, combined_estimate)
# (Correlation_error = 0.38723691161754376, Volatility_error = 0.002500211816000063)
calculate_mean_abs_distance(true_covar, simple_estimate)
# (Correlation_error = 0.5321534542489482, Volatility_error = 0.010511960080115556)
calculate_mean_abs_distance(true_covar, bnhls_estimate)
# (Correlation_error = 0.7422120933301078, Volatility_error = 0.006815323622470541)
calculate_mean_abs_distance(true_covar, spectral_estimate)
# (Correlation_error = 0.5227424813357473, Volatility_error = 0.007669889385330695)
calculate_mean_abs_distance(true_covar, preav_estimate)
# (Correlation_error = 0.1840684108352901, Volatility_error = 0.0022421828719004925)
calculate_mean_abs_distance(true_covar, two_scales_estimate)
# (Correlation_error = 0.38238270061443486, Volatility_error = 0.0022421828719004925)
```
We can see that in this particular case the correlation matrix calculated with the preaveraging method performed the best.

Now examining the data we can see that we have some assets that trade more frequently than the others.
```
ticks_per_asset(ts)
# Dict{Symbol, Int64} with 4 entries:
#   :asset_4 => 3454
#   :asset_3 => 3242
#   :asset_2 => 1340
#   :asset_1 => 1964
```
While we have 3454 price updates for `asset_4` we only have 1340 for `asset_2`. Potentially we could improve the bnhls estimate if we use a blocking and regularisation technique (Hautsch, Kyj and Oomen  2012).

We can start this by first making a `DataFrame` detailing what assets should be in what block.
We will generate a new block if the minimum number of ticks of a new block has 20% more ticks than the minimum of the previous:
```
new_block_threshold = 1.2
blocking_frame = put_assets_into_blocks_by_trading_frequency(
                        ts, new_block_threshold, :bnhls_covariance)
```
This `blocking_frame` is a regular `DataFrame` with six columns where each row represents a different estimation. The order of the rows is the order of estimations (so the results of later estimations may overwrite earlier ones). The first column is named :assets and has the type `Set{Symbol}` which represents the assets in each estimation. The second column contains a symbol representing the function that will be used in the estimation of that block. The third column has the name :optional\_parameters and is of type `NamedTuple` that can provide optional parameters to the covariance function in the second column.
Every covariance estimation has a function signature with two common arguments before the semicolon (For a `SortedDataFrame` and a vector of symbols representing what assets to use). There can also be a number of named optional arguments which can be sourced from a `NamedTuple`.
The `blockwise_estimation` function then estimates a block with the line
```
blocking_frame[i,:f](ts, collect(blocking_frame[i,:assets]);
                     blocking_frame[i,:optional_parameters]... )
```
Thus a user can insert a named tuple containing whatever optional parameters are used by the function.

The fourth, fifth and sixth columns contains the number of assets in the block, the mean number of ticks in the block and the mean time per tick.
These do not do anything in the subsequent `blockwise_estimation` function but can be used to alter the `DataFrame`.
Now in the current case we may decide to estimate the block containing all assets using the `spectral_covariance` method.
```
one_asset_row = findall(blocking_frame[:,:number_of_assets] .== 4)
blocking_frame[one_asset_row, :f] = :spectral_covariance
```

We can now estimate the blockwise estimated `CovarianceMatrix` as:
```
block_estimate = blockwise_estimation(ts, blocking_frame)
```
After a blockwise estimation the result may often not be PSD. So we could regularise at this point:
```
reg_block_estimate = regularise(block_estimate , ts, :nearest_correlation_matrix)
```
Finally we might seek to use one of our estimated `CovarianceMatrix`s to calculate an actual covariance matrix over some interval. This can be done with the code:
```
covariance_interval = 1000
covar = covariance(combined_estimate, covariance_interval)
```
Note that the time units of the covariance\_interval here should be the same units as the `CovarianceMatrix` struct's volatility which are the same units as the time dimension in the `SortedDataFrame` that is used to estimate the `CovarianceMatrix`.
