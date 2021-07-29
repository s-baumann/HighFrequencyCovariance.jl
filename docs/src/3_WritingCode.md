
# Using HighFrequencyCovariance

## Loading in data

We first load our data into a dataframe. As an example we have a dataframe of price updates (like that in df below).
Then we can put our data into a SortedDataFrame by putting the dataframe and the names of the time, label and value columns into the constructor:
```
using HighFrequencyCovariance
using DataFrames
df = DataFrame(:stock => [:A,:B,:A,:A,:A,:B,:A,:B,:B], :time => [1,2,3,4,5,5,6,7,8],
               :logprice => [1.01,2.0,1.011,1.02,1.011,2.2,1.0001,2.2,2.3])
ts = SortedDataFrame(df, :time, :stock, :logprice)
```
In a real setting this is how we would turn our dataframe of data into a SortedDataFrame.

For the succeeding sections it is useful to get more realistic time series data. So we will generate some Monte Carlo data here using the generate\_random\_path function which generates a random correlation matrix, volatilities, price update times and microstructure noises and generates a SortedDataFrame from a random time series consistent with these.
```
using HighFrequencyCovariance
dims = 4
ticks = 4000
ts, true_covar, true_micro_noise, true_update_rates = generate_random_path(dims, ticks)
```

## Estimating a covariance matrix

As this is a Monte Carlo we already have the true CovarianceMatrix in the true\_covar variable. As we don't have this in applied settings we will disregard this for now and try to estimate it using our data from ts\_data:
```
assets              = get_assets(ts)
simple_estimate     = simple_covariance(ts, assets)
bnhls_estimate      = bnhls_covariance(ts, assets)
spectral_estimate   = spectral_covariance(ts, assets)
preav_estimate      = preaveraged_covariance(ts, assets)
two_scales_estimate = two_scales_covariance(ts, assets)
```
We may alternatively use the estimate\_covariance function:

```
bnhls_estimate2     = estimate_covariance(ts, assets, :BNHLS)
spectral_estimate2  = estimate_covariance(ts, assets, :Spectral)
```

Now we may be particularly interested in one of the estimates, for instance the bnhls estimate. We can first see if the correlation matrix is valid:
```
valid_correlation_matrix(bnhls_estimate)
# true
```
and fortunately it is. We could also examine the others similarly and see that they all deliver valid correlation matrices. One thing we might try then is to average over all of the more advanced methods and use the result as our correlation matrix estimate. This is easy to achieve by using the **combine\_covariance\_matrices** function.
```
matrices = [spectral_estimate, preav_estimate, two_scales_estimate, bnhls_estimate]
combined_estimate = combine_covariance_matrices(matrices)
```

Now we can compare how close each of the estimates is to the true correlation matrix. We can do this by examining the mean absolute difference between estimated correlations.
```
calculate_mean_abs_distance(true_covar, combined_estimate)
# (Correlation_error = 0.16456385637458595, Volatility_error = 0.004544707431296664)
calculate_mean_abs_distance(true_covar, simple_estimate)
# (Correlation_error = 0.448889645637147, Volatility_error = 0.011006475712027963)
calculate_mean_abs_distance(true_covar, bnhls_estimate)
# (Correlation_error = 0.23409341003189574, Volatility_error = 0.001093802689437351)
calculate_mean_abs_distance(true_covar, spectral_estimate)
# (Correlation_error = 0.22234619862055216, Volatility_error = 0.00385672089771947)
calculate_mean_abs_distance(true_covar, preav_estimate)
# (Correlation_error = 0.14664816908395706, Volatility_error = 0.0017134509530432648)
calculate_mean_abs_distance(true_covar, two_scales_estimate)
# (Correlation_error = 0.3577017001321916, Volatility_error = 0.0017134509530432648)
```
We can see that in this particular case the correlation matrix calculated with preaveraging performed the best.

Now examining the data we can see that we have some assets that trade more frequently than the others.
```
ticks_per_asset(ts)
# Dict{Symbol,Int64} with 4 entries:
#  :asset_4 => 6848
#  :asset_3 => 6588
#  :asset_2 => 2630
#  :asset_1 => 3934
```
While we have 6848 price updates for asset\_4 we only have 2630 for asset\_2. Potentially we could improve the bnhls estimate if we use a blocking and regularisation technique (Hautsch, Kyj and Oomen  2012).

We can start this by first making a dataframe detailing what assets should be in what block.
We will generate a new block if the minimum number of ticks of a new block has 20% more ticks than the minimum of the previous:
```
new_block_threshold = 1.2
blocking_frame = put_assets_into_blocks_by_trading_frequency(
                        ts, new_block_threshold, bnhls_covariance)
```
This blocking\_frame is a regular dataframe with six columns where each row represents a different estimation. The order of the rows is the order of estimations (so the results of later estimations may overwrite earlier ones). The first column is named :assets and has the type Set{Symbol} which represents the assets
in each estimation. The second column contains the function that will be used in the estimation of that block. The third column has the name :optional\_parameters and is of type NamedTuple that can provide optional parameters to the covariance function in the second column.
Every covariance estimation has a function signature with two common arguments before the semicolon (For a SortedDataFrame and a vector of symbols representing what assets to use). There can also be a number of named optional arguments which can be sourced from a NamedTuple.
The **blockwise\_estimation** function then estimates a block with the line
```
blocking_frame[i,:f](ts, collect(blocking_frame[i,:assets]);
                     blocking_frame[i,:optional_parameters]... )
```
Thus a user can insert a named tuple containing whatever optional parameters are used by the function.

The fourth, fifth and sixth columns contains the number of assets in the block, the mean number of ticks in the block and the mean time per tick.
These do not do anything in the subsequent **blockwise\_estimation** function but can be used to alter the dataframe.
Now in the current case we may decide to estimate the block containing all assets using the **spectral\_covariance** method.
```
one_asset_row = findall(blocking_frame[:,:number_of_assets] .== 4)
blocking_frame[one_asset_row, :f] = spectral_covariance
```

We can now estimate the blockwise estimated CovarianceMatrix as:
```
block_estimate = blockwise_estimation(ts, blocking_frame)
```
After a blockwise estimation the result may often not be PSD. So we could regularise at this point:
```
reg_block_estimate = nearest_correlation_matrix(block_estimate , ts)
```
Finally we might seek to use one of our estimated **CovarianceMatrix**s to calculate an actual covariance matrix over some interval. This can be done with the code:
```
covariance_interval = 1000
covar = covariance(combined_estimate, covariance_interval)
```
Note that the time units of the covariance\_interval here should be the same units as the CovarianceMatrix struct's volatility which are the same units as the time dimension in the SortedDataFrame that is used to estimate the CovarianceMatrix.
